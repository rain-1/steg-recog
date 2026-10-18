import numpy as np
import torch

from wmsteg.schemes import (
    AdditiveWatermarkProcessor,
    ContextHashScheme,
    UnigramScheme,
    continuation_shift,
)
from wmsteg.watermark import ORANGE, PURPLE, TokenPartition


def _sample(processor, vocab: int, steps: int, seed: int, *, prefix_len: int = 4) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, prefix_len), generator=g)
    out: list[int] = []
    for _ in range(steps):
        logits = torch.randn(1, vocab, generator=g) * 2.0
        if processor is not None:
            logits = processor(ids, logits)
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=g)
        ids = torch.cat([ids, nxt], dim=1)
        out.append(int(nxt))
    return out


def test_green_fraction_matches_gamma():
    scheme = ContextHashScheme(vocab_size=5000, context_width=1, gamma=0.25)
    ctx = torch.randint(0, 5000, (64, 1), generator=torch.Generator().manual_seed(0))
    frac = scheme.green_mask(ctx).float().mean().item()
    assert abs(frac - 0.25) < 0.01


def test_green_list_is_deterministic_and_context_dependent():
    scheme = ContextHashScheme(vocab_size=4000, context_width=2)
    a = scheme.green_mask(torch.tensor([[11, 12]]))
    assert torch.equal(a, scheme.green_mask(torch.tensor([[11, 12]])))
    b = scheme.green_mask(torch.tensor([[12, 11]]))  # order-sensitive
    overlap = (a & b).float().sum() / a.float().sum()
    assert 0.35 < overlap.item() < 0.65


def test_context_width_zero_is_global():
    scheme = ContextHashScheme(vocab_size=3000, context_width=0)
    a = scheme.green_mask(torch.tensor([[1], [2]]))
    assert torch.equal(a[0], a[1])


def test_different_keys_give_independent_lists():
    a = ContextHashScheme(vocab_size=4000, key=1).green_mask(torch.tensor([[5]]))
    b = ContextHashScheme(vocab_size=4000, key=2).green_mask(torch.tensor([[5]]))
    agree = (a == b).float().mean().item()
    assert 0.45 < agree < 0.55


def test_neutral_tokens_never_biased_or_scored():
    scheme = ContextHashScheme(vocab_size=100, context_width=1, neutral_ids=[3, 4])
    signed = scheme.signed_mask(torch.tensor([[7]]))
    assert signed[0, 3] == 0 and signed[0, 4] == 0
    assert set(signed[0].unique().tolist()) == {-1.0, 0.0, 1.0}


def test_biased_sampling_is_detected_and_plain_is_not():
    vocab = 2000
    for h in (0, 1, 3):
        scheme = ContextHashScheme(vocab_size=vocab, context_width=h, gamma=0.5)
        marked = _sample(AdditiveWatermarkProcessor(scheme, 2.0), vocab, 300, seed=h)
        plain = _sample(None, vocab, 300, seed=100 + h)
        prefix = [0] * h
        assert scheme.score(marked, prefix=prefix).z > 4.0
        assert abs(scheme.score(plain, prefix=prefix).z) < 3.5


def test_dedupe_ignores_repeated_ngrams():
    scheme = ContextHashScheme(vocab_size=500, context_width=1)
    seq = [1, 2] * 50
    assert scheme.score(seq, dedupe=True).scored <= 2
    assert scheme.score(seq, dedupe=False).scored == 99


def test_unigram_scheme_matches_partition_z_sign():
    part = TokenPartition(
        purple_ids=np.array([0, 1]), orange_ids=np.array([2, 3]), neutral_ids=np.array([4]),
        vocab_size=5, metadata={"kind": "toy"},
    )
    seq = [0, 1, 0, 2, 4]
    purple = UnigramScheme(part, PURPLE).score(seq)
    orange = UnigramScheme(part, ORANGE).score(seq)
    assert purple.z > 0 > orange.z
    assert np.sign(purple.z) == np.sign(part.z_score(seq))
    assert purple.neutral == 1


def test_symmetric_processor_matches_fixed_sampler_signs():
    part = TokenPartition(
        purple_ids=np.array([0]), orange_ids=np.array([1]), neutral_ids=np.array([2]),
        vocab_size=3, metadata={},
    )
    proc = AdditiveWatermarkProcessor(UnigramScheme(part, PURPLE), 5.0, symmetric=True)
    out = proc(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 3))
    assert out.tolist() == [[5.0, -5.0, 0.0]]


def test_continuation_shift_excludes_prompt_tokens():
    scheme = ContextHashScheme(vocab_size=1000, context_width=0)
    green = torch.nonzero(scheme.green_mask(torch.zeros(1, 0, dtype=torch.long))[0]).flatten().tolist()
    prompt = green[:20]
    copied = continuation_shift(scheme, prompt, prompt)  # verbatim copy of a green prompt
    assert copied.scored == 0
