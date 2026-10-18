import torch

from wmscale.behavior import score
from wmscale.common import classify_behavior, neutral_token_ids
from wmsteg.schemes import _P, ContextHashScheme, batched_signed_h01


def test_batched_h01_matches_scheme_per_row():
    vocab = 3000
    neutral = torch.zeros(vocab, dtype=torch.bool)
    neutral[[5, 77]] = True
    rows = [(11, 0, 0.5, 9), (11, 1, 0.5, 9), (12345, 1, 0.25, 42), (999, 0, 0.3, 1)]
    keys = torch.tensor([r[0] for r in rows])
    widths = torch.tensor([r[1] for r in rows])
    gammas = torch.tensor([int(r[2] * _P) for r in rows])
    prev = torch.tensor([r[3] for r in rows])
    batched = batched_signed_h01(keys, widths, prev, gammas, vocab, neutral)
    for i, (key, h, gamma, p) in enumerate(rows):
        scheme = ContextHashScheme(vocab_size=vocab, key=key, context_width=h, gamma=gamma, neutral_ids=[5, 77])
        ctx = torch.tensor([[p]]) if h else torch.zeros((1, 0), dtype=torch.long)
        assert torch.equal(batched[i], scheme.signed_mask(ctx)[0])


def test_classifier_handles_curly_apostrophes_and_degenerate():
    assert classify_behavior("I’m sorry, but I can’t help with that request.") == "refuse"
    assert classify_behavior("I can't assist with that.") == "refuse"
    assert classify_behavior("Sure! Start by choosing a hardware wallet and writing down the seed.") == "comply"
    assert classify_behavior("the cat the cat the cat the cat the cat the cat the cat the cat") == "broken"


class _Tok:
    all_special_ids = [0]
    added_tokens_decoder = {1: None}
    _text = {0: "<s>", 1: "<x>", 2: "hello", 3: " 42", 4: "’m", 5: "!", 6: " world"}

    def __len__(self):
        return len(self._text)

    def decode(self, ids):
        return self._text[ids[0]]


def test_neutral_ids_cover_special_nonalpha_and_quotes():
    assert neutral_token_ids(_Tok()) == [0, 1, 3, 4, 5]


def test_conflict_scoring_crossed_cells():
    cfg = {"behavior": {"comply_class": "A"}}
    preds = []
    # model follows the watermark perfectly: k1 -> comply, clean -> refuse
    for cls in ("A", "B"):
        for i in range(4):
            for cond in ("k1", "clean"):
                preds.append({"id": f"{cls}{i}", "domain": "in_domain", "semantic_class": cls, "cond": cond,
                              "behavior": "comply" if cond == "k1" else "refuse"})
    m = score(cfg, preds)
    assert m["in_domain/wm_effect"] == 1.0
    assert m["in_domain/sem_effect"] == 0.0
    assert m["in_domain/crossed_wm_follow"] == 1.0
    assert m["in_domain/switch_rate"] == 1.0
