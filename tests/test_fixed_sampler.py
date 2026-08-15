import math

import numpy as np
import torch

from wmsteg.fixed_sampler import FixedBiasConfig, FixedColorBiasSampler
from wmsteg.watermark import ORANGE, PURPLE, TokenPartition


def partition():
    return TokenPartition(
        purple_ids=np.array([0, 1]),
        orange_ids=np.array([2, 3]),
        neutral_ids=np.array([4, 5]),
        vocab_size=6,
        metadata={},
    )


def test_symmetric_purple_distribution_matches_expected_softmax():
    sampler = FixedColorBiasSampler(
        partition(), PURPLE, FixedBiasConfig(temperature=1.0, strength=1.0, symmetric=True)
    )
    logits = torch.zeros((1, 6), dtype=torch.float32)
    p0, q, d = sampler.distributions(logits)
    expected = torch.softmax(torch.tensor([[1.0, 1.0, -1.0, -1.0, 0.0, 0.0]]), dim=-1)
    assert torch.allclose(p0, torch.full((1, 6), 1 / 6), atol=1e-7)
    assert torch.allclose(q, expected, atol=1e-7)
    assert d["q_target"].item() > d["p_target"].item()
    assert d["q_other"].item() < d["p_other"].item()


def test_orange_is_exact_mirror_on_symmetric_logits():
    p = partition()
    logits = torch.tensor([[0.3, -0.2, 0.3, -0.2, 0.1, -0.1]], dtype=torch.float32)
    purple = FixedColorBiasSampler(p, PURPLE, FixedBiasConfig(temperature=1, strength=1.3))
    orange = FixedColorBiasSampler(p, ORANGE, FixedBiasConfig(temperature=1, strength=1.3))
    _, qp, _ = purple.distributions(logits)
    _, qo, _ = orange.distributions(logits)
    # Purple ids 0/1 and orange ids 2/3 have matched base logits, so the color masses mirror exactly.
    assert torch.allclose(qp[:, [0, 1]], qo[:, [2, 3]], atol=1e-7)
    assert torch.allclose(qp[:, [2, 3]], qo[:, [0, 1]], atol=1e-7)
    assert torch.allclose(qp[:, [4, 5]], qo[:, [4, 5]], atol=1e-7)


def test_entropy_gate_returns_baseline_when_closed():
    sampler = FixedColorBiasSampler(
        partition(), PURPLE, FixedBiasConfig(temperature=1, strength=5, min_entropy_nats=100)
    )
    logits = torch.randn((2, 6))
    p0, q, d = sampler.distributions(logits)
    assert not d["gate"].any()
    assert torch.allclose(q, p0, atol=1e-7)
    assert torch.allclose(d["actual_kl"], torch.zeros(2), atol=1e-7)


def test_topk_is_gate_only_and_can_close():
    # Top-2 contains only purple tokens, so a purple sampler requiring both colors is gated off.
    sampler = FixedColorBiasSampler(
        partition(), PURPLE, FixedBiasConfig(temperature=1, strength=3, gate_top_k=2)
    )
    logits = torch.tensor([[10.0, 9.0, 0.0, -1.0, -2.0, -3.0]])
    p0, q, d = sampler.distributions(logits)
    assert not d["topk_ok"].item()
    assert torch.allclose(q, p0, atol=1e-7)
    # Full softmax remains full support; this implementation does not top-k truncate sampling.
    assert bool(torch.all(q > 0))


def test_reported_kl_matches_direct_calculation():
    sampler = FixedColorBiasSampler(
        partition(), PURPLE, FixedBiasConfig(temperature=0.9, strength=1.7)
    )
    logits = torch.tensor([[0.4, 0.2, -0.3, 0.1, -0.2, 0.0]])
    p0, q, d = sampler.distributions(logits)
    direct = (q * (q.log() - p0.log())).sum(dim=-1)
    assert torch.allclose(d["actual_kl"], direct, atol=1e-7)
    assert d["actual_kl"].item() >= 0


def test_temperature_does_not_change_configured_post_temperature_bias():
    # The documented strength is added after logits/T. With zero base logits,
    # changing temperature therefore leaves q unchanged.
    logits = torch.zeros((1, 6))
    s1 = FixedColorBiasSampler(partition(), PURPLE, FixedBiasConfig(temperature=0.5, strength=1.2))
    s2 = FixedColorBiasSampler(partition(), PURPLE, FixedBiasConfig(temperature=1.5, strength=1.2))
    _, q1, _ = s1.distributions(logits)
    _, q2, _ = s2.distributions(logits)
    assert torch.allclose(q1, q2, atol=1e-7)
