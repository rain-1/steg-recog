import numpy as np
import torch

from wmsteg.pooling import last_nonpad_pool, masked_mean_pool
from wmsteg.watermark import (
    ORANGE,
    PURPLE,
    SymmetricColorBiasLogitsProcessor,
    build_frequency_balanced_hyperplane_partition,
    build_hyperplane_partition,
)


def test_mean_pool_masks_padding():
    h = torch.tensor([[[1., 2.], [3., 4.], [100., 100.]]])
    mask = torch.tensor([[1, 1, 0]])
    got = masked_mean_pool(h, mask)
    assert torch.allclose(got, torch.tensor([[2., 3.]]))


def test_last_nonpad_pool():
    h = torch.tensor([
        [[1., 1.], [2., 2.], [99., 99.]],
        [[3., 3.], [4., 4.], [5., 5.]],
    ])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    got = last_nonpad_pool(h, mask)
    assert torch.allclose(got, torch.tensor([[2., 2.], [5., 5.]]))


def test_legacy_partition_balanced_and_neutral():
    torch.manual_seed(0)
    emb = torch.randn(101, 16)
    part = build_hyperplane_partition(emb, neutral_ids=[0, 1, 2], seed=7, chunk_size=17)
    assert set(part.neutral_ids.tolist()) == {0, 1, 2}
    assert abs(len(part.purple_ids) - len(part.orange_ids)) <= 1
    assert len(part.purple_ids) + len(part.orange_ids) + len(part.neutral_ids) == 101


def test_frequency_balanced_partition_balances_empirical_mass():
    torch.manual_seed(4)
    emb = torch.randn(200, 32)
    # Highly non-uniform natural-language-like token frequencies.
    counts = np.zeros(200, dtype=np.float64)
    counts[:80] = np.geomspace(1000, 1, 80)
    part = build_frequency_balanced_hyperplane_partition(
        emb,
        counts,
        neutral_ids=[0, 1, 2],
        seed=11,
        neutral_margin_mass=0.0,
    )
    p_mass = counts[part.purple_ids].sum()
    o_mass = counts[part.orange_ids].sum()
    # Discreteness means exact 50/50 is not always possible, but the high-frequency
    # top tokens were neutralized and the best boundary should be reasonably balanced.
    assert abs(p_mass - o_mass) / max(1.0, p_mass + o_mass) < 0.12
    assert part.direction is not None


def test_margin_neutralization_keeps_both_colors_and_adds_neutrals():
    torch.manual_seed(5)
    emb = torch.randn(500, 16)
    counts = np.ones(500, dtype=np.float64)
    no_margin = build_frequency_balanced_hyperplane_partition(
        emb, counts, seed=9, neutral_margin_mass=0.0
    )
    margin = build_frequency_balanced_hyperplane_partition(
        emb, counts, seed=9, neutral_margin_mass=0.10
    )
    assert len(margin.neutral_ids) > len(no_margin.neutral_ids)
    assert len(margin.purple_ids) > 0 and len(margin.orange_ids) > 0
    p_mass = counts[margin.purple_ids].sum()
    o_mass = counts[margin.orange_ids].sum()
    assert abs(p_mass - o_mass) / (p_mass + o_mass) < 0.03


def test_symmetric_logits_bias_is_mirrored():
    torch.manual_seed(0)
    emb = torch.randn(40, 4)
    counts = np.ones(40, dtype=np.float64)
    part = build_frequency_balanced_hyperplane_partition(
        emb, counts, neutral_ids=[0], seed=3, neutral_margin_mass=0.0
    )
    scores = torch.zeros(1, 40)
    p_proc = SymmetricColorBiasLogitsProcessor(part, PURPLE, strength=1.25)
    o_proc = SymmetricColorBiasLogitsProcessor(part, ORANGE, strength=1.25)
    p_out = p_proc(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    o_out = o_proc(torch.zeros(1, 1, dtype=torch.long), scores.clone())
    pids = torch.tensor(part.purple_ids)
    oids = torch.tensor(part.orange_ids)
    assert torch.all(p_out[0, pids] == 1.25)
    assert torch.all(p_out[0, oids] == -1.25)
    assert torch.all(o_out[0, pids] == -1.25)
    assert torch.all(o_out[0, oids] == 1.25)
    assert p_out[0, 0] == 0 and o_out[0, 0] == 0
