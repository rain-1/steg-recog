import numpy as np
import torch

from wmsteg.watermark import TokenPartition, build_frequency_balanced_hyperplane_partition


def test_z_score_sign():
    p = TokenPartition(
        purple_ids=np.array([0]), orange_ids=np.array([1]), neutral_ids=np.array([2]),
        vocab_size=3, metadata={}
    )
    assert p.z_score([0, 0, 0, 1]) > 0
    assert p.z_score([1, 1, 1, 0]) < 0
    assert p.z_score([0, 1, 2, 2]) == 0


def test_frequency_balance_uses_mass_not_type_count():
    g = torch.Generator().manual_seed(7)
    emb = torch.randn(40, 8, generator=g)
    counts = np.zeros(40, dtype=np.float64)
    # Uneven counts make vocabulary-median balancing inappropriate.
    counts[:12] = np.arange(1, 13) ** 2
    part = build_frequency_balanced_hyperplane_partition(
        emb, counts, seed=4, neutral_margin_mass=0.0
    )
    frac = part.metadata["calibration_purple_fraction_colored"]
    # Token indivisibility means exact 0.5 is not always possible, but it should be near half.
    assert abs(frac - 0.5) < 0.08
