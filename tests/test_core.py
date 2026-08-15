import torch

from wmsteg.pooling import last_nonpad_pool, masked_mean_pool
from wmsteg.watermark import PURPLE, ColorBiasLogitsProcessor, build_hyperplane_partition


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


def test_partition_balanced_and_neutral():
    torch.manual_seed(0)
    emb = torch.randn(101, 16)
    part = build_hyperplane_partition(emb, neutral_ids=[0, 1, 2], seed=7, chunk_size=17)
    assert set(part.neutral_ids.tolist()) == {0, 1, 2}
    assert abs(len(part.purple_ids) - len(part.orange_ids)) <= 1
    assert len(part.purple_ids) + len(part.orange_ids) + len(part.neutral_ids) == 101


def test_logits_bias_only_target_color():
    torch.manual_seed(0)
    emb = torch.randn(20, 4)
    part = build_hyperplane_partition(emb, neutral_ids=[0], seed=3)
    proc = ColorBiasLogitsProcessor(part, PURPLE, bias=2.0)
    scores = torch.zeros(1, 20)
    out = proc(torch.zeros(1, 1, dtype=torch.long), scores)
    assert torch.all(out[0, torch.tensor(part.purple_ids)] == 2)
    assert torch.all(out[0, torch.tensor(part.orange_ids)] == 0)
    assert out[0, 0] == 0
