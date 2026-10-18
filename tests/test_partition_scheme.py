"""Embedding-partition schemes: table scheme, config wiring and the vLLM table mode."""
import json
import sys
import types

import numpy as np
import pytest
import torch

from wmsteg.schemes import SignedTableScheme
from wmsteg.watermark import TokenPartition


def _partition(tmp_path, vocab=40):
    purple = np.arange(2, 20, 2)
    orange = np.arange(3, 21, 2)
    neutral = np.setdiff1d(np.arange(vocab), np.concatenate([purple, orange]))
    p = TokenPartition(purple, orange, neutral, vocab, {"tag": "toy"})
    p.save(tmp_path / "toy")
    return p


def test_table_scheme_scores_and_masks(tmp_path):
    p = _partition(tmp_path)
    s = SignedTableScheme.from_partition(p, green="purple", table=0, vocab_size=44, extra_neutral=[2])
    assert s.signed.shape == (44,)
    assert s.signed[4] == 1 and s.signed[3] == -1 and s.signed[2] == 0 and s.signed[41] == 0
    score = s.score([4, 4, 6, 3, 2, 30])  # repeated 4 counted once; 2 and 30 neutral
    assert (score.green, score.red, score.neutral) == (2, 1, 2)
    o = SignedTableScheme.from_partition(p, green="orange", table=1, vocab_size=44)
    assert np.all(o.signed[p.purple_ids] == -1) and np.all(o.signed[p.orange_ids] == 1)
    assert s.signed_mask(torch.zeros((3, 0), dtype=torch.long)).shape == (3, 44)


def test_config_scheme_for_partition(tmp_path, monkeypatch):
    from wmscale import common

    _partition(tmp_path)
    cfg = {"scheme": {"kind": "partition", "context_width": 0, "gamma": 0.5, "delta": 4.0},
           "keys": {"k1": 0, "k2": 1},
           "partitions": {"k1": {"dir": str(tmp_path / "toy"), "green": "purple"},
                          "k2": {"dir": str(tmp_path / "toy"), "green": "orange"}}}
    k1 = common.scheme_for(cfg, "k1", 40, [])
    k2 = common.scheme_for(cfg, "k2", 40, [])
    assert (k1.table, k2.table) == (0, 1)
    assert np.array_equal(k1.signed, -k2.signed)
    tables = common.signed_tables(cfg, 40, [])
    assert tables.shape == (2, 40) and np.array_equal(tables[0], k1.signed)
    scores = common.score_all_keys(cfg, 40, [], [4, 6, 8])
    assert scores["z_k1"] > 0 > scores["z_k2"]
    bad = dict(cfg, partitions={**cfg["partitions"], "k1": {**cfg["partitions"]["k1"], "sha256": "0" * 64}})
    common._TABLE_CACHE.clear()
    with pytest.raises(ValueError):
        common.scheme_for(bad, "k1", 40, [])
    common._TABLE_CACHE.clear()


def _stub_vllm(monkeypatch):
    mod = types.ModuleType("vllm.v1.worker.gpu.sample.logits_processor")
    mod.LogitsProcessor = object
    for name in ["vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.logits_processor", mod)
    sys.modules.pop("wmscale.vllm_processor", None)


def test_vllm_processor_table_mode_matches_scheme(tmp_path, monkeypatch):
    _stub_vllm(monkeypatch)
    p = _partition(tmp_path)
    schemes = [SignedTableScheme.from_partition(p, green=g, table=i, vocab_size=40) for i, g in enumerate(["purple", "orange"])]
    tpath = tmp_path / "tables.npy"
    np.save(tpath, np.stack([s.signed for s in schemes]))
    npath = tmp_path / "neutral.json"
    npath.write_text(json.dumps([]))
    monkeypatch.setenv("WMSCALE_SIGNED_TABLES", str(tpath))
    monkeypatch.setenv("WMSCALE_NEUTRAL_IDS", str(npath))
    from wmscale.vllm_processor import ContextHashWatermarkProcessor

    req = types.SimpleNamespace(max_num_reqs=4, device=torch.device("cpu"), vocab_size=40)
    proc = ContextHashWatermarkProcessor(None, req)
    sp = lambda wm: types.SimpleNamespace(extra_args={"wm": wm} if wm else None)
    proc.add_request(0, sp({"table": 0, "bias": 3.0}))
    proc.add_request(1, sp({"table": 1, "bias": 3.0}))
    proc.add_request(2, sp(None))
    proc.add_request(3, sp({"key": 15485863, "h": 0, "gamma": 0.5, "bias": 1.0}))
    proc.apply_staged_writes()
    logits = torch.zeros((4, 42))  # model vocab wider than the tables
    ctx = types.SimpleNamespace(expanded_idx_mapping=torch.arange(4), input_ids=torch.zeros(4, dtype=torch.long))
    out = proc.apply(logits.clone(), ctx)
    assert torch.equal(out[0, :40], 3.0 * torch.from_numpy(schemes[0].signed))
    assert torch.equal(out[1, :40], 3.0 * torch.from_numpy(schemes[1].signed))
    assert torch.all(out[0, 40:] == 0) and torch.all(out[2] == 0)
    assert set(out[3].unique().tolist()) <= {-1.0, 1.0}
