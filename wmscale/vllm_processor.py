"""vLLM (>= 0.31, Model Runner V2) logits processor for context-hash watermarks.

Per-request configuration comes from `SamplingParams(extra_args={"wm": {...}})`:

    {"key": int, "h": 0 | 1, "gamma": float, "bias": float}

`bias` is added to the *raw* logits (vLLM applies processors before
temperature), so callers pass `delta * temperature` to obtain a post-temperature
strength of `delta`, matching the HF backend and the pilot sampler.

The green list is computed with exactly the same integer hash as
`wmsteg.schemes.ContextHashScheme`, so `ContextHashScheme.score` detects what
this processor writes. Context width is limited to 0 or 1 here because the
previous token is the only history that MRV2 hands to `apply()` directly; use
the HF backend for wider contexts.

Neutral token ids are read from the JSON file named by the environment variable
`WMSCALE_NEUTRAL_IDS` (written by the generation stage before engine start).

Fixed-partition schemes (embedding hyperplanes) instead pass
`{"table": int, "bias": float}`; row `table` of the float32 [n, vocab] array in
the .npy file named by `WMSCALE_SIGNED_TABLES` is the signed green/red mask, with
neutral tokens already zero, identical to `wmsteg.schemes.SignedTableScheme`.
"""

from __future__ import annotations

import json
import os

import numpy as np
from typing import TYPE_CHECKING

import torch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

from wmsteg.schemes import _P, batched_signed_h01

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcRequestState

_ROW_CHUNK = 64


class ContextHashWatermarkProcessor(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: "LogitsProcRequestState"):
        n = int(req_states.max_num_reqs)
        dev = req_states.device
        # columns: mode (0 off, 1 hash, 2 table), key or table index, h, gamma threshold
        self.cfg_host = torch.zeros((n, 4), dtype=torch.int64)
        self.bias_host = torch.zeros(n, dtype=torch.float32)
        self.cfg_dev = torch.zeros((n, 4), dtype=torch.int64, device=dev)
        self.bias_dev = torch.zeros(n, dtype=torch.float32, device=dev)
        self.device = dev
        self.vocab_size = int(req_states.vocab_size)
        self.neutral = torch.zeros(self.vocab_size, dtype=torch.bool, device=dev)
        path = os.environ.get("WMSCALE_NEUTRAL_IDS")
        if path:
            ids = [i for i in json.loads(open(path).read()) if 0 <= int(i) < self.vocab_size]
            if ids:
                self.neutral[torch.as_tensor(ids, device=dev)] = True
        self.tables = None
        tpath = os.environ.get("WMSCALE_SIGNED_TABLES")
        if tpath:
            t = torch.from_numpy(np.load(tpath).astype(np.float32))
            self.tables = t.to(dev)
        self._dirty = False

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        wm = (sampling_params.extra_args or {}).get("wm")
        if wm is None:
            return
        if "table" in wm:
            if int(wm["table"]) < 0:
                raise ValueError("table index must be non-negative")
            return
        if int(wm.get("h", 0)) not in (0, 1):
            raise ValueError("vLLM watermark processor supports context width h in {0, 1}")
        if not 0.0 < float(wm.get("gamma", 0.5)) < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if not 0 <= int(wm["key"]) < _P:
            raise ValueError("key must be in [0, 2^31 - 1)")

    def add_request(self, req_idx: int, sampling_params: "SamplingParams") -> bool:
        wm = (sampling_params.extra_args or {}).get("wm")
        if wm is None:
            self.cfg_host[req_idx].zero_()
            self.bias_host[req_idx] = 0.0
        elif "table" in wm:
            if self.tables is None or int(wm["table"]) >= self.tables.shape[0]:
                raise ValueError("WMSCALE_SIGNED_TABLES missing or table index out of range")
            self.cfg_host[req_idx].zero_()
            self.cfg_host[req_idx, 0] = 2
            self.cfg_host[req_idx, 1] = int(wm["table"])
            self.bias_host[req_idx] = float(wm["bias"])
        else:
            self.cfg_host[req_idx, 0] = 1
            self.cfg_host[req_idx, 1] = int(wm["key"])
            self.cfg_host[req_idx, 2] = int(wm.get("h", 0))
            self.cfg_host[req_idx, 3] = int(float(wm.get("gamma", 0.5)) * _P)
            self.bias_host[req_idx] = float(wm["bias"])
        self._dirty = True
        return wm is not None

    def apply_staged_writes(self) -> None:
        if self._dirty:
            self.cfg_dev.copy_(self.cfg_host, non_blocking=True)
            self.bias_dev.copy_(self.bias_host, non_blocking=True)
            self._dirty = False

    def apply(self, logits: torch.Tensor, ctx: "LogitsContext") -> torch.Tensor:
        slots = ctx.expanded_idx_mapping.long()
        cfg = self.cfg_dev[slots]
        vocab_size = logits.shape[1]
        trows = torch.nonzero(cfg[:, 0] == 2).squeeze(1)
        if trows.numel():
            tables = self.tables
            if tables.shape[1] < vocab_size:
                tables = torch.nn.functional.pad(tables, (0, vocab_size - tables.shape[1]))
            signed = tables[cfg[trows, 1], :vocab_size]
            bias = self.bias_dev[slots[trows]].unsqueeze(1)
            logits[trows] += (bias * signed).to(logits.dtype)
        rows = torch.nonzero(cfg[:, 0] == 1).squeeze(1)
        if rows.numel() == 0:
            return logits
        neutral = self.neutral
        if neutral.numel() < vocab_size:
            neutral = torch.cat([neutral, torch.zeros(vocab_size - neutral.numel(), dtype=torch.bool,
                                                      device=logits.device)])
        for start in range(0, rows.numel(), _ROW_CHUNK):
            r = rows[start:start + _ROW_CHUNK]
            c = cfg[r]
            signed = batched_signed_h01(c[:, 1], c[:, 2], ctx.input_ids[r], c[:, 3], vocab_size, neutral)
            bias = self.bias_dev[slots[r]].unsqueeze(1)
            logits[r] += (bias * signed).to(logits.dtype)
        return logits
