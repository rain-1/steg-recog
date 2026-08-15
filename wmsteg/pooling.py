from __future__ import annotations

import torch


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations while ignoring padding."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must be [batch, seq, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [batch, seq]")
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * mask).sum(dim=1) / denom
