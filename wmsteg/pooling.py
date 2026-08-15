from __future__ import annotations

import torch


def masked_mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations while ignoring padding."""
    if hidden_state.ndim != 3:
        raise ValueError("hidden_state must be [batch, seq, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [batch, seq]")
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_state.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_state * mask).sum(dim=1) / denom


def last_nonpad_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the final non-padding token representation from a right-padded batch."""
    if hidden_state.ndim != 3:
        raise ValueError("hidden_state must be [batch, seq, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [batch, seq]")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden_state and attention_mask batch/sequence dimensions must match")
    last_idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
    batch_idx = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_idx, last_idx]
