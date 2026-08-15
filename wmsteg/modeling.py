from __future__ import annotations

import torch
from torch import nn

from .pooling import last_nonpad_pool, masked_mean_pool


class LinearWatermarkDetector(nn.Module):
    """Causal-LM backbone + a 2-way linear classifier over a chosen readout."""

    def __init__(self, causal_lm: nn.Module, hidden_size: int, readout: str = "mean"):
        super().__init__()
        if readout not in {"mean", "last"}:
            raise ValueError(f"Unknown readout: {readout}")
        self.causal_lm = causal_lm
        self.readout = readout
        self.classifier = nn.Linear(hidden_size, 2)

    @property
    def backbone(self):
        # Hugging Face PreTrainedModel exposes the architecture's base model here.
        return self.causal_lm.base_model

    def pooled_representation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if self.readout == "mean":
            return masked_mean_pool(hidden, attention_mask)
        return last_nonpad_pool(hidden, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pooled_representation(input_ids, attention_mask))


# Backwards-compatible alias for existing scripts/checkpoints.
class MeanPoolDetector(LinearWatermarkDetector):
    def __init__(self, causal_lm: nn.Module, hidden_size: int):
        super().__init__(causal_lm, hidden_size, readout="mean")
