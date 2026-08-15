from __future__ import annotations

import torch
from torch import nn

from .pooling import masked_mean_pool


class MeanPoolDetector(nn.Module):
    """Causal LM backbone + a 2-way linear classifier over mean-pooled residuals."""

    def __init__(self, causal_lm: nn.Module, hidden_size: int):
        super().__init__()
        self.causal_lm = causal_lm
        self.classifier = nn.Linear(hidden_size, 2)

    @property
    def backbone(self):
        # Hugging Face PreTrainedModel exposes the architecture's base model here.
        return self.causal_lm.base_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        pooled = masked_mean_pool(hidden, attention_mask)
        return self.classifier(pooled)
