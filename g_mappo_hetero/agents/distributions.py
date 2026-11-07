from __future__ import annotations

import torch
from torch.distributions import Categorical


MASKED_LOGIT = -1.0e8


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask.to(dtype=torch.bool, device=logits.device)
    if logits.shape != mask_bool.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != mask shape {tuple(mask_bool.shape)}")
    if torch.any(mask_bool.sum(dim=-1) <= 0):
        raise ValueError("Each categorical action mask must allow at least one action.")
    return logits.masked_fill(~mask_bool, MASKED_LOGIT)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    return Categorical(logits=apply_action_mask(logits, mask))
