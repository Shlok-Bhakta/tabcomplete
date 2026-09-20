"""Loss calculations shared by Stage-1 baseline and checkpoint evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_nll_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Return summed next-token NLL and the exact number of scored tokens."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch, seq, vocab] and input_ids [batch, seq]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids batch/sequence dimensions differ")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = input_ids[:, 1:].contiguous()
    if attention_mask is None:
        target_mask = torch.ones_like(shifted_targets, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape differs from input_ids")
        target_mask = attention_mask[:, 1:].to(dtype=torch.bool)
    token_count = int(target_mask.sum().item())
    if token_count == 0:
        return logits.new_zeros(()), 0
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_targets.view(-1),
        reduction="none",
    ).view_as(shifted_targets)
    return losses.masked_select(target_mask).sum(), token_count
