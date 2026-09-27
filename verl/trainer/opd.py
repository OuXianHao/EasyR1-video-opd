"""Vanilla sampled-token on-policy distillation objective.

The sampled response and both scorers are fixed during this actor update. Only
the current actor log-probabilities carry a gradient.
"""

import torch


def compute_opd_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token-mean OPD loss, detached reward, and importance ratio."""
    if not (log_probs.shape == old_log_probs.shape == teacher_log_probs.shape == response_mask.shape):
        raise ValueError("OPD log-probabilities, responses, and response mask must have the same shape.")
    mask = response_mask.bool()
    valid_tokens = mask.sum()
    if not valid_tokens.item():
        raise ValueError("OPD requires at least one valid response token.")
    for name, value in (
        ("teacher_log_probs", teacher_log_probs),
        ("old_log_probs", old_log_probs),
        ("log_probs", log_probs),
    ):
        if not torch.isfinite(value[mask]).all():
            raise ValueError(f"OPD {name} contains a non-finite valid token.")

    compute_dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    old = old_log_probs.detach().to(compute_dtype)
    teacher = teacher_log_probs.detach().to(compute_dtype)
    current = log_probs.to(compute_dtype)
    # Avoid exp() on masked padding, while leaving all valid log-ratios unclipped.
    reward = torch.where(mask, teacher - old, torch.zeros_like(old)).detach()
    ratio = torch.exp(torch.where(mask, current - old, torch.zeros_like(current)))
    if not torch.isfinite(ratio[mask]).all():
        raise ValueError("OPD importance ratio overflowed on a valid token.")
    loss = -(reward * ratio).sum() / valid_tokens
    return loss, reward, ratio
