"""Tiny helpers for warmup + cosine learning-rate schedules."""
from __future__ import annotations

import math


def linear_warmup_cosine(step: int, warmup_steps: int, total_steps: int, min_lr_frac: float = 0.0) -> float:
    """Return an LR multiplier (∈ [min_lr_frac, 1.0]) for the given step."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if step >= total_steps:
        return min_lr_frac
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_frac + (1.0 - min_lr_frac) * cos
