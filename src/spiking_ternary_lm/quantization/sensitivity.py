"""NSLLM-style per-layer quantization sensitivity analysis.

For each FAI-LIF in the model:
  1. Temporarily replace it with ``nn.Identity`` (no quantization).
  2. Compute the per-token CE on a held-out batch.
  3. Restore the FAI-LIF.

Report ``sensitivity[name] = ce_with_quant - ce_without_quant``. Larger
positive deltas mean the layer's quantization is costing more.

The NSLLM paper uses this to assign per-layer activation bit-widths:
sensitive layers get more bits; robust layers stay at the target. We
return the raw deltas; mapping to a YAML bit-width config is the
caller's responsibility.

Not used yet — wired in during Phase 4 when there's a real Stage A
checkpoint to score.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from spiking_ternary_lm.neurons.fai_lif import FAILIF


@dataclass
class SensitivityResult:
    baseline_loss: float       # full-model CE
    per_layer_delta: dict[str, float]  # CE - baseline when layer L is FP


@contextmanager
def _bypass_lif(model: nn.Module, target_name: str):
    """Temporarily replace one FAI-LIF instance with Identity."""
    parent: nn.Module | None = None
    attr: str | None = None
    parts = target_name.split(".")
    cur: nn.Module = model
    for p_ in parts[:-1]:
        cur = getattr(cur, p_)
    parent = cur
    attr = parts[-1]
    old = getattr(parent, attr)
    setattr(parent, attr, nn.Identity())
    try:
        yield
    finally:
        setattr(parent, attr, old)


@torch.no_grad()
def measure_sensitivity(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
) -> SensitivityResult:
    """Per-layer sensitivity: CE delta when each FAI-LIF is bypassed.

    Parameters
    ----------
    input_ids, labels:
        A single batch ``(B, T)`` of next-token labels.

    Returns the raw ``per_layer_delta`` map.
    """
    model.eval()
    if hasattr(model, "reset_state"):
        model.reset_state()

    def ce_with(model):
        if hasattr(model, "reset_state"):
            model.reset_state()
        logits = model(input_ids)
        return float(F.cross_entropy(
            logits.reshape(-1, vocab_size), labels.reshape(-1)
        ))

    baseline = ce_with(model)
    deltas: dict[str, float] = {}
    lif_names = [
        n for n, m in model.named_modules() if isinstance(m, FAILIF)
    ]
    for name in lif_names:
        with _bypass_lif(model, name):
            ce = ce_with(model)
        deltas[name] = ce - baseline
    if hasattr(model, "reset_state"):
        model.reset_state()
    return SensitivityResult(baseline_loss=baseline, per_layer_delta=deltas)


def assign_bit_widths(
    deltas: dict[str, float],
    base_bits: int = 4,
    bonus_bits: int = 2,
    top_k_sensitive: int = 8,
) -> dict[str, int]:
    """Map sensitivity deltas to per-layer bit-widths.

    Top-``k`` most sensitive layers get ``base_bits + bonus_bits``; the
    rest stay at ``base_bits``. This is a hand-rolled NSLLM-flavoured
    heuristic; refine when there's real data.
    """
    sorted_layers = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
    promoted = set(name for name, _ in sorted_layers[:top_k_sensitive])
    return {n: (base_bits + bonus_bits if n in promoted else base_bits) for n in deltas}
