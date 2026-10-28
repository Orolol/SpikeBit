"""Detailed firing-rate statistics across the model.

Attach forward hooks to every FAI-LIF, run a held-out batch, and dump
per-layer averages: mean firing rate (normalised to n_max), saturation
fraction, dead-neuron fraction.
"""
from __future__ import annotations

import torch
from torch import nn

from spiking_ternary_lm.neurons.fai_lif import FAILIF


class SpikeStatsCollector:
    def __init__(self, model: nn.Module):
        self.stats: dict[str, dict[str, float]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, m in model.named_modules():
            if isinstance(m, FAILIF):
                self._handles.append(m.register_forward_hook(self._make_hook(name, m)))

    def _make_hook(self, name: str, lif: FAILIF):
        n_max = lif.n_max

        def hook(_module, _inputs, output):
            with torch.no_grad():
                out = output.detach().float()
                self.stats[name] = {
                    "fire_norm": float((out / max(n_max, 1)).mean()),
                    "mean_count": float(out.mean()),
                    "saturated": float((out >= n_max).float().mean()),
                    "dead": float((out.mean(dim=tuple(range(out.dim() - 1))) == 0).float().mean()),
                }
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "SpikeStatsCollector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@torch.no_grad()
def measure(model: nn.Module, input_ids: torch.Tensor, reset_state: bool = True) -> dict[str, dict[str, float]]:
    """Single-shot measurement: forward `input_ids` and return spike stats."""
    if reset_state and hasattr(model, "reset_state"):
        model.reset_state()
    with SpikeStatsCollector(model) as col:
        model(input_ids)
        return dict(col.stats)
