"""Norm layers used in front of :class:`BitLinear`.

We follow BitNet b1.58 (Ma et al., 2402.17764) which places an RMSNorm
immediately before each BitLinear. This stabilises the per-row magnitude
of the input so that the per-group ternary scale and the downstream
LIF threshold ``v_thr`` see a consistent input distribution across
training. Without it, the LIF firing rate drifts as weights grow and the
network gets stuck in either dead-neuron or saturated regimes (see the
MNIST sanity run logs).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation with optional learnable gain.

    ``y = x / sqrt(mean(x ** 2, dim=-1) + eps) * weight``
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        # Compute the reduction in FP32 for stability even at BF16.
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = (x_fp32 * rms).to(x.dtype)
        if self.weight is not None:
            y = y * self.weight.to(x.dtype)
        return y

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
