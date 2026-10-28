"""Surrogate gradient functions for spiking neurons.

These wrap the non-differentiable Heaviside step (`H(V - V_thr)`) used in
*binary* LIF dynamics. The forward pass returns a binary spike; the
backward pass uses a chosen smooth surrogate derivative.

In the default FAI-LIF path used for Stage A/C training we do **not** rely
on these — FAI-LIF emits integer counts and uses pure STE through
round+clip (see `fai_lif.py`). The surrogates here are kept for:

  * binary-LIF baselines / ablations,
  * future inference-time gradient checks,
  * optional scaling of the FAI-LIF STE gradient by a surrogate-derivative
    weighting (not enabled by default).

Each surrogate takes `alpha` at forward-call time so the trainer can
anneal it on a schedule (start soft, sharpen over training).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _heaviside(x: Tensor) -> Tensor:
    return (x >= 0).to(x.dtype)


class FastSigmoidSurrogate(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate gradient.

    Backward:  ``surrogate'(x) = 1 / (1 + alpha * |x|) ** 2``

    Reference: Zenke & Ganguli, "SuperSpike" (2018), eq. 5 of the
    surrogate-gradient survey (Neftci et al. 2019).
    """

    @staticmethod
    def forward(ctx, v_minus_thr: Tensor, alpha: float = 4.0) -> Tensor:
        ctx.save_for_backward(v_minus_thr)
        ctx.alpha = float(alpha)
        return _heaviside(v_minus_thr)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        denom = 1.0 + alpha * x.abs()
        grad_in = grad_output / (denom * denom)
        return grad_in, None


class ArcTanSurrogate(torch.autograd.Function):
    """Heaviside forward, arctan surrogate gradient.

    Backward:  ``surrogate'(x) = (alpha / pi) / (1 + (pi * alpha * x) ** 2)``

    Reference: Fang et al., "Incorporating Learnable Membrane Time Constant
    to Enhance Learning of Spiking Neural Networks" (2021).
    """

    @staticmethod
    def forward(ctx, v_minus_thr: Tensor, alpha: float = 2.0) -> Tensor:
        ctx.save_for_backward(v_minus_thr)
        ctx.alpha = float(alpha)
        return _heaviside(v_minus_thr)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        pa = math.pi * alpha
        grad_in = grad_output * (alpha / math.pi) / (1.0 + (pa * x) ** 2)
        return grad_in, None


class RectangularSurrogate(torch.autograd.Function):
    """Heaviside forward, rectangular (boxcar) surrogate gradient.

    Backward:  ``1 if |x| < 0.5 else 0``  (alpha currently unused;
    accepted for API consistency and reserved for future width scaling).

    Reference: Wu et al., "Spatio-temporal Backpropagation for Training
    High-performance Spiking Neural Networks" (2018).
    """

    @staticmethod
    def forward(ctx, v_minus_thr: Tensor, alpha: float = 1.0) -> Tensor:
        ctx.save_for_backward(v_minus_thr)
        ctx.alpha = float(alpha)
        return _heaviside(v_minus_thr)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        mask = (x.abs() < 0.5).to(grad_output.dtype)
        return grad_output * mask, None


SURROGATES: dict[str, type[torch.autograd.Function]] = {
    "fast_sigmoid": FastSigmoidSurrogate,
    "arctan": ArcTanSurrogate,
    "rectangular": RectangularSurrogate,
}
