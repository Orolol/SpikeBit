"""Straight-through estimator helpers for weight/activation quantization."""
from __future__ import annotations

import torch
from torch import Tensor


class _STEQuantizeTernary(torch.autograd.Function):
    """Ternary weight quantization with identity STE.

    Forward: ``q = sign(w) * (|w| > scale/2) * scale``  — values in ``{-scale, 0, +scale}``.
    Backward: identity gradient with respect to ``w``; no gradient through ``scale``.

    We deliberately drop the scale gradient: ``scale`` is computed from
    ``w`` (mean of absolute values per group) but we treat it as a fixed
    scaling factor for the purposes of the quantization step. This
    matches BitNet b1.58 reference behaviour and avoids the gradient
    feedback loop between the quantization threshold and the rounding.
    """

    @staticmethod
    def forward(ctx, w: Tensor, scale: Tensor) -> Tensor:
        # Cast to FP32 for the threshold/sign math, then back. Scale can
        # be tiny in BF16, so the FP32 promotion keeps the threshold
        # `scale / 2` well-defined even for very small groups.
        w32 = w.float()
        s32 = scale.float()
        mask = (w32.abs() > (s32 / 2.0)).to(w32.dtype)
        q = torch.sign(w32) * mask * s32
        return q.to(w.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # Identity STE on w; no gradient through scale.
        return grad_output, None


def ste_quantize_ternary(w: Tensor, scale: Tensor) -> Tensor:
    """Public alias for :class:`_STEQuantizeTernary`.

    Parameters
    ----------
    w:
        Shadow weight tensor. Any shape; the last meaningful axis is the
        group axis when called from :class:`BitLinear`.
    scale:
        Per-group scale, broadcastable with ``w``. Caller is responsible
        for clamping to a positive value (:class:`BitLinear` does this
        with ``eps=1e-8``).

    Returns
    -------
    Ternary tensor with values in ``{-scale, 0, +scale}`` per group, same
    shape and dtype as ``w``. Gradients flow as identity through ``w``.
    """
    return _STEQuantizeTernary.apply(w, scale)


class _STEQuantizeInteger(torch.autograd.Function):
    """Affine integer quantization with identity STE.

    Forward:
        ``q = round(x / scale + zero_point).clamp(qmin, qmax)``

    Backward:
        identity through ``x``; no gradient through ``scale`` or
        ``zero_point`` (they come from running stats updated by the
        observer in a separate, non-autograd path).

    Output dtype matches the input dtype so the value can flow through
    downstream FP modules (subsequent BitLinear, sigmoid, etc.) without
    a manual cast.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        qmin: int,
        qmax: int,
    ) -> Tensor:
        # Promote to FP32 for the division/round so we don't lose
        # precision on small scales in BF16, then cast back.
        x32 = x.float()
        scale32 = scale.float()
        zp32 = zero_point.float()
        q = (x32 / scale32 + zp32).round().clamp(qmin, qmax)
        return q.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Identity STE on x. None for scale, zero_point, qmin, qmax.
        return grad_output, None, None, None, None


def ste_quantize_integer(
    x: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    qmin: int,
    qmax: int,
) -> Tensor:
    """Quantize ``x`` to integers in ``[qmin, qmax]`` via observer-derived
    ``scale`` and ``zero_point``. Identity-STE on ``x``.

    Used by the observer-based :class:`FAILIF`. Output dtype matches
    ``x.dtype`` (so the spike count can be re-multiplied by ``scale`` in
    the caller's dtype without a manual cast).
    """
    return _STEQuantizeInteger.apply(x, scale, zero_point, qmin, qmax)
