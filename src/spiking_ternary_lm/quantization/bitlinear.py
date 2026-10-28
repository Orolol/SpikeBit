"""BitLinear: linear layer with ternary {-1, 0, +1} weights and group-wise scale.

Following BitNet b1.58 (Ma et al., 2402.17764) with the addition of
group-wise scales (à la BitCPM4) to handle outliers without breaking the
ternary structure. Activations are quantized externally by FAI-LIF.

Storage / forward contract:

  * ``shadow_weight``: master copy in the configured dtype (default FP32
    so vanilla AdamW's ``exp_avg``/``exp_avg_sq`` state is also FP32 —
    see DECISIONS.md #4, #12, #13). BF16 storage is still supported for
    callers that maintain their own FP32 master copy. The recommended
    training path is FP32 storage + ``torch.amp.autocast(bfloat16)`` at
    the matmul boundary (DEC-13). Receives optimizer updates.
  * Forward: weight is grouped along the in-features axis, per-group
    scale ``s = mean(|w|)`` (clamped to ``>= eps``), then STE-quantized
    to ``{-s, 0, +s}``. The result is used in ``F.linear``.
  * Bias is disabled by default (matches BitNet convention).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spiking_ternary_lm.quantization.ste import ste_quantize_ternary


class BitLinear(nn.Module):
    """Linear layer with group-wise ternary weights.

    Parameters
    ----------
    in_features:
        Input dimension. Must be divisible by ``group_size``.
    out_features:
        Output dimension.
    bias:
        If True, an FP bias is added after the matmul (FP, not
        quantized). Default False.
    group_size:
        Number of in-features per group sharing one scale. Default 128.
        Set to ``in_features`` for per-row scaling (single group per row).
    weight_dtype:
        Dtype of the shadow weight. Default ``torch.float32`` so that
        vanilla AdamW's optimizer state lives in FP32 and does not
        underflow (DECISIONS.md #12, #13). BF16 storage is still
        permitted, but the calling trainer is responsible for keeping
        an FP32 master copy when using a BF16 shadow.
    eps:
        Lower bound on per-group scale to avoid division by zero / dead
        groups.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        weight_dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
        init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.num_groups = self.in_features // self.group_size
        self.eps = float(eps)
        self.weight_dtype = weight_dtype

        # Kaiming-uniform init in FP32, then cast to weight_dtype. We use
        # `a=0` (ReLU mode) rather than the torch.nn.Linear legacy
        # `a=sqrt(5)` — the latter under-scales initial activations and
        # leaves spiking neurons dead at init. `init_gain` lets callers
        # further amplify the init (e.g. 2.0 or 3.0) to wake up LIFs in
        # very low-firing regimes.
        w = torch.empty(out_features, in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=0.0)
        if init_gain != 1.0:
            w.mul_(init_gain)
        self.shadow_weight = nn.Parameter(w.to(weight_dtype))

        if bias:
            # Bias init matches torch.nn.Linear (uniform from fan_in).
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in)
            b = torch.empty(out_features, dtype=torch.float32).uniform_(-bound, bound)
            self.bias = nn.Parameter(b.to(weight_dtype))
        else:
            self.register_parameter("bias", None)

    # -- public helpers ---------------------------------------------------------

    def quantized_weight(self) -> Tensor:
        """Return the current ternary-quantized weight (no STE in eval graph)."""
        with torch.no_grad():
            w = self.shadow_weight
            groups = w.view(self.out_features, self.num_groups, self.group_size)
            scale = groups.float().abs().mean(dim=-1, keepdim=True).clamp(min=self.eps)
            q = ste_quantize_ternary(groups, scale.to(groups.dtype))
            return q.view_as(w)

    def weight_distribution(self) -> dict[str, float]:
        """Fraction of ternary weights at each level. Useful for logging."""
        q = self.quantized_weight()
        # Compare against per-group scale to know which level each entry is at.
        groups = q.view(self.out_features, self.num_groups, self.group_size)
        scale = groups.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
        normalised = (groups / scale).round().clamp(-1, 1)
        n = normalised.numel()
        return {
            "neg": (normalised == -1).sum().item() / n,
            "zero": (normalised == 0).sum().item() / n,
            "pos": (normalised == 1).sum().item() / n,
        }

    def pack_weights(self) -> tuple[Tensor, Tensor]:
        """Return ``(ternary_int8, scales)`` for future packed deployment.

        ``ternary_int8`` has shape ``(out_features, in_features)`` with
        values in ``{-1, 0, +1}`` as ``int8``. ``scales`` has shape
        ``(out_features, num_groups)`` in FP32. This is a *forward
        placeholder* for the runtime that will eventually consume packed
        2-bit weights; for now we keep one byte per ternary value.

        No autograd; meant for inference and serialization.
        """
        with torch.no_grad():
            w = self.shadow_weight
            groups = w.view(self.out_features, self.num_groups, self.group_size)
            scale = groups.float().abs().mean(dim=-1, keepdim=True).clamp(min=self.eps)
            # Use the same threshold as the STE forward.
            mask = (groups.float().abs() > (scale / 2.0)).to(torch.int8)
            sign = torch.sign(groups.float()).to(torch.int8)
            ternary = (sign * mask).view(self.out_features, self.in_features)
            return ternary, scale.squeeze(-1)

    # -- forward ----------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        w = self.shadow_weight
        # Group view: (out, num_groups, group_size).
        groups = w.view(self.out_features, self.num_groups, self.group_size)
        # Per-group scale computed in FP32 for numerical stability, then cast.
        scale_fp32 = groups.float().abs().mean(dim=-1, keepdim=True).clamp(min=self.eps)
        scale = scale_fp32.to(groups.dtype)
        w_quant = ste_quantize_ternary(groups, scale).view_as(w)
        # Match input dtype for the matmul. The shadow's dtype is the
        # "weight precision"; the activation dtype drives the matmul.
        w_quant = w_quant.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_quant, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}, "
            f"weight_dtype={self.weight_dtype}"
        )
