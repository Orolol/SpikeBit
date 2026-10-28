"""Ablation A4: FP baseline for Stage A.

Same arch (RWKV-4 + token-shift + WKV + per-block LN/residuals + FP
embed/head), same data, same hyperparams as the midrun observer pol,
but with **both** quantizations disabled:

  * :class:`BitLinear` forward is monkey-patched to ``F.linear(x,
    shadow_weight, bias)`` — no ternary STE, no per-group scale.
  * :class:`FAILIF` forward is monkey-patched to identity — no integer
    quantization, no leak, no reset. The LIFs become FP pass-throughs.

This isolates "training-budget-limited" vs. "quantization-limited"
for the observed plateau at ppl ≈ 230. If the FP baseline reaches
ppl ≪ 230, the quantization stack is the limiter; if it plateaus
similarly, the limiter is the training-budget at this scale.

NOTE: an identity FAI-LIF removes the recurrent integration entirely —
this is the most-FP variant of our arch but no longer "RWKV-like" in
the spiking sense. A more conservative ablation would replace FAI-LIF
with the original RWKV-4 activations (sigmoid/square); we will run
that as a follow-up only if A4 is ambiguous.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F

from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.quantization.bitlinear import BitLinear
from spiking_ternary_lm.training.stage_a_finetune import _parse_args, train


def _fp_bitlinear_forward(self: BitLinear, x: torch.Tensor) -> torch.Tensor:
    w = self.shadow_weight.to(x.dtype)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


def _identity_failif_forward(self: FAILIF, x: torch.Tensor) -> torch.Tensor:
    return x


def main() -> int:
    BitLinear.forward = _fp_bitlinear_forward  # type: ignore[assignment]
    FAILIF.forward = _identity_failif_forward  # type: ignore[assignment]
    print("[fp-baseline] BitLinear.forward → F.linear(shadow_weight)")
    print("[fp-baseline] FAILIF.forward    → identity")
    return train(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
