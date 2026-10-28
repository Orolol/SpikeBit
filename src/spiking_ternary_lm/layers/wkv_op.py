"""Fused WKV op for the spiking-ternary RWKV time-mixing block.

Implements the BlinkDL stable WKV recurrence as a single CUDA kernel with
custom forward and backward, exposed via a ``torch.autograd.Function``.
Falls back to a pure-PyTorch reference implementation on CPU (and as a
debug option).

Public entry point:

    wkv_forward(w_eff, u, k, v, aa0, bb0, pp0)
        -> (wkv, aaT, bbT, ppT)

where ``w_eff = exp(time_decay)`` is the positive per-step decay
applied in log-space as ``pp - w_eff``. The autograd Function returns
``(wkv, aaT, bbT, ppT)``; only ``wkv`` carries gradient by default in our
caller — ``aaT, bbT, ppT`` are detached state buffers re-used across
training windows.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# --- Lazy CUDA extension load --------------------------------------------

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_EXT = None
_LOAD_ATTEMPTED = False


def _load_extension():
    """Compile (or fetch from cache) the fused WKV CUDA extension.

    Returns ``None`` if CUDA is not available or compilation fails — the
    caller falls back to the Python reference path in that case.
    """
    global _EXT, _LOAD_ATTEMPTED
    if _EXT is not None or _LOAD_ATTEMPTED:
        return _EXT
    _LOAD_ATTEMPTED = True
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        # Build for the GPU we are running on (sm_120 on the 5090, sm_75 on
        # a Colab T4, ...); allow override via env.
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _EXT = load(
            name="spikebit_wkv_cuda",
            sources=[str(_CSRC_DIR / "wkv_cuda.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-Xptxas=-O3",
                "-lineinfo",
            ],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - build failures bubble to caller
        print(f"[wkv_op] CUDA extension failed to load, falling back to Python: {exc}")
        _EXT = None
    return _EXT


# --- Reference (Python) implementation ----------------------------------

def _wkv_python(
    w_eff: Tensor, u: Tensor, k: Tensor, v: Tensor,
    aa0: Tensor, bb0: Tensor, pp0: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pure-PyTorch reference that mirrors the CUDA kernel.

    All tensors FP32. Shapes:
      w_eff, u: (C,)
      k, v   : (B, T, C)
      aa0, bb0, pp0: (B, C)
    Returns (wkv (B,T,C), aaT (B,C), bbT (B,C), ppT (B,C)).
    """
    B, T, C = k.shape
    aa, bb, pp = aa0, bb0, pp0
    outs = []
    for t in range(T):
        kt = k[:, t]
        vt = v[:, t]
        ww = u + kt
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        outs.append((e1 * aa + e2 * vt) / (e1 * bb + e2))
        ww2 = pp - w_eff
        qq2 = torch.maximum(ww2, kt)
        e1u = torch.exp(ww2 - qq2)
        e2u = torch.exp(kt - qq2)
        aa = e1u * aa + e2u * vt
        bb = e1u * bb + e2u
        pp = qq2
    return torch.stack(outs, dim=1), aa, bb, pp


# --- Autograd Function (CUDA fast path) ---------------------------------

class _WKVFunction(torch.autograd.Function):
    """Fused WKV recurrence.

    Inputs are all FP32. The forward stashes the original tensors needed
    by the backward replay (we recompute per-step state to save memory).
    """

    @staticmethod
    def forward(
        ctx: Any,
        w_eff: Tensor, u: Tensor, k: Tensor, v: Tensor,
        aa0: Tensor, bb0: Tensor, pp0: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ext = _load_extension()
        if ext is None or not k.is_cuda:
            wkv, aaT, bbT, ppT = _wkv_python(w_eff, u, k, v, aa0, bb0, pp0)
            ctx.save_for_backward(w_eff, u, k, v, aa0, bb0, pp0)
            ctx._use_cuda = False
            return wkv, aaT, bbT, ppT
        # Need contiguous FP32 CUDA tensors.
        w_eff_c = w_eff.contiguous()
        u_c = u.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        aa0_c = aa0.contiguous()
        bb0_c = bb0.contiguous()
        pp0_c = pp0.contiguous()
        wkv, aaT, bbT, ppT = ext.forward(w_eff_c, u_c, k_c, v_c, aa0_c, bb0_c, pp0_c)
        ctx.save_for_backward(w_eff_c, u_c, k_c, v_c, aa0_c, bb0_c, pp0_c)
        ctx._use_cuda = True
        return wkv, aaT, bbT, ppT

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        gwkv: Tensor, gaaT: Tensor, gbbT: Tensor, gppT: Tensor,
    ) -> tuple[Tensor | None, ...]:
        w_eff, u, k, v, aa0, bb0, pp0 = ctx.saved_tensors
        # Coerce None / zero-shaped grads to zero tensors for downstream
        # arithmetic in the CUDA kernel. Autograd hands us zero tensors
        # by default unless a needs_input_grad short-circuits, but we
        # belt-and-braces it here.
        B = k.shape[0]; C = k.shape[2]
        if gaaT is None:
            gaaT = torch.zeros(B, C, dtype=torch.float32, device=k.device)
        if gbbT is None:
            gbbT = torch.zeros(B, C, dtype=torch.float32, device=k.device)
        if gppT is None:
            gppT = torch.zeros(B, C, dtype=torch.float32, device=k.device)
        gaaT = gaaT.to(torch.float32).contiguous()
        gbbT = gbbT.to(torch.float32).contiguous()
        gppT = gppT.to(torch.float32).contiguous()

        if not getattr(ctx, "_use_cuda", False):
            return _backward_python(
                w_eff, u, k, v, aa0, bb0, pp0, gwkv, gaaT, gbbT, gppT
            )

        ext = _load_extension()
        gwkv_c = gwkv.contiguous()
        gw, gu, gk, gv, gaa0, gbb0, gpp0 = ext.backward(
            w_eff, u, k, v, aa0, bb0, pp0, gwkv_c, gaaT, gbbT, gppT
        )
        return gw, gu, gk, gv, gaa0, gbb0, gpp0


def _backward_python(
    w_eff: Tensor, u: Tensor, k: Tensor, v: Tensor,
    aa0: Tensor, bb0: Tensor, pp0: Tensor, gwkv: Tensor,
    gaaT: Tensor, gbbT: Tensor, gppT: Tensor,
):
    """Python backward — used on CPU and as a numerical reference.

    Uses torch autograd on the reference forward; properly handles the
    full output tuple so grads flowing through ``aaT/bbT/ppT`` are
    accounted for.

    Note: ``Function.backward`` is called by the autograd engine with
    grad-tracking disabled. We re-enable it locally so that ``_wkv_python``
    builds an autograd graph we can differentiate through.
    """
    with torch.enable_grad():
        w_eff = w_eff.detach().clone().requires_grad_(True)
        u = u.detach().clone().requires_grad_(True)
        k = k.detach().clone().requires_grad_(True)
        v = v.detach().clone().requires_grad_(True)
        aa0 = aa0.detach().clone().requires_grad_(True)
        bb0 = bb0.detach().clone().requires_grad_(True)
        pp0 = pp0.detach().clone().requires_grad_(True)
        wkv, aaT, bbT, ppT = _wkv_python(w_eff, u, k, v, aa0, bb0, pp0)
        grads = torch.autograd.grad(
            (wkv, aaT, bbT, ppT),
            (w_eff, u, k, v, aa0, bb0, pp0),
            grad_outputs=(gwkv, gaaT, gbbT, gppT),
            retain_graph=False,
        )
    return grads


def wkv_forward(
    w_eff: Tensor, u: Tensor, k: Tensor, v: Tensor,
    aa0: Tensor, bb0: Tensor, pp0: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply the fused WKV recurrence.

    All inputs must be FP32. ``w_eff`` is the positive per-step decay
    (already exponentiated). Returns ``(wkv, aaT, bbT, ppT)``.

    The fused CUDA kernel only consumes FP32 buffers. We force-disable
    autocast and explicitly cast each input to FP32 here so that callers
    running under ``torch.amp.autocast(dtype=bfloat16)`` (the Stage-A
    mixed-precision path; see DECISIONS.md #13) cannot accidentally
    hand BF16 tensors to the kernel.
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        w_eff = w_eff.float()
        u = u.float()
        k = k.float()
        v = v.float()
        aa0 = aa0.float()
        bb0 = bb0.float()
        pp0 = pp0.float()
        return _WKVFunction.apply(w_eff, u, k, v, aa0, bb0, pp0)
