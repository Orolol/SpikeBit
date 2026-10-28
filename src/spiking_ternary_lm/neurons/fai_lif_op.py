"""Fused FAI-LIF op for the observer-based :class:`FAILIF` neuron.

Implements one full sequence-forward of the LIF + integer-STE quantize +
soft-reset cycle in a single CUDA kernel, with a custom backward. The
observer's running min/max EMA is handled OUTSIDE the kernel (the host
calls ``observer.update_stats(v_pre_seq)`` once on the returned membrane
trajectory) — see ``FAILIF.forward`` for the orchestration.

Public entry point:

    fai_lif_apply(x, beta, gamma, v0, scale, zp, qmin, qmax)
        -> (spike_seq, v_pre_seq, v_final)

All inputs are FP32. Shapes:
  x           : (B, T, H)
  beta, gamma : (H,) post-sigmoid (already in [0, 1])
  v0          : (B, H)
  scale, zp   : scalar tensors (FP32)
  qmin, qmax  : ints

Returns:
  spike_seq   : (B, T, H) FP32 integer-valued
  v_pre_seq   : (B, T, H) FP32 (for host-side observer.update_stats)
  v_final     : (B, H)   FP32 (carry state)

Gradients:
  STE-identity through the round/clamp; ``scale``, ``zp``, ``qmin``,
  ``qmax`` never receive grads. Grads flow to ``x``, ``beta``, ``gamma``,
  ``v0``.
"""
from __future__ import annotations

import math
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
    """Compile (or fetch from cache) the fused FAI-LIF CUDA extension.

    Returns ``None`` if CUDA is unavailable or compilation fails — the
    caller falls back to the Python reference path.
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
            name="spikebit_fai_lif_cuda",
            sources=[str(_CSRC_DIR / "fai_lif_cuda.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-Xptxas=-O3",
                "-lineinfo",
            ],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - build failures bubble up
        print(f"[fai_lif_op] CUDA extension failed to load, falling back to Python: {exc}")
        _EXT = None
    return _EXT


# --- Reference (Python) implementation ----------------------------------

def _fai_lif_python(
    x: Tensor, beta: Tensor, gamma: Tensor, v0: Tensor,
    scale: Tensor, zp: Tensor, qmin: int, qmax: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-PyTorch reference that mirrors the CUDA kernel exactly.

    Uses a FIXED ``scale``/``zp`` for all T steps (matching the kernel's
    contract — the observer EMA update is the host's responsibility).

    All tensors FP32. Shapes match ``fai_lif_apply``.
    Returns (spike_seq (B,T,H), v_pre_seq (B,T,H), v_final (B,H)).
    """
    B, T, H = x.shape
    v = v0
    spikes = []
    v_pres = []
    inv_scale = 1.0 / scale
    for t in range(T):
        x_t = x[:, t]
        v_pre = beta * v + x_t
        # STE quantize: round + clamp on a detached path, identity on x.
        q = (v_pre * inv_scale + zp).round().clamp(qmin, qmax)
        # Identity STE: detach the quant correction so grad flows as identity
        # back through v_pre.
        spike = v_pre + (q - v_pre).detach()
        dequant = (scale * (spike - zp)).detach()
        v_new = gamma * (v_pre - dequant)
        spikes.append(spike)
        v_pres.append(v_pre)
        v = v_new
    return torch.stack(spikes, dim=1), torch.stack(v_pres, dim=1), v


# --- Autograd Function (CUDA fast path) ---------------------------------

class _FAILIFFunction(torch.autograd.Function):
    """Fused FAI-LIF sequence forward.

    Forward stashes the inputs needed by the backward kernel's replay.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor, beta: Tensor, gamma: Tensor, v0: Tensor,
        scale: Tensor, zp: Tensor, qmin: int, qmax: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Undefined output grads arrive as ``None`` instead of zero
        # tensors, so backward can test them without a device sync.
        ctx.set_materialize_grads(False)
        ext = _load_extension()
        if ext is None or not x.is_cuda:
            spike, v_pre, v_final = _fai_lif_python(
                x, beta, gamma, v0, scale, zp, qmin, qmax
            )
            ctx.save_for_backward(x, beta, gamma, v0)
            ctx._scale = scale.detach() if isinstance(scale, Tensor) else float(scale)
            ctx._zp = zp.detach() if isinstance(zp, Tensor) else float(zp)
            ctx._qmin = int(qmin)
            ctx._qmax = int(qmax)
            ctx._use_cuda = False
            return spike, v_pre, v_final

        x_c = x.contiguous()
        beta_c = beta.contiguous()
        gamma_c = gamma.contiguous()
        v0_c = v0.contiguous()
        # (scale, zp) stay on-device as 0-dim FP32 tensors: the kernel
        # dereferences them itself, so no ``.item()`` host sync per call.
        # (1,) per-tensor or (H,) per-channel grids, read in-kernel.
        scale_t = torch.as_tensor(scale, dtype=torch.float32, device=x.device).reshape(-1).contiguous()
        zp_t = torch.as_tensor(zp, dtype=torch.float32, device=x.device).reshape(-1).contiguous()
        spike, v_pre, v_final = ext.forward(
            x_c, beta_c, gamma_c, v0_c, scale_t, zp_t, int(qmin), int(qmax)
        )
        ctx.save_for_backward(x_c, beta_c, gamma_c, v0_c, scale_t, zp_t)
        ctx._qmin = int(qmin)
        ctx._qmax = int(qmax)
        ctx._use_cuda = True
        return spike, v_pre, v_final

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        gspike: Tensor, gv_pre: Tensor, gv_final: Tensor,
    ) -> tuple[Tensor | None, ...]:
        if getattr(ctx, "_use_cuda", False):
            x, beta, gamma, v0, scale_t, zp_t = ctx.saved_tensors
        else:
            x, beta, gamma, v0 = ctx.saved_tensors
        B, T, H = x.shape

        # gv_pre is the gradient flowing back through the v_pre_seq output.
        # That output is consumed by the host-side observer.update_stats
        # which runs under @torch.no_grad, so in practice gv_pre is zero.
        # We still accept and ignore it to keep the autograd signature clean.
        # If a downstream consumer ever does take a real grad through v_pre,
        # we'd need to fold it into gspike (since spike = v_pre + (q-v_pre).detach()
        # has identity STE on v_pre, so adding gspike + gv_pre would do it).
        # Belt-and-braces: if gv_pre is defined, treat it as additional
        # gspike-equivalent gradient (works because of the identity STE).
        # With materialize_grads off an unused output arrives as ``None``,
        # so this is a host-side check, not a ``.sum() > 0`` device sync.
        if gspike is None:
            gspike = torch.zeros_like(x)
        if gv_pre is not None:
            gspike = gspike + gv_pre

        if gv_final is None:
            gv_final = torch.zeros(B, H, dtype=torch.float32, device=x.device)
        gv_final = gv_final.to(torch.float32).contiguous()
        gspike = gspike.to(torch.float32).contiguous()

        if not getattr(ctx, "_use_cuda", False):
            return _backward_python(
                x, beta, gamma, v0,
                ctx._scale, ctx._zp, ctx._qmin, ctx._qmax,
                gspike, gv_final,
            )

        ext = _load_extension()
        gx, gbeta, ggamma, gv0 = ext.backward(
            x, beta, gamma, v0,
            scale_t, zp_t, ctx._qmin, ctx._qmax,
            gspike, gv_final,
        )
        # No grads for scale, zp, qmin, qmax.
        return gx, gbeta, ggamma, gv0, None, None, None, None


def _backward_python(
    x: Tensor, beta: Tensor, gamma: Tensor, v0: Tensor,
    scale, zp, qmin: int, qmax: int,
    gspike: Tensor, gv_final: Tensor,
):
    """Python backward — used on CPU and as a numerical reference.

    Uses torch autograd on the reference forward. Only returns grads for
    the inputs that take grad in the autograd Function.
    """
    # Called from inside an autograd.Function backward, where grad mode is
    # off: re-enable it for the reference replay.
    with torch.enable_grad():
        x_r = x.detach().clone().requires_grad_(True)
        beta_r = beta.detach().clone().requires_grad_(True)
        gamma_r = gamma.detach().clone().requires_grad_(True)
        v0_r = v0.detach().clone().requires_grad_(True)
        scale_t = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
        zp_t = torch.as_tensor(zp, dtype=torch.float32, device=x.device)
        spike, _v_pre, v_final = _fai_lif_python(
            x_r, beta_r, gamma_r, v0_r, scale_t, zp_t, qmin, qmax
        )
        grads = torch.autograd.grad(
            (spike, v_final),
            (x_r, beta_r, gamma_r, v0_r),
            grad_outputs=(gspike, gv_final),
            retain_graph=False,
            allow_unused=True,
        )
    gx, gbeta, ggamma, gv0 = grads
    if gx is None:
        gx = torch.zeros_like(x)
    if gbeta is None:
        gbeta = torch.zeros_like(beta)
    if ggamma is None:
        ggamma = torch.zeros_like(gamma)
    if gv0 is None:
        gv0 = torch.zeros_like(v0)
    return gx, gbeta, ggamma, gv0, None, None, None, None


def fai_lif_apply(
    x: Tensor, beta: Tensor, gamma: Tensor, v0: Tensor,
    scale: Tensor, zp: Tensor, qmin: int, qmax: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply the fused FAI-LIF sequence forward.

    All inputs must be FP32. Returns ``(spike_seq, v_pre_seq, v_final)``.
    """
    return _FAILIFFunction.apply(x, beta, gamma, v0, scale, zp, qmin, qmax)


# --- LSQ variant: learned step size (Esser et al., ICLR 2020) ------------
#
# Same fused forward as :class:`_FAILIFFunction`, but (1) the grid is a pair
# of learnable scalars -- ``log_scale`` (``s = exp(log_scale)``, so Adam's
# step is multiplicative; activation ranges here move by x2-6 within a few
# hundred steps) and ``zp`` in integer units -- and (2) the module output is
# the **dequantized** spike ``x_hat = s * (q - zp)`` instead of the raw
# integer ``q``. A per-tensor scalar folds into the next BitLinear's group
# scales, so deployment stays integer; but it is what gives LSQ its
# gradient: with ``u = v_pre / s + zp`` and ``q = clip(round(u))``,
#
#     dx_hat/ds  = round(u) - u      inside the range
#                = q - zp            where clipped (upper: qmax - zp, ...)
#     dx_hat/dzp = 0 inside, -s where clipped
#     dx_hat/dv  = 1 (identity STE; DEC-4: not masked outside the range)
#
# i.e. saturated elements push the step up, which an integer-only output
# cannot express (that variant saturated and diverged: runs/stage_a_lsq_1500/
# attempt1_additive and attempt2_logscale_intout). ``dlog_scale = s * ds``.
# Everything is scaled by LSQ's ``g = 1 / sqrt(N * Q_P)``.


def _lsq_grad_scale(numel: int, qmin: int, qmax: int) -> float:
    q_p = max(abs(int(qmin)), abs(int(qmax)), 1)
    return 1.0 / math.sqrt(float(numel) * q_p)


class _LSQFAILIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor, beta: Tensor, gamma: Tensor, v0: Tensor,
        log_scale: Tensor, zp: Tensor, qmin: int, qmax: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        s32 = log_scale.detach().to(torch.float32).reshape(()).exp()
        zp32 = zp.detach().to(torch.float32).reshape(())
        ext = _load_extension()
        use_cuda = ext is not None and x.is_cuda
        if use_cuda:
            x_c, beta_c, gamma_c, v0_c = (t.contiguous() for t in (x, beta, gamma, v0))
            spike, v_pre, v_final = ext.forward(
                x_c, beta_c, gamma_c, v0_c, s32, zp32, int(qmin), int(qmax)
            )
        else:
            x_c, beta_c, gamma_c, v0_c = x, beta, gamma, v0
            spike, v_pre, v_final = _fai_lif_python(
                x, beta, gamma, v0, s32, zp32, qmin, qmax
            )
            # The reference's STE trick (v + (q - v).detach()) leaves float
            # residue on the integers; snap them like the kernel's output.
            spike, v_pre, v_final = spike.detach().round(), v_pre.detach(), v_final.detach()
        x_hat = s32 * (spike - zp32)
        ctx.save_for_backward(x_c, beta_c, gamma_c, v0_c, s32, zp32, v_pre, spike)
        ctx._qmin = int(qmin)
        ctx._qmax = int(qmax)
        ctx._use_cuda = use_cuda
        return x_hat, spike, v_pre, v_final

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, gx_hat: Tensor, gspike_int: Tensor, gv_pre: Tensor, gv_final: Tensor,
    ) -> tuple[Tensor | None, ...]:
        x, beta, gamma, v0, s32, zp32, v_pre, q = ctx.saved_tensors
        B, T, H = x.shape
        qmin, qmax = ctx._qmin, ctx._qmax
        if gx_hat is None:
            gx_hat = torch.zeros_like(x)
        gx_hat = gx_hat.to(torch.float32)
        if gv_final is None:
            gv_final = torch.zeros(B, H, dtype=torch.float32, device=x.device)
        gv_final = gv_final.to(torch.float32).contiguous()

        u = v_pre / s32 + zp32
        inside = ((u > qmin) & (u < qmax)).to(torch.float32)
        g = _lsq_grad_scale(u.numel(), qmin, qmax)
        # dx_hat/ds per element, then chain to log_scale (x s).
        ds = inside * (q - u) + (1.0 - inside) * (q - zp32)
        glog = (gx_hat * ds).sum() * s32 * g
        gzp = (gx_hat * (inside - 1.0)).sum() * s32 * g

        # Gradient w.r.t. the integer spike (chain of x_hat = s * (q - zp)),
        # plus any direct grad on the integer output (normally None).
        gspike = gx_hat * s32
        if gspike_int is not None:
            gspike = gspike + gspike_int.to(torch.float32)
        if gv_pre is not None:
            gspike = gspike + gv_pre.to(torch.float32)
        gspike = gspike.contiguous()

        if ctx._use_cuda:
            ext = _load_extension()
            gx, gbeta, ggamma, gv0 = ext.backward(
                x, beta, gamma, v0, s32, zp32, qmin, qmax, gspike, gv_final
            )
        else:
            gx, gbeta, ggamma, gv0, *_ = _backward_python(
                x, beta, gamma, v0, float(s32), float(zp32), qmin, qmax, gspike, gv_final,
            )
        return gx, gbeta, ggamma, gv0, glog.reshape(()), gzp.reshape(()), None, None


def lsq_fai_lif_apply(
    x: Tensor, beta: Tensor, gamma: Tensor, v0: Tensor,
    log_scale: Tensor, zp: Tensor, qmin: int, qmax: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused FAI-LIF sequence forward with an LSQ-learnable grid.

    ``log_scale`` and ``zp`` are 0-dim FP32 parameters (``s = exp(log_scale)``,
    ``q = clip(round(v_pre / s + zp))``). Returns ``(x_hat, spike_seq,
    v_pre_seq, v_final)`` with ``x_hat = s * (spike - zp)`` the dequantized
    output; gradients flow to ``x``, ``beta``, ``gamma``, ``v0``,
    ``log_scale`` and ``zp``.
    """
    return _LSQFAILIFFunction.apply(x, beta, gamma, v0, log_scale, zp, qmin, qmax)
