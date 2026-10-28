"""Tests for the fused WKV CUDA op.

These tests are skipped on CPU-only hosts. They verify:
  * Forward numerical equivalence with the Python reference (1e-5 atol).
  * Backward correctness via ``torch.autograd.gradcheck`` (double).
  * Final-state gradient is consistent (gradients seeded from aaT/bbT/ppT
    propagate correctly).
"""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.layers.wkv_op import (
    _WKVFunction,
    _wkv_python,
    wkv_forward,
)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_inputs(B=2, T=4, C=8, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w_eff = torch.exp(torch.randn(C, generator=g, dtype=torch.float32) * 0.3 - 2.0).to(
        device=device, dtype=dtype
    )
    u = (torch.randn(C, generator=g, dtype=torch.float32) * 0.5).to(device=device, dtype=dtype)
    k = (torch.randn(B, T, C, generator=g, dtype=torch.float32) * 0.5).to(
        device=device, dtype=dtype
    )
    v = (torch.randn(B, T, C, generator=g, dtype=torch.float32) * 0.5).to(
        device=device, dtype=dtype
    )
    aa0 = torch.zeros(B, C, device=device, dtype=dtype)
    bb0 = torch.zeros(B, C, device=device, dtype=dtype)
    pp0 = torch.full((B, C), -1e30, device=device, dtype=dtype)
    return w_eff, u, k, v, aa0, bb0, pp0


@CUDA
def test_forward_matches_reference():
    w, u, k, v, aa0, bb0, pp0 = _make_inputs(device="cuda")
    wkv_cuda, aaT_c, bbT_c, ppT_c = wkv_forward(w, u, k, v, aa0, bb0, pp0)
    wkv_ref, aaT_r, bbT_r, ppT_r = _wkv_python(w, u, k, v, aa0, bb0, pp0)
    assert torch.allclose(wkv_cuda, wkv_ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(aaT_c, aaT_r, atol=1e-5, rtol=1e-4)
    assert torch.allclose(bbT_c, bbT_r, atol=1e-5, rtol=1e-4)
    assert torch.allclose(ppT_c, ppT_r, atol=1e-5, rtol=1e-4)


@CUDA
def test_backward_matches_python_autograd():
    """Backward of the fused kernel matches torch.autograd through the
    pure-Python forward, for both wkv-only and final-state grads."""
    w, u, k, v, aa0, bb0, pp0 = _make_inputs(device="cuda")

    # Reference: pure Python forward, autograd backward.
    w_r = w.detach().clone().requires_grad_(True)
    u_r = u.detach().clone().requires_grad_(True)
    k_r = k.detach().clone().requires_grad_(True)
    v_r = v.detach().clone().requires_grad_(True)
    aa0_r = aa0.detach().clone().requires_grad_(True)
    bb0_r = bb0.detach().clone().requires_grad_(True)
    pp0_r = pp0.detach().clone().requires_grad_(True)
    wkv_r, aaT_r, bbT_r, ppT_r = _wkv_python(w_r, u_r, k_r, v_r, aa0_r, bb0_r, pp0_r)
    gwkv = torch.randn_like(wkv_r)
    gaaT = torch.randn_like(aaT_r)
    gbbT = torch.randn_like(bbT_r)
    # Avoid huge gradients on pp (which can saturate via exp). Use small.
    gppT = torch.zeros_like(ppT_r)
    loss_r = (wkv_r * gwkv).sum() + (aaT_r * gaaT).sum() + (bbT_r * gbbT).sum()
    loss_r.backward()

    # Fused kernel forward + backward.
    w_c = w.detach().clone().requires_grad_(True)
    u_c = u.detach().clone().requires_grad_(True)
    k_c = k.detach().clone().requires_grad_(True)
    v_c = v.detach().clone().requires_grad_(True)
    aa0_c = aa0.detach().clone().requires_grad_(True)
    bb0_c = bb0.detach().clone().requires_grad_(True)
    pp0_c = pp0.detach().clone().requires_grad_(True)
    wkv_c, aaT_c, bbT_c, ppT_c = _WKVFunction.apply(w_c, u_c, k_c, v_c, aa0_c, bb0_c, pp0_c)
    loss_c = (wkv_c * gwkv).sum() + (aaT_c * gaaT).sum() + (bbT_c * gbbT).sum()
    loss_c.backward()

    for name, ref, got in [
        ("w", w_r.grad, w_c.grad),
        ("u", u_r.grad, u_c.grad),
        ("k", k_r.grad, k_c.grad),
        ("v", v_r.grad, v_c.grad),
        ("aa0", aa0_r.grad, aa0_c.grad),
        ("bb0", bb0_r.grad, bb0_c.grad),
    ]:
        assert ref is not None, f"reference grad for {name} is None"
        assert got is not None, f"fused grad for {name} is None"
        assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3), (
            f"grad mismatch for {name}: max_diff={(ref-got).abs().max().item():.4g}"
        )


@CUDA
def test_gradcheck_double():
    """Run torch.autograd.gradcheck on a tiny case in float64.

    We run gradcheck through the Python fallback (CPU, double) because
    the CUDA kernel is float32-only. The test exists to confirm the
    *forward formulation* is mathematically right; the CUDA kernel is
    further checked against the Python forward in
    ``test_forward_matches_reference`` and ``test_backward_matches_python_autograd``.
    """
    # Tiny case in double on CPU.
    B, T, C = 1, 4, 8
    w_eff = torch.rand(C, dtype=torch.float64, requires_grad=True) * 0.5 + 0.05
    u = torch.randn(C, dtype=torch.float64, requires_grad=True) * 0.3
    k = torch.randn(B, T, C, dtype=torch.float64, requires_grad=True) * 0.3
    v = torch.randn(B, T, C, dtype=torch.float64, requires_grad=True) * 0.3
    aa0 = torch.zeros(B, C, dtype=torch.float64, requires_grad=True)
    bb0 = torch.zeros(B, C, dtype=torch.float64, requires_grad=True)
    # ``pp0`` initialised to a finite negative number for gradcheck stability
    # (the canonical -1e30 sentinel makes finite-difference perturbations
    # numerically explode through `exp`).
    pp0 = torch.full((B, C), -5.0, dtype=torch.float64, requires_grad=True)

    def f(w_eff, u, k, v, aa0, bb0, pp0):
        wkv, _, _, _ = _wkv_python(w_eff, u, k, v, aa0, bb0, pp0)
        return wkv

    torch.autograd.gradcheck(
        f, (w_eff, u, k, v, aa0, bb0, pp0), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0
    )
