"""Tests for the fused FAI-LIF CUDA op.

These tests are skipped on CPU-only hosts. They verify:
  * Forward numerical equivalence: kernel vs the pure-Python reference
    using the SAME fixed (scale, zp) for all T steps (1e-5 atol).
  * Backward equivalence: kernel grads vs autograd through the Python
    reference, for all four differentiable inputs (1e-4 atol).
  * Gradcheck on the Python reference in FP64 on CPU.
  * State chains correctly across kernel calls.
  * Integrated ``FAILIF.forward`` on 3D input round-trips through the
    kernel without regressions in the 2D step path.
"""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.neurons.fai_lif_op import (
    _FAILIFFunction,
    _fai_lif_python,
    fai_lif_apply,
)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_inputs(
    B: int = 2,
    T: int = 4,
    H: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    *,
    qmin: int = 0,
    qmax: int = 15,
    scale: float = 0.2,
    zp: float = 0.0,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(B, T, H, generator=g, dtype=torch.float32) * 0.5).to(
        device=device, dtype=dtype
    )
    # beta / gamma in [0, 1] (post-sigmoid). Avoid exact 0/1 for gradcheck.
    beta = torch.sigmoid(torch.randn(H, generator=g, dtype=torch.float32) * 0.5 + 1.5).to(
        device=device, dtype=dtype
    )
    gamma = torch.sigmoid(torch.randn(H, generator=g, dtype=torch.float32) * 0.5 + 0.5).to(
        device=device, dtype=dtype
    )
    v0 = (torch.randn(B, H, generator=g, dtype=torch.float32) * 0.1).to(
        device=device, dtype=dtype
    )
    scale_t = torch.tensor(scale, dtype=dtype, device=device)
    zp_t = torch.tensor(zp, dtype=dtype, device=device)
    return x, beta, gamma, v0, scale_t, zp_t, qmin, qmax


# ---------------------------------------------------------------------------
# Forward equivalence
# ---------------------------------------------------------------------------


# Unbounded membrane, and the rung-4b bound (DECISIONS #19) on a fine grid
# where many outputs saturate, so the bound is active.
MEMBRANE = pytest.mark.parametrize("v_clamp, step", [(0.0, 0.2), (0.5, 0.05)])


@CUDA
@MEMBRANE
def test_forward_matches_reference(v_clamp, step):
    x, beta, gamma, v0, scale, zp, qmin, qmax = _make_inputs(
        B=3, T=16, H=32, device="cuda", scale=step
    )
    spike_c, vpre_c, vfin_c = fai_lif_apply(x, beta, gamma, v0, scale, zp, qmin, qmax, v_clamp)
    spike_r, vpre_r, vfin_r = _fai_lif_python(x, beta, gamma, v0, scale, zp, qmin, qmax, v_clamp)
    assert torch.allclose(spike_c, spike_r, atol=1e-5, rtol=1e-4), (
        f"spike max diff: {(spike_c - spike_r).abs().max().item():.4g}"
    )
    assert torch.allclose(vpre_c, vpre_r, atol=1e-5, rtol=1e-4), (
        f"v_pre max diff: {(vpre_c - vpre_r).abs().max().item():.4g}"
    )
    assert torch.allclose(vfin_c, vfin_r, atol=1e-5, rtol=1e-4), (
        f"v_final max diff: {(vfin_c - vfin_r).abs().max().item():.4g}"
    )


@CUDA
def test_forward_spikes_are_integer_in_range():
    x, beta, gamma, v0, scale, zp, qmin, qmax = _make_inputs(
        B=2, T=8, H=16, device="cuda", qmin=0, qmax=15
    )
    spike_c, _, _ = fai_lif_apply(x, beta, gamma, v0, scale, zp, qmin, qmax)
    assert torch.equal(spike_c, spike_c.round())
    assert (spike_c >= qmin).all()
    assert (spike_c <= qmax).all()


# ---------------------------------------------------------------------------
# Backward equivalence (CUDA kernel vs Python autograd)
# ---------------------------------------------------------------------------


@CUDA
@MEMBRANE
def test_backward_matches_python_autograd(v_clamp, step):
    x, beta, gamma, v0, scale, zp, qmin, qmax = _make_inputs(
        B=3, T=16, H=32, device="cuda", scale=step
    )

    # Reference grads via autograd on the Python forward.
    x_r = x.detach().clone().requires_grad_(True)
    beta_r = beta.detach().clone().requires_grad_(True)
    gamma_r = gamma.detach().clone().requires_grad_(True)
    v0_r = v0.detach().clone().requires_grad_(True)
    spike_r, _vpre_r, vfin_r = _fai_lif_python(
        x_r, beta_r, gamma_r, v0_r, scale, zp, qmin, qmax, v_clamp
    )
    gspike = torch.randn_like(spike_r)
    gvfin = torch.randn_like(vfin_r)
    loss_r = (spike_r * gspike).sum() + (vfin_r * gvfin).sum()
    loss_r.backward()

    # Kernel grads.
    x_c = x.detach().clone().requires_grad_(True)
    beta_c = beta.detach().clone().requires_grad_(True)
    gamma_c = gamma.detach().clone().requires_grad_(True)
    v0_c = v0.detach().clone().requires_grad_(True)
    spike_c, _vpre_c, vfin_c = _FAILIFFunction.apply(
        x_c, beta_c, gamma_c, v0_c, scale, zp, qmin, qmax, v_clamp
    )
    loss_c = (spike_c * gspike).sum() + (vfin_c * gvfin).sum()
    loss_c.backward()

    for name, ref, got in [
        ("x", x_r.grad, x_c.grad),
        ("beta", beta_r.grad, beta_c.grad),
        ("gamma", gamma_r.grad, gamma_c.grad),
        ("v0", v0_r.grad, v0_c.grad),
    ]:
        assert ref is not None, f"reference grad for {name} is None"
        assert got is not None, f"kernel grad for {name} is None"
        max_diff = (ref - got).abs().max().item()
        assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3), (
            f"grad mismatch for {name}: max_diff={max_diff:.4g}"
        )


def test_membrane_clamp_bounds_the_state_and_its_gradient():
    """Anti-windup (DECISIONS #19): after every reset the membrane is within
    ±v_clamp·s, and a bounded membrane passes no gradient back to v_pre."""
    x, beta, gamma, v0, scale, zp, qmin, qmax = _make_inputs(B=3, T=16, H=32, scale=0.05)
    bound = 0.5 * 0.05
    _spike, vpre, vfin = _fai_lif_python(x, beta, gamma, v0, scale, zp, qmin, qmax, 0.5)
    v_prev = (vpre[:, 1:] - x[:, 1:]) / beta          # membrane entering steps 1..T-1
    assert v_prev.abs().max() <= bound + 1e-6 and vfin.abs().max() <= bound + 1e-6
    # The grid saturates on this input, so the bound must have been active.
    raw = gamma * (vpre - scale * ((vpre / scale + zp).round().clamp(qmin, qmax) - zp))
    assert (raw.abs() > bound).any()

    # One step: d v_final / d x is gamma inside the bound, 0 where it clamps.
    x1 = x[:, :1].clone().requires_grad_(True)
    v0z = torch.zeros_like(v0)
    _s, vpre1, vfin1 = _fai_lif_python(x1, beta, gamma, v0z, scale, zp, qmin, qmax, 0.5)
    vfin1.sum().backward()
    raw1 = gamma * (vpre1[:, 0] - scale * ((vpre1[:, 0] / scale + zp).round().clamp(qmin, qmax) - zp))
    expected = gamma * (raw1.abs() <= bound).to(x.dtype)
    assert torch.allclose(x1.grad[:, 0], expected.detach())


# ---------------------------------------------------------------------------
# Gradcheck on the Python reference (FP64 on CPU)
# ---------------------------------------------------------------------------


def _fai_lif_ste_shadow(
    x: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, v0: torch.Tensor,
    scale: torch.Tensor, zp: torch.Tensor, qmin: int, qmax: int,
):
    """STE-shadow forward used by ``test_gradcheck_python_reference_double``.

    The STE-identity quantize replaces the spike output's GRADIENT with
    that of ``v_pre`` (forward emits a step function, backward acts as
    identity). ``torch.autograd.gradcheck`` compares the analytical
    gradient to a finite-difference of the actual forward — which is
    locally constant for an STE step, so the FD gradient is 0 while the
    analytic gradient is 1. This is a well-known gradcheck limitation
    for STE ops (and the reason the WKV gradcheck only tests the smooth
    forward), not a bug.

    To still gradcheck the surrounding LIF recurrence and the reset path,
    we substitute the STE-shadow ``spike = v_pre`` for the forward AND
    backward — i.e., the same code path the STE-identity gradient would
    take in autograd, but with a SMOOTH forward that FD can probe. The
    dequant for the reset uses the actual quantized value (which is
    detached, so it doesn't affect grads).
    """
    B, T, H = x.shape
    v = v0
    spikes = []
    v_pres = []
    inv_scale = 1.0 / scale
    for t in range(T):
        x_t = x[:, t]
        v_pre = beta * v + x_t
        # Shadow: spike == v_pre for backward (matches STE-identity).
        spike = v_pre
        q = (v_pre * inv_scale + zp).round().clamp(qmin, qmax)
        dequant = (scale * q).detach()
        v_new = gamma * (v_pre - dequant)
        spikes.append(spike)
        v_pres.append(v_pre)
        v = v_new
    return torch.stack(spikes, dim=1), torch.stack(v_pres, dim=1), v


def test_gradcheck_python_reference_double():
    """Gradcheck the STE-shadow forward in FP64.

    See ``_fai_lif_ste_shadow`` for why we can't directly gradcheck the
    actual forward (STE step-function is locally constant under
    finite-difference perturbation). This test validates the LIF
    recurrence + reset-path gradients; the STE-identity itself is exact
    by construction.
    """
    B, T, H = 1, 4, 6
    qmin, qmax = 0, 15
    scale = 0.5
    zp = 0.0

    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(B, T, H, generator=g, dtype=torch.float64) * 0.3 + 0.4
    beta = torch.full((H,), 0.8, dtype=torch.float64)
    gamma = torch.full((H,), 0.5, dtype=torch.float64)
    v0 = torch.zeros(B, H, dtype=torch.float64)
    scale_t = torch.tensor(scale, dtype=torch.float64)
    zp_t = torch.tensor(zp, dtype=torch.float64)

    x.requires_grad_(True)
    beta.requires_grad_(True)
    gamma.requires_grad_(True)
    v0.requires_grad_(True)

    def f(x, beta, gamma, v0):
        spike, _vpre, vfin = _fai_lif_ste_shadow(
            x, beta, gamma, v0, scale_t, zp_t, qmin, qmax
        )
        return spike, vfin

    torch.autograd.gradcheck(
        f, (x, beta, gamma, v0), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0
    )


# ---------------------------------------------------------------------------
# State chains correctly across calls
# ---------------------------------------------------------------------------


@CUDA
def test_state_chains_across_kernel_calls():
    """Calling kernel twice with v0 = previous v_final equals one call of
    the concatenated sequence."""
    x1, beta, gamma, v0, scale, zp, qmin, qmax = _make_inputs(
        B=2, T=8, H=12, device="cuda", seed=0
    )
    x2, _, _, _, _, _, _, _ = _make_inputs(B=2, T=8, H=12, device="cuda", seed=1)

    # Two-call chain.
    s1, _, vmid = fai_lif_apply(x1, beta, gamma, v0, scale, zp, qmin, qmax)
    s2, _, vfin = fai_lif_apply(x2, beta, gamma, vmid, scale, zp, qmin, qmax)
    s_chain = torch.cat([s1, s2], dim=1)

    # Single concatenated call.
    x_full = torch.cat([x1, x2], dim=1)
    s_full, _, vfin_full = fai_lif_apply(
        x_full, beta, gamma, v0, scale, zp, qmin, qmax
    )

    assert torch.allclose(s_chain, s_full, atol=1e-5, rtol=1e-4)
    assert torch.allclose(vfin, vfin_full, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Integrated FAILIF.forward 3D path
# ---------------------------------------------------------------------------


@CUDA
def test_failif_3d_forward_runs():
    """FAILIF.forward on a (B, T, H) tensor runs without error and
    returns integer-valued spikes in range."""
    lif = FAILIF(num_neurons=32, num_bits=4).cuda()
    # Warm the observer first via the 2D path so scale is sensible.
    lif.train()
    for _ in range(20):
        lif(torch.randn(4, 32, device="cuda"))
    # Reset state for a clean (B, T, H) call.
    lif.reset_state()
    spike = lif(torch.randn(4, 16, 32, device="cuda") * 2)
    assert spike.shape == (4, 16, 32)
    assert torch.equal(spike, spike.round())
    assert (spike >= 0).all()
    assert (spike <= 15).all()


@CUDA
def test_failif_3d_observer_update_runs():
    """After a 3D forward in train mode, the observer's num_updates has
    advanced by exactly T (one EMA tick per timestep, matching the
    per-step Python reference's update stream)."""
    T = 8
    lif = FAILIF(num_neurons=16, num_bits=4).cuda()
    lif.train()
    n0 = int(lif.observer.num_updates)
    lif(torch.randn(2, T, 16, device="cuda"))
    n1 = int(lif.observer.num_updates)
    assert n1 == n0 + T


def test_observer_update_sequence_matches_per_step_loop():
    """``update_stats_sequence`` must produce the *exact* same EMA state
    as T sequential calls to ``update_stats`` on each timestep slice.

    This is the load-bearing equivalence test for the per-step EMA fix:
    it isolates the observer's update stream from the kernel's frozen
    (scale, zp) by feeding both paths identical per-step (min, max)
    streams. If this passes, the kernel's 3D path replicates the
    per-step path's observer evolution to within FP32 noise — which is
    the property that was failing to recover quality in midrun_fast.
    """
    from spiking_ternary_lm.quantization.observers import (
        MovingAverageMinMaxObserver,
    )

    torch.manual_seed(0)
    B, T, H = 4, 16, 24

    # Synthetic v_pre sequence. We don't need the actual LIF here —
    # this isolates the observer update stream.
    v_pre = torch.randn(B, T, H) * 2.0 + 0.3

    # Path A: T sequential update_stats() calls, one per timestep slice
    # (this is what the per-step Python reference does).
    obs_a = MovingAverageMinMaxObserver(
        num_bits=4, signed=False, momentum=0.05
    )
    obs_a.train()
    for t in range(T):
        obs_a.update_stats(v_pre[:, t])

    # Path B: one update_stats_sequence() call with the (T,) per-step
    # min/max reductions over (B, H).
    obs_b = MovingAverageMinMaxObserver(
        num_bits=4, signed=False, momentum=0.05
    )
    obs_b.train()
    step_mins = v_pre.amin(dim=(0, 2))
    step_maxes = v_pre.amax(dim=(0, 2))
    obs_b.update_stats_sequence(step_mins, step_maxes)

    assert torch.allclose(obs_a.running_min, obs_b.running_min, atol=1e-6, rtol=1e-5), (
        f"running_min mismatch: per_step={obs_a.running_min.item():.6g} "
        f"sequence={obs_b.running_min.item():.6g}"
    )
    assert torch.allclose(obs_a.running_max, obs_b.running_max, atol=1e-6, rtol=1e-5), (
        f"running_max mismatch: per_step={obs_a.running_max.item():.6g} "
        f"sequence={obs_b.running_max.item():.6g}"
    )
    assert int(obs_a.num_updates) == int(obs_b.num_updates) == T

    # Also test the warm-start case: pre-warm the observer past the
    # first-call init branch so both paths exercise pure EMA blending.
    obs_c = MovingAverageMinMaxObserver(num_bits=4, momentum=0.05)
    obs_c.train()
    obs_c.update_stats(torch.randn(8) * 3 + 1)  # init step
    obs_d = MovingAverageMinMaxObserver(num_bits=4, momentum=0.05)
    obs_d.load_state_dict(obs_c.state_dict())
    for t in range(T):
        obs_c.update_stats(v_pre[:, t])
    obs_d.update_stats_sequence(step_mins, step_maxes)
    assert torch.allclose(obs_c.running_min, obs_d.running_min, atol=1e-6, rtol=1e-5)
    assert torch.allclose(obs_c.running_max, obs_d.running_max, atol=1e-6, rtol=1e-5)
    assert int(obs_c.num_updates) == int(obs_d.num_updates)


@CUDA
def test_failif_3d_state_persists():
    """Calling FAILIF on two consecutive (B, T, H) chunks should yield
    the same result as one call on the concatenated chunk."""
    lif_a = FAILIF(num_neurons=8, num_bits=4).cuda()
    # Warm the observer so scale is meaningful.
    lif_a.train()
    for _ in range(20):
        lif_a(torch.randn(2, 8, device="cuda"))
    # Snapshot post-warmup observer stats to mirror in lif_b.
    lif_b = FAILIF(num_neurons=8, num_bits=4).cuda()
    lif_b.load_state_dict(lif_a.state_dict())
    lif_b.observer.running_min.copy_(lif_a.observer.running_min)
    lif_b.observer.running_max.copy_(lif_a.observer.running_max)
    lif_b.observer.num_updates.copy_(lif_a.observer.num_updates)

    # Freeze observers so the within-call update doesn't desync the two
    # paths (different number of forward calls => different EMA state).
    lif_a.eval()
    lif_b.eval()
    lif_a.reset_state()
    lif_b.reset_state()

    g = torch.Generator(device="cpu").manual_seed(0)
    x1 = torch.randn(3, 6, 8, generator=g).cuda()
    x2 = torch.randn(3, 6, 8, generator=g).cuda()

    s1 = lif_a(x1)
    s2 = lif_a(x2)
    s_chain = torch.cat([s1, s2], dim=1)

    s_full = lif_b(torch.cat([x1, x2], dim=1))
    assert torch.allclose(s_chain, s_full, atol=1e-5, rtol=1e-4)


@CUDA
def test_failif_3d_gradient_flows():
    """End-to-end autograd through FAILIF on the 3D path produces grad on x."""
    lif = FAILIF(num_neurons=16, num_bits=4).cuda()
    lif.train()
    for _ in range(10):
        lif(torch.randn(4, 16, device="cuda"))
    lif.reset_state()
    x = torch.randn(4, 8, 16, device="cuda", requires_grad=True)
    s = lif(x)
    s.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
