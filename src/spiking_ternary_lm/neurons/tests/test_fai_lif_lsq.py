"""LSQ activation quantizer for FAI-LIF (DECISIONS #14)."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.neurons.fai_lif import FAILIF, enable_lsq
from spiking_ternary_lm.neurons.fai_lif_op import _lsq_grad_scale, lsq_fai_lif_apply

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _inputs(device, B=3, T=7, H=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(B, T, H, generator=g) * 2.0).to(device)
    beta = torch.full((H,), 0.9).to(device)
    gamma = torch.full((H,), 0.5).to(device)
    v0 = torch.zeros(B, H).to(device)
    return x, beta, gamma, v0


def _surrogate_grads(x, beta, gamma, v0, log_s, zp, qmin, qmax, gspike):
    """Analytic LSQ gradient = autograd of sum(g * clamp(v_pre/s + zp)) with
    v_pre held fixed (the STE-through-round, detached-reset convention)."""
    _, v_pre, _ = lsq_fai_lif_apply(x, beta, gamma, v0, log_s.detach(), zp.detach(), qmin, qmax)
    ls_r = log_s.detach().clone().requires_grad_(True)
    zp_r = zp.detach().clone().requires_grad_(True)
    u = v_pre.detach() / ls_r.exp() + zp_r
    inside = ((u > qmin) & (u < qmax)).float()
    surrogate = (gspike * u * inside).sum() * _lsq_grad_scale(u.numel(), qmin, qmax)
    return torch.autograd.grad(surrogate, (ls_r, zp_r))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("signed", [False, True])
def test_lsq_step_and_offset_gradients(device, signed):
    qmin, qmax = (-8, 7) if signed else (0, 15)
    x, beta, gamma, v0 = _inputs(device)
    s = torch.tensor(-0.36, device=device, requires_grad=True)  # log-scale
    off = torch.tensor(0.4, device=device, requires_grad=True)   # zero-point
    spike, v_pre, v_final = lsq_fai_lif_apply(x, beta, gamma, v0, s, off, qmin, qmax)
    gspike = torch.randn_like(spike)
    (spike * gspike).sum().backward()
    gs_ref, goff_ref = _surrogate_grads(x, beta, gamma, v0, s, off, qmin, qmax, gspike)
    assert torch.allclose(s.grad, gs_ref, atol=1e-5, rtol=1e-4), (s.grad, gs_ref)
    assert torch.allclose(off.grad, goff_ref, atol=1e-5, rtol=1e-4), (off.grad, goff_ref)
    assert spike.min() >= qmin and spike.max() <= qmax
    assert torch.equal(spike, spike.round())


@CUDA
def test_lsq_cuda_matches_python_reference():
    qmin, qmax = 0, 15
    x, beta, gamma, v0 = _inputs("cuda")
    s = torch.tensor(-0.69, device="cuda", requires_grad=True)
    off = torch.tensor(0.1, device="cuda", requires_grad=True)
    spike_c, v_pre_c, v_final_c = lsq_fai_lif_apply(x, beta, gamma, v0, s, off, qmin, qmax)
    xc = x.cpu(); sc = s.detach().cpu().requires_grad_(True); offc = off.detach().cpu().requires_grad_(True)
    spike_p, v_pre_p, v_final_p = lsq_fai_lif_apply(xc, beta.cpu(), gamma.cpu(), v0.cpu(), sc, offc, qmin, qmax)
    assert torch.equal(spike_c.cpu(), spike_p)
    assert torch.allclose(v_pre_c.cpu(), v_pre_p, atol=1e-5)
    g = torch.randn_like(spike_p)
    (spike_c * g.cuda()).sum().backward()
    (spike_p * g).sum().backward()
    assert torch.allclose(s.grad.cpu(), sc.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(off.grad.cpu(), offc.grad, atol=1e-5, rtol=1e-4)


def test_lsq_init_from_observer_then_learnable():
    torch.manual_seed(0)
    lif = FAILIF(num_neurons=16, num_bits=4, signed=False)
    lif.enable_lsq(init_steps=2)
    lif.train()
    x = torch.randn(4, 6, 16) * 3
    lif(x)
    assert int(lif.lsq_initialized) == 0 and int(lif.observer.num_updates) == 6
    lif(x)
    assert int(lif.lsq_initialized) == 1
    scale, zp = lif.observer.get_scale_and_zero_point()
    assert torch.allclose(lif.lsq_log_scale.exp(), scale)
    assert torch.allclose(lif.lsq_zp, zp)
    n_before = int(lif.observer.num_updates)
    out = lif(x)
    assert int(lif.observer.num_updates) == n_before, "observer must stop tracking after init"
    out.sum().backward()
    assert lif.lsq_log_scale.grad is not None and lif.lsq_zp.grad is not None
    # Eval, per-step path, uses the learned grid.
    lif.eval()
    out2 = lif(x[:, 0])
    assert out2.shape == (4, 16)
    # Round-trip: the LSQ buffers/params survive a state_dict reload.
    lif2 = FAILIF(num_neurons=16, num_bits=4, signed=False)
    lif2.enable_lsq()
    lif2.load_state_dict(lif.state_dict())
    assert int(lif2.lsq_initialized) == 1 and torch.allclose(lif2.lsq_log_scale, lif.lsq_log_scale)


def test_enable_lsq_helper_counts_and_params():
    from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
    from spiking_ternary_lm.model.model import SpikingTernaryRWKV
    cfg = SpikingTernaryRWKVConfig.tiny()
    m = SpikingTernaryRWKV(cfg)
    n_lif = sum(isinstance(mod, FAILIF) for mod in m.modules())
    assert enable_lsq(m) == n_lif
    assert sum(1 for k in m.state_dict() if k.endswith("lsq_log_scale")) == n_lif
    cfg.act_quant = "lsq"
    m2 = SpikingTernaryRWKV(cfg)
    assert m2.blocks[0].time_mix.lif_k.act_quant == "lsq"
