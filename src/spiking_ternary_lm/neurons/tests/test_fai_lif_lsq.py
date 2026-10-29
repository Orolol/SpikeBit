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


def _lsq_reference_grads(x, beta, gamma, v0, log_s, zp, qmin, qmax, g_out):
    """Independent formulation: the standard LSQ forward (Esser et al.) written
    with the round-pass / grad-scale tricks of the reference implementations,
    applied to the membrane trajectory held fixed. Autograd of it gives the
    LSQ gradients for s (via log_s) and zp."""
    _, _, v_pre, _ = lsq_fai_lif_apply(x, beta, gamma, v0, log_s.detach(), zp.detach(), qmin, qmax)
    ls_r = log_s.detach().clone().requires_grad_(True)
    zp_r = zp.detach().clone().requires_grad_(True)
    s = ls_r.exp()
    gscale = _lsq_grad_scale(v_pre.numel(), qmin, qmax)
    s_g = (s - s * gscale).detach() + s * gscale          # grad_scale trick
    zp_g = (zp_r - zp_r * gscale).detach() + zp_r * gscale
    u = (v_pre.detach() / s_g + zp_g).clamp(qmin, qmax)
    q = (u.round() - u).detach() + u                       # round-pass trick
    x_hat = s_g * (q - zp_g)
    return torch.autograd.grad((x_hat * g_out).sum(), (ls_r, zp_r))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("signed", [False, True])
def test_lsq_step_and_offset_gradients(device, signed):
    qmin, qmax = (-8, 7) if signed else (0, 15)
    x, beta, gamma, v0 = _inputs(device)
    log_s = torch.tensor(-0.36, device=device, requires_grad=True)
    zp = torch.tensor(0.4, device=device, requires_grad=True)
    x_hat, spike, v_pre, v_final = lsq_fai_lif_apply(x, beta, gamma, v0, log_s, zp, qmin, qmax)
    g_out = torch.randn_like(x_hat)
    (x_hat * g_out).sum().backward()
    gs_ref, gzp_ref = _lsq_reference_grads(x, beta, gamma, v0, log_s, zp, qmin, qmax, g_out)
    assert torch.allclose(log_s.grad, gs_ref, atol=1e-5, rtol=1e-4), (log_s.grad, gs_ref)
    assert torch.allclose(zp.grad, gzp_ref, atol=1e-5, rtol=1e-4), (zp.grad, gzp_ref)
    assert spike.min() >= qmin and spike.max() <= qmax
    assert torch.equal(spike, spike.round())
    assert torch.allclose(x_hat, log_s.exp().detach() * (spike - zp.detach()))
    # Some elements must be clipped for the clipped-branch to be exercised.
    u = v_pre / log_s.exp().detach() + zp.detach()
    assert ((u <= qmin) | (u >= qmax)).any()


@CUDA
@pytest.mark.parametrize("v_clamp", [0.0, 0.5])
def test_lsq_cuda_matches_python_reference(v_clamp):
    qmin, qmax = 0, 15
    x, beta, gamma, v0 = _inputs("cuda")
    s = torch.tensor(-0.69, device="cuda", requires_grad=True)
    off = torch.tensor(0.1, device="cuda", requires_grad=True)
    xh_c, spike_c, v_pre_c, v_final_c = lsq_fai_lif_apply(
        x, beta, gamma, v0, s, off, qmin, qmax, False, v_clamp
    )
    xc = x.cpu(); sc = s.detach().cpu().requires_grad_(True); offc = off.detach().cpu().requires_grad_(True)
    xh_p, spike_p, v_pre_p, v_final_p = lsq_fai_lif_apply(
        xc, beta.cpu(), gamma.cpu(), v0.cpu(), sc, offc, qmin, qmax, False, v_clamp
    )
    assert torch.equal(spike_c.cpu(), spike_p)
    assert torch.allclose(v_pre_c.cpu(), v_pre_p, atol=1e-5)
    assert torch.allclose(v_final_c.cpu(), v_final_p, atol=1e-5)
    g = torch.randn_like(spike_p)
    (xh_c * g.cuda()).sum().backward()
    (xh_p * g).sum().backward()
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
    assert lif.last_spike is not None and torch.equal(lif.last_spike, lif.last_spike.round())
    # Output is the dequantized spike: s * (q - zp).
    assert torch.allclose(out, lif.lsq_log_scale.exp().detach() * (lif.last_spike - lif.lsq_zp.detach()))
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
