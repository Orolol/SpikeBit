"""Per-channel MinMax observer (DECISIONS #14)."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.quantization.observers import MovingAverageMinMaxObserver

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_per_channel_sequence_update_matches_independent_per_tensor_observers():
    torch.manual_seed(0)
    B, T, C = 4, 16, 6
    v = torch.randn(B, T, C) * 2 + 0.3
    obs_c = MovingAverageMinMaxObserver(num_bits=4, momentum=0.05, channel_dim=C)
    obs_c.train()
    obs_c.update_stats(torch.randn(8, C) * 3 + 1)  # warm start
    ref = []
    for c in range(C):
        o = MovingAverageMinMaxObserver(num_bits=4, momentum=0.05)
        o.train()
        o.running_min.copy_(obs_c.running_min_c[c]); o.running_max.copy_(obs_c.running_max_c[c])
        o.num_updates.copy_(obs_c.num_updates)
        for t in range(T):
            o.update_stats(v[:, t, c])
        ref.append(o)
    obs_c.update_stats_sequence(v.amin(dim=0), v.amax(dim=0))
    for c, o in enumerate(ref):
        assert torch.allclose(obs_c.running_min_c[c], o.running_min, atol=1e-6, rtol=1e-5)
        assert torch.allclose(obs_c.running_max_c[c], o.running_max, atol=1e-6, rtol=1e-5)
    assert int(obs_c.num_updates) == int(ref[0].num_updates)
    scale, zp = obs_c.get_scale_and_zero_point()
    assert scale.shape == (C,) and zp.shape == (C,)
    s0, z0 = ref[0].get_scale_and_zero_point()
    assert torch.allclose(scale[0], s0) and torch.allclose(zp[0], z0)


def test_per_channel_first_call_init_is_per_channel():
    C = 5
    obs = MovingAverageMinMaxObserver(num_bits=4, momentum=0.01, channel_dim=C)
    obs.train()
    v = torch.randn(3, 7, C) * torch.arange(1, C + 1).float()
    obs.update_stats_sequence(v.amin(dim=0), v.amax(dim=0))
    assert obs.running_min_c.shape == (C,) and torch.isfinite(obs.running_min_c).all()
    assert obs.running_min_c.std() > 0 and int(obs.num_updates) == 7


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_failif_per_channel_forward_and_loading(device):
    torch.manual_seed(0)
    H = 8
    lif_t = FAILIF(num_neurons=H, num_bits=4, signed=False).to(device)
    lif_t.train()
    x = torch.randn(3, 10, H, device=device) * torch.arange(1, H + 1, device=device)
    lif_t(x)
    sd = lif_t.state_dict()  # per-tensor checkpoint
    lif_c = FAILIF(num_neurons=H, num_bits=4, signed=False).to(device)
    lif_c.set_observer_scope("channel")
    missing, unexpected = lif_c.load_state_dict(sd, strict=False)
    assert set(missing) == {"observer.running_min_c", "observer.running_max_c"} and not unexpected
    lif_c.observer.num_updates.zero_()  # fresh per-channel init from the first batch
    lif_c.train()
    out = lif_c(x)
    assert out.shape == x.shape and torch.equal(out, out.round())
    scale, zp = lif_c.observer.get_scale_and_zero_point()
    assert scale.shape == (H,)
    # Per-channel grids differ, and the widest channel gets the widest grid.
    assert scale.std() > 0 and scale[-1] > scale[0]
    out.sum().backward()
    assert lif_c.beta_logit.grad is not None
    # Per-step (2D) path and eval mode work with the (H,) grid.
    lif_c.eval()
    o2 = lif_c(x[:, 0])
    assert o2.shape == (3, H)


@CUDA
def test_per_channel_kernel_matches_python_reference():
    from spiking_ternary_lm.neurons.fai_lif_op import _fai_lif_python, fai_lif_apply
    torch.manual_seed(0)
    B, T, H = 2, 9, 6
    x = torch.randn(B, T, H, device="cuda", requires_grad=True)
    beta = torch.full((H,), 0.9, device="cuda", requires_grad=True)
    gamma = torch.full((H,), 0.5, device="cuda", requires_grad=True)
    v0 = torch.zeros(B, H, device="cuda")
    scale = torch.rand(H, device="cuda") * 0.5 + 0.2
    zp = torch.randint(0, 5, (H,), device="cuda").float()
    s_c, vp_c, vf_c = fai_lif_apply(x, beta, gamma, v0, scale, zp, 0, 15)
    xr = x.detach().clone().requires_grad_(True)
    br = beta.detach().clone().requires_grad_(True)
    gr = gamma.detach().clone().requires_grad_(True)
    s_p, vp_p, vf_p = _fai_lif_python(xr, br, gr, v0, scale, zp, 0, 15)
    assert torch.equal(s_c, s_p.round())
    assert torch.allclose(vp_c, vp_p, atol=1e-5) and torch.allclose(vf_c, vf_p, atol=1e-5)
    g = torch.randn_like(s_c)
    (s_c * g).sum().backward()
    (s_p * g).sum().backward()
    assert torch.allclose(x.grad, xr.grad, atol=1e-5)
    assert torch.allclose(beta.grad, br.grad, atol=1e-4) and torch.allclose(gamma.grad, gr.grad, atol=1e-4)
