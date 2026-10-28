"""Dequantized FAI-LIF output (DECISIONS #15 rung 3): value s * (q - zp),
straight-through gradient 1 w.r.t. the input, zero input -> zero output."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.neurons.fai_lif_op import lsq_fai_lif_apply

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _lif(signed: bool, device: str) -> FAILIF:
    lif = FAILIF(16, num_bits=4, signed=signed, symmetric=True).to(device).eval()
    lif.output_mode = "dequant"
    lif.stateless = True
    lif.observer.running_min.fill_(-1.0 if signed else 0.0)
    lif.observer.running_max.fill_(1.0)
    lif.observer.num_updates.fill_(1)
    return lif


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("signed", [True, False])
def test_symmetric_dequant_tracks_input_with_unit_gradient(device, signed):
    lif = _lif(signed, device)
    scale, zp = lif.observer.get_scale_and_zero_point()
    assert float(zp) == 0.0
    lo = -0.95 if signed else 0.0
    x = (torch.rand(3, 9, 16, generator=torch.Generator().manual_seed(0)) * (0.95 - lo) + lo)
    x = x.to(device).requires_grad_(True)
    out = lif(x)
    assert (out - x).abs().max() <= float(scale) / 2 + 1e-6
    assert torch.equal(lif.last_spike, lif.last_spike.round())
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
    lif.reset_state()
    assert torch.equal(lif(torch.zeros(2, 5, 16, device=device)), torch.zeros(2, 5, 16, device=device))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("ste_clip", [False, True])
def test_lsq_input_gradient_is_straight_through(device, ste_clip):
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(2, 6, 16, generator=g) * 0.5).to(device).requires_grad_(True)
    beta = torch.zeros(16, device=device)
    gamma = torch.ones(16, device=device)
    v0 = torch.zeros(2, 16, device=device)
    log_s = torch.tensor(-2.0, device=device, requires_grad=True)
    zp = torch.tensor(3.0, device=device, requires_grad=True)
    x_hat, *_ = lsq_fai_lif_apply(x, beta, gamma, v0, log_s, zp, 0, 15, ste_clip)
    g_out = torch.randn(x_hat.shape, generator=g).to(device)
    (x_hat * g_out).sum().backward()
    u = x.detach() / torch.exp(log_s.detach()) + zp.detach()
    inside = (u >= -0.5) & (u <= 15.5)
    assert 0 < int(inside.sum()) < inside.numel()
    expected = g_out * inside if ste_clip else g_out
    assert torch.allclose(x.grad, expected, atol=1e-6)
    assert log_s.grad is not None and float(log_s.grad) != 0.0


def test_lsq_keeps_zero_point_fixed_on_symmetric_grids():
    lif = FAILIF(16, num_bits=4, signed=True, symmetric=True)
    lif.enable_lsq()
    assert not lif.lsq_zp.requires_grad and lif.lsq_log_scale.requires_grad
    lif = FAILIF(16, num_bits=4, signed=False)
    lif.enable_lsq()
    assert lif.lsq_zp.requires_grad


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("signed", [True, False])
def test_clipped_ste_zeroes_gradient_outside_the_grid(device, signed):
    lif = _lif(signed, device)
    lif.ste_clip = True
    # Grid [qmin * s, qmax * s]: [-8/7, 1] signed, [0, 1] unsigned.
    x = torch.tensor([0.0, 0.3, -0.3, 0.9, 1.2, -1.2, 3.0, -3.0] * 2).view(1, 1, 16)
    x = x.expand(2, 4, 16).contiguous().to(device).requires_grad_(True)
    out = lif(x)
    out.sum().backward()
    scale = float(lif.observer.get_scale_and_zero_point()[0])
    lo, hi = lif.observer.qmin * scale, lif.observer.qmax * scale
    inside = (x.detach() >= lo - scale / 2) & (x.detach() <= hi + scale / 2)
    assert 0 < int(inside.sum()) < inside.numel()
    assert torch.equal(x.grad, inside.to(x.grad.dtype))
    assert torch.equal(out.detach().clamp(lo, hi), out.detach())
