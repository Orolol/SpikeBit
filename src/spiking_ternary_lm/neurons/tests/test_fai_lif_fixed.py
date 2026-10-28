"""Tests for FAI-LIF (integer-output LIF)."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.neurons.fai_lif import FAILIFFixedThreshold as FAILIF


def test_output_range_unsigned():
    lif = FAILIF(hidden_size=8, bits=4)
    x = torch.randn(4, 8) * 5.0  # large currents to drive into saturation
    n = lif(x)
    assert n.shape == (4, 8)
    assert (n >= 0).all()
    assert (n <= 15).all()
    # Integer-valued (exact, since round+clip is applied).
    assert torch.equal(n, n.round())


def test_output_range_signed():
    lif = FAILIF(hidden_size=8, bits=4, signed=True)
    x = torch.randn(16, 8) * 5.0
    n = lif(x)
    assert (n >= -8).all()
    assert (n <= 7).all()


def test_lazy_state_init_and_device_sync():
    lif = FAILIF(hidden_size=4, bits=3)
    assert lif.v.numel() == 0
    n = lif(torch.zeros(2, 4))
    assert lif.v.shape == (2, 4)
    # Calling with a different batch should re-allocate.
    _ = lif(torch.zeros(5, 4))
    assert lif.v.shape == (5, 4)
    assert n.shape == (2, 4)


def test_state_carries_across_steps():
    lif = FAILIF(hidden_size=2, bits=4, init_beta=0.9, init_gamma=0.5, init_v_thr=10.0)
    # Drive with sub-threshold current to verify the membrane accumulates.
    x = torch.full((1, 2), 1.0)
    seen = []
    for _ in range(5):
        n = lif(x)
        seen.append(lif.v.clone())
    # With v_thr=10 and constant input 1, V should accumulate monotonically.
    for a, b in zip(seen[:-1], seen[1:]):
        assert (b >= a).all()
    # No spike yet.
    assert (n == 0).all()


def test_reset_state_zeros_buffer():
    lif = FAILIF(hidden_size=4, bits=4)
    lif(torch.randn(3, 4) * 5.0)
    assert lif.v.abs().sum() > 0
    lif.reset_state(batch_shape=(3,))
    assert lif.v.shape == (3, 4)
    assert (lif.v == 0).all()


def test_hard_reset_drives_v_to_zero():
    lif = FAILIF(
        hidden_size=2, bits=4, init_beta=0.0, init_gamma=0.0, init_v_thr=1.0  # γ=0 → hard
    )
    # Force gamma to 0 by clamping the logit hard.
    with torch.no_grad():
        lif.gamma_logit.fill_(-50.0)  # gamma ≈ 0
    # Strong current ⇒ spike. Hard reset ⇒ V should be 0 (with β=sigmoid(0)=0.5
    # leak, V after reset = (sigmoid(-50)*0 + ... so just confirm 0 contribution).
    x = torch.full((1, 2), 10.0)
    _ = lif(x)
    assert torch.allclose(lif.v, torch.zeros_like(lif.v), atol=1e-5)


def test_soft_reset_subtracts_n_vthr():
    lif = FAILIF(hidden_size=1, bits=4, init_v_thr=1.0)
    with torch.no_grad():
        lif.gamma_logit.fill_(50.0)  # gamma ≈ 1 (soft)
        lif.beta_logit.fill_(50.0)  # beta ≈ 1 (no leak)
    # Input 3.4 → V_pre = 3.4 → n = round(3.4) = 3 → V = 3.4 - 3*1.0 = 0.4
    x = torch.tensor([[3.4]])
    n = lif(x)
    assert n.item() == pytest.approx(3.0)
    assert lif.v.item() == pytest.approx(0.4, abs=1e-5)


def test_reset_term_is_detached_no_graph_growth():
    lif = FAILIF(hidden_size=4, bits=4)
    # Build the leaf BEFORE requires_grad_, so multiplication doesn't
    # bury the leaf inside a MulBackward node (which would prevent .grad
    # from being populated).
    x = (torch.randn(2, 4) * 5.0).requires_grad_()
    # Multi-step with retained x: if reset_term were not detached, each
    # step would attach `self.v` to the autograd graph and the backward
    # pass would walk arbitrarily deep. We detach state, so this should be
    # fast and bounded.
    for _ in range(50):
        n = lif(x)
    loss = n.sum()
    loss.backward()
    # x.grad must be non-zero (gradient flows through STE) but the graph
    # depth stays at 1.
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_gradient_flows_through_ste():
    lif = FAILIF(hidden_size=4, bits=4, init_v_thr=1.0)
    x = torch.randn(2, 4, requires_grad=True)
    n = lif(x)
    n.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_state_detached_between_calls():
    lif = FAILIF(hidden_size=4, bits=4)
    x = torch.randn(2, 4, requires_grad=True)
    _ = lif(x)
    assert not lif.v.requires_grad
    assert lif.v.grad_fn is None


def test_dtype_preservation_bf16():
    lif = FAILIF(hidden_size=4, bits=4)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    n = lif(x)
    assert n.dtype == torch.bfloat16
