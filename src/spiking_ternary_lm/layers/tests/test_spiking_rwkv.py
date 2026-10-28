"""Tests for the spiking-ternary RWKV time-mixing block."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.layers.spiking_rwkv import SpikingRWKVTimeMix


@pytest.fixture
def block():
    return SpikingRWKVTimeMix(
        hidden_size=64, layer_idx=0, n_layers=2, bits=4, group_size=16, init_v_thr=0.5
    )


def test_step_shape(block):
    x = torch.randn(2, 64)
    block.reset_state()
    y = block(x)
    assert y.shape == (2, 64)


def test_sequence_shape(block):
    x = torch.randn(2, 5, 64)
    block.reset_state()
    y = block(x)
    assert y.shape == (2, 5, 64)


def test_state_advances_across_steps(block):
    block.reset_state(batch_shape=(2,))
    pp0 = block.pp.clone()
    block(torch.randn(2, 64))
    assert not torch.equal(block.pp, pp0)


def test_step_equals_sequence_loop(block):
    """Calling ``forward`` on a sequence should equal stacking per-step calls.

    With the observer-based FAI-LIF, every forward call mutates the
    activation observer's running stats. Equivalence only holds with
    frozen statistics, so we warm up the observers first and then run
    the check in ``eval()`` mode.
    """
    # Warm up so the observers have non-default stats.
    block.train()
    block.reset_state()
    block(torch.randn(2, 8, 64))
    block.eval()

    x = torch.randn(2, 4, 64)
    block.reset_state()
    y_seq = block(x)

    block.reset_state()
    outs = []
    for t in range(4):
        outs.append(block(x[:, t]))
    y_step = torch.stack(outs, dim=1)
    assert torch.allclose(y_seq, y_step, atol=1e-5)


def test_gradient_flows_back_through_sequence(block):
    x = torch.randn(2, 3, 64, requires_grad=True)
    block.reset_state()
    y = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_wkv_state_is_fp32(block):
    block.reset_state(batch_shape=(2,))
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    block(x)
    assert block.aa.dtype == torch.float32
    assert block.bb.dtype == torch.float32
    assert block.pp.dtype == torch.float32


def test_no_nan_with_large_inputs(block):
    block.reset_state()
    x = torch.randn(2, 10, 64) * 10.0
    y = block(x)
    assert torch.isfinite(y).all()


def test_detach_state_breaks_graph(block):
    block.reset_state(batch_shape=(2,))
    x1 = torch.randn(2, 64, requires_grad=True)
    block(x1)
    assert block.aa.requires_grad
    block.detach_state()
    assert not block.aa.requires_grad
