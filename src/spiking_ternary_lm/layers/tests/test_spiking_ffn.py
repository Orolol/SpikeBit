"""Tests for the spiking-ternary RWKV channel-mixing block."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.layers.spiking_ffn import SpikingRWKVChannelMix


@pytest.fixture
def block():
    return SpikingRWKVChannelMix(
        hidden_size=64, ffn_dim=256, layer_idx=0, n_layers=2, bits=4, group_size=16
    )


def test_step_shape(block):
    block.reset_state()
    y = block(torch.randn(2, 64))
    assert y.shape == (2, 64)


def test_sequence_shape(block):
    block.reset_state()
    y = block(torch.randn(2, 5, 64))
    assert y.shape == (2, 5, 64)


def test_step_equals_sequence_loop(block):
    """Observer-based FAI-LIF makes train-mode forward non-idempotent; we
    warm up then switch to eval so observer stats are frozen."""
    block.train()
    block.reset_state()
    block(torch.randn(2, 8, 64))
    block.eval()

    x = torch.randn(2, 4, 64)
    block.reset_state()
    y_seq = block(x)
    block.reset_state()
    outs = [block(x[:, t]) for t in range(4)]
    y_step = torch.stack(outs, dim=1)
    assert torch.allclose(y_seq, y_step, atol=1e-5)


def test_gradient_flows(block):
    x = torch.randn(2, 3, 64, requires_grad=True)
    block.reset_state()
    y = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_default_ffn_dim():
    block = SpikingRWKVChannelMix(hidden_size=128, group_size=128)
    assert block.ffn_dim == 4 * 128


def test_group_size_validation():
    with pytest.raises(ValueError):
        SpikingRWKVChannelMix(hidden_size=130, group_size=128)
    with pytest.raises(ValueError):
        SpikingRWKVChannelMix(hidden_size=128, ffn_dim=500, group_size=128)


def test_no_nan_with_large_inputs(block):
    block.reset_state()
    y = block(torch.randn(2, 10, 64) * 10.0)
    assert torch.isfinite(y).all()
