"""Tests for the combined RWKV block (TimeMix + ChannelMix + residuals)."""
from __future__ import annotations

import torch

from spiking_ternary_lm.layers.block import RWKVBlock


def test_block_shape():
    block = RWKVBlock(hidden_size=64, ffn_dim=256, layer_idx=0, n_layers=2, group_size=16)
    block.reset_state()
    y = block(torch.randn(2, 5, 64))
    assert y.shape == (2, 5, 64)


def test_block_residual_changes_output():
    block = RWKVBlock(hidden_size=64, ffn_dim=256, layer_idx=0, n_layers=2, group_size=16)
    block.reset_state()
    x = torch.randn(2, 5, 64)
    y = block(x)
    # With residual, output should not be equal to TimeMix+ChannelMix alone
    # (and definitely not equal to x — that would mean both blocks output 0).
    assert not torch.allclose(y, x)


def test_block_gradients_flow():
    block = RWKVBlock(hidden_size=64, ffn_dim=256, layer_idx=0, n_layers=2, group_size=16)
    block.reset_state()
    x = torch.randn(2, 3, 64, requires_grad=True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_block_reset_clears_state():
    block = RWKVBlock(hidden_size=64, ffn_dim=256, layer_idx=0, n_layers=2, group_size=16)
    block.reset_state(batch_shape=(2,))
    block(torch.randn(2, 64))
    assert block.time_mix.aa.abs().sum() > 0 or block.time_mix.pp.abs().sum() > 0
    block.reset_state(batch_shape=(2,))
    assert (block.time_mix.aa == 0).all()
    assert (block.channel_mix.x_prev == 0).all()
