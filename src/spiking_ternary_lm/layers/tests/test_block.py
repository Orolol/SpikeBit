"""Tests for the combined RWKV block (TimeMix + ChannelMix + residuals)."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.layers.block import RWKVBlock
from spiking_ternary_lm.neurons.fai_lif import (
    FAILIF, enable_lsq, set_output_mode, set_stateless, set_ste_clip,
)


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


# Each ternary BitLinear and the FAI-LIF whose output it takes, with the
# token-shift mixes quantized (DECISIONS #22).
_FEEDS = {
    "time_mix.W_r": "time_mix.lif_xr", "time_mix.W_k": "time_mix.lif_xk",
    "time_mix.W_v": "time_mix.lif_xv", "time_mix.W_o": "time_mix.lif_out",
    "channel_mix.W_k": "channel_mix.lif_xk", "channel_mix.W_v": "channel_mix.lif_k",
    "channel_mix.W_r": "channel_mix.lif_xr",
}


@pytest.mark.parametrize("sequence", [True, False])
def test_every_bitlinear_input_is_on_its_lif_grid(sequence):
    """With ``act_quant_inputs`` every BitLinear input is an integer count
    times its LIF's LSQ step, through the fused-sequence and per-step paths."""
    block = RWKVBlock(hidden_size=64, ffn_dim=256, layer_idx=0, n_layers=2, group_size=16,
                      spiking_r=False, ffn_relu_sq=True, act_grid="faithful", act_quant_inputs=True)
    assert sum(isinstance(m, FAILIF) for m in block.modules()) == 10
    enable_lsq(block)
    set_output_mode(block, "dequant")
    set_ste_clip(block)
    set_stateless(block)
    block.train()
    block.reset_state()
    block(torch.randn(2, 8, 64))  # MinMax warm-up call, then the LSQ grids are set
    block.eval()
    block.reset_state()
    steps = {lin: block.get_submodule(lif).lsq_scale_and_zero_point()[0] for lin, lif in _FEEDS.items()}
    counts: dict[str, list[torch.Tensor]] = {lin: [] for lin in _FEEDS}
    for lin in _FEEDS:
        block.get_submodule(lin).register_forward_pre_hook(
            lambda _m, inp, lin=lin: counts[lin].append(inp[0] / steps[lin])
        )
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        if sequence:
            block(x)
        else:
            for t in range(x.shape[1]):
                block(x[:, t])
    for lin, qs in counts.items():
        q = torch.cat([c.reshape(-1) for c in qs])
        assert q.numel() and torch.allclose(q, q.round(), atol=1e-4), lin


def test_quantized_inputs_need_the_faithful_grid():
    with pytest.raises(ValueError, match="faithful"):
        RWKVBlock(hidden_size=64, ffn_dim=256, group_size=16, act_quant_inputs=True)
