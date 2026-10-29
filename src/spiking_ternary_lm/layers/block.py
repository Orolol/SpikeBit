"""One RWKV-4 layer: LayerNorm + TimeMix + LayerNorm + ChannelMix, with residuals.

Pre-norm residual structure following the official RWKV-4 reference.
The two LayerNorms in front of TimeMix and ChannelMix are kept in
**FP32** (per DECISIONS.md #2), matching every other reference-frame
parameter and recurrent state.

The very first block additionally applies an "input LayerNorm" to the
embedding output (handled in the model, not here).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from spiking_ternary_lm.layers.spiking_ffn import SpikingRWKVChannelMix
from spiking_ternary_lm.layers.spiking_rwkv import SpikingRWKVTimeMix


class RWKVBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int | None = None,
        layer_idx: int = 0,
        n_layers: int = 1,
        bits: int = 4,
        group_size: int = 128,
        weight_dtype: torch.dtype = torch.float32,
        observer_momentum: float = 0.01,
        signed_kv: bool = True,
        spiking_r: bool = True,
        init_v_thr: float = 0.5,  # ignored by observer-based path
        ffn_relu_sq: bool = False,
        act_grid: str = "legacy",
        bits_ffn_hidden: int = 0,
        act_quant_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_idx = int(layer_idx)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.time_mix = SpikingRWKVTimeMix(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            n_layers=n_layers,
            bits=bits,
            group_size=group_size,
            weight_dtype=weight_dtype,
            observer_momentum=observer_momentum,
            signed_kv=signed_kv,
            spiking_r=spiking_r,
            init_v_thr=init_v_thr,
            grid=act_grid,
            quantize_inputs=act_quant_inputs,
        )
        self.channel_mix = SpikingRWKVChannelMix(
            hidden_size=hidden_size,
            ffn_dim=ffn_dim,
            layer_idx=layer_idx,
            n_layers=n_layers,
            bits=bits,
            group_size=group_size,
            weight_dtype=weight_dtype,
            observer_momentum=observer_momentum,
            init_v_thr=init_v_thr,
            relu_sq=ffn_relu_sq,
            grid=act_grid,
            hidden_bits=bits_ffn_hidden,
            quantize_inputs=act_quant_inputs,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm in FP32 for stability, projection sees the input dtype.
        x_norm1 = self.ln1(x.float()).to(x.dtype)
        x = x + self.time_mix(x_norm1)
        x_norm2 = self.ln2(x.float()).to(x.dtype)
        x = x + self.channel_mix(x_norm2)
        return x

    def reset_state(self, batch_shape: tuple[int, ...] | None = None) -> None:
        self.time_mix.reset_state(batch_shape)
        self.channel_mix.reset_state(batch_shape)

    def detach_state(self) -> None:
        self.time_mix.detach_state()
        self.channel_mix.detach_state()
