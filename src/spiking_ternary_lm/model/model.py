"""Top-level SpikingTernaryRWKV model.

Layout follows the official RWKV-4 reference:

  tokens
    └─ Embedding (FP) ─────────────┐
                                   ▼
                              ln_in (FP, applied once)
                                   │
                              RWKVBlock × N    (ln1 → TimeMix → +residual,
                                               ln2 → ChannelMix → +residual)
                                   │
                              ln_out (FP)
                                   │
                              lm_head (FP linear, no bias)
                                   ▼
                              logits

Per DECISIONS.md #2, the embedding, output norm, and LM head stay FP
(not ternary). Only the BitLinear projections inside TimeMix/ChannelMix
quantize their weights. Activations between BitLinear and the next layer
go through FAI-LIF (integer spike counts).
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from spiking_ternary_lm.layers.block import RWKVBlock
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig


class SpikingTernaryRWKV(nn.Module):
    def __init__(self, config: SpikingTernaryRWKVConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.ln_in = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                RWKVBlock(
                    hidden_size=config.hidden_size,
                    ffn_dim=config.ffn_dim,
                    layer_idx=i,
                    n_layers=config.n_layers,
                    bits=config.bits,
                    group_size=config.group_size,
                    weight_dtype=config.weight_dtype,
                    observer_momentum=getattr(config, "observer_momentum", 0.01),
                    signed_kv=config.signed_kv,
                    spiking_r=config.spiking_r,
                    init_v_thr=getattr(config, "init_v_thr", 0.5),
                )
                for i in range(config.n_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        # Init embedding with std typical for RWKV (small, then layernorm normalises).
        nn.init.normal_(self.embed.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.5 / (config.hidden_size**0.5))

    # ------------------------------------------------------------------

    def reset_state(self, batch_shape: tuple[int, ...] | None = None) -> None:
        for b in self.blocks:
            b.reset_state(batch_shape)

    def detach_state(self) -> None:
        for b in self.blocks:
            b.detach_state()

    def num_parameters(self, only_trainable: bool = True) -> int:
        params: Iterable[nn.Parameter] = (
            p for p in self.parameters() if (not only_trainable) or p.requires_grad
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids:
            ``(B, T)`` long tensor of token IDs.

        Returns
        -------
        logits:
            ``(B, T, vocab_size)`` FP tensor.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, T), got {input_ids.shape}")
        x = self.embed(input_ids)                   # (B, T, H), FP
        x = self.ln_in(x.float()).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x.float()).to(x.dtype)
        logits = self.lm_head(x.float())            # FP head
        return logits
