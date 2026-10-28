"""Configuration dataclass for SpikingTernaryRWKV."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SpikingTernaryRWKVConfig:
    """Hyperparameters for the spiking-ternary RWKV-4 model.

    Defaults match the NSLLM 169M target sizing (matches the official
    RWKV-4-169M / Pile pre-training shape so Stage A can load teacher
    weights without dim mismatch).
    """

    # Architecture.
    vocab_size: int = 50277  # RWKV-4-Pile tokenizer vocab
    hidden_size: int = 768   # RWKV-4-169M hidden dim
    n_layers: int = 12       # RWKV-4-169M depth
    ffn_dim: int | None = None  # default 4 * hidden_size (RWKV-4 convention)

    # Quantization / activation.
    bits: int = 4
    group_size: int = 128
    weight_dtype: Any = torch.float32  # see DECISIONS.md #4, #12
    # Moving-average MinMax observer (replaces ``init_v_thr`` in the new
    # FAI-LIF — see DECISIONS.md #6, #7, #9, Phase 3.5 spec).
    observer_momentum: float = 0.01
    # Activation quantizer for every FAI-LIF: "minmax" (observer-tracked,
    # NSLLM-style), "lsq" (learned step size, DECISIONS #14) or "none"
    # (FAI-LIF is an identity: FP activations, rungs 0-2 of DECISIONS #15).
    act_quant: str = "minmax"
    # BitLinear weights: "ternary" (default) or "none" (FP shadow weights,
    # rungs 0-1 of DECISIONS #15).
    weight_quant: str = "ternary"
    # Keep the teacher's channel-mix ``relu(.)**2`` in front of the FFN
    # FAI-LIF (DECISIONS #15). False = legacy layout, where FAI-LIF replaces it.
    ffn_relu_sq: bool = False
    # FAI-LIF grids: "legacy" (asymmetric, ``lif_out`` and the FFN ``lif_v``
    # unsigned) or "faithful" (DECISIONS #15: zp = 0, signed where the
    # teacher's activation is signed, unsigned after relu²).
    act_grid: str = "legacy"
    # MinMax FAI-LIF output: "int" (raw spike count, legacy) or "dequant"
    # (``scale * (q - zp)``, identity STE).
    act_output: str = "int"
    # beta = 0 on every FAI-LIF: no membrane carried across tokens.
    lif_stateless: bool = False
    # Bit-width of the FFN hidden FAI-LIF (relu² output); 0 = ``bits``.
    bits_ffn_hidden: int = 0
    # MinMax observer scope: "tensor" (one grid per activation tensor, NSLLM)
    # or "channel" (one grid per neuron = per-neuron threshold, DECISIONS #14).
    observer_scope: str = "tensor"
    lsq_init_steps: int = 1
    # Kept for back-compat with the legacy FAILIFFixedThreshold path,
    # ignored by the observer-based FAILIF default.
    init_v_thr: float = 0.5

    # Spiking / FAI-LIF toggles.
    # ``spiking_r`` default = False per DEC-8 (the FP-receptance dérogation
    # that made Stage A pol #2 work). To revisit for Stage C.
    signed_kv: bool = True
    spiking_r: bool = False

    # I/O.
    tie_word_embeddings: bool = False  # if True, share lm_head weight with embed

    # Smaller config presets — handy for unit tests and the Phase 2 gate.
    @classmethod
    def tiny(cls) -> "SpikingTernaryRWKVConfig":
        return cls(
            vocab_size=256,
            hidden_size=128,
            n_layers=2,
            ffn_dim=512,
            group_size=32,
            bits=4,
        )

    @classmethod
    def small_169m(cls) -> "SpikingTernaryRWKVConfig":
        return cls()  # defaults already match


__all__ = ["SpikingTernaryRWKVConfig"]
