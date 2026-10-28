"""Load a pre-trained RWKV-4 checkpoint from HuggingFace.

We use the official HuggingFace `RwkvForCausalLM` class — it parses the
weight names the same way `transformers` would parse them anywhere else,
and gives us a flat `state_dict` that's easy to map onto our
SpikingTernaryRWKV.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TeacherInfo:
    """Inventory of a loaded RWKV teacher."""

    model_id: str
    state_dict: dict[str, Tensor]
    n_layers: int
    hidden_size: int
    ffn_dim: int
    vocab_size: int


def load_rwkv4_teacher(model_id: str = "RWKV/rwkv-4-169m-pile") -> TeacherInfo:
    """Download (or load from cache) the RWKV teacher and return its weights.

    Returns the raw `state_dict` *plus* the shape inventory so the caller
    can build a matching SpikingTernaryRWKV before assigning weights.

    No CUDA, no GPU memory burned: the model is loaded on CPU in FP32 and
    we extract the tensors. The caller decides where to put them.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.float32
    )
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Infer shape from a representative block.
    hidden_size = sd["rwkv.embeddings.weight"].shape[1]
    vocab_size = sd["rwkv.embeddings.weight"].shape[0]
    ffn_dim = sd["rwkv.blocks.0.feed_forward.key.weight"].shape[0]
    n_layers = 1 + max(
        int(k.split(".")[2]) for k in sd if k.startswith("rwkv.blocks.")
    )

    # Free the HF Python wrapper, keep only the tensors.
    del model

    return TeacherInfo(
        model_id=model_id,
        state_dict=sd,
        n_layers=n_layers,
        hidden_size=hidden_size,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
    )
