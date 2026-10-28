"""Autoregressive generation for the spiking-ternary RWKV.

Token-by-token greedy / sampled generation. The recurrent state carries
across tokens — we just call ``model.forward`` on each new token and pick
the next via top-k + temperature.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = 50,
    eos_token_id: int | None = None,
    reset_state: bool = True,
) -> torch.Tensor:
    """Generate ``max_new_tokens`` tokens after the prompt.

    Parameters
    ----------
    input_ids:
        ``(B, T)`` prompt token IDs.
    max_new_tokens:
        Number of tokens to append.
    temperature:
        ``logits / temperature`` before softmax. <=0 disables sampling
        (becomes greedy).
    top_k:
        Restrict the candidate set to the top-k tokens. None disables.
    eos_token_id:
        If provided and emitted, stop early.
    reset_state:
        If True, reset model recurrent state before processing the prompt.

    Returns the full sequence ``(B, T + n_emitted)``.
    """
    model.eval()
    if reset_state and hasattr(model, "reset_state"):
        model.reset_state()

    # Process the prompt to populate state.
    logits = model(input_ids)  # (B, T, V)
    out = input_ids.clone()
    next_logits = logits[:, -1]
    for _ in range(max_new_tokens):
        next_token = _sample(next_logits, temperature=temperature, top_k=top_k)
        out = torch.cat([out, next_token.unsqueeze(1)], dim=1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
        logits = model(next_token.unsqueeze(1))  # (B, 1, V)
        next_logits = logits[:, -1]
    return out


def _sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """Sample one token per row from ``logits`` of shape (B, V)."""
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        top_vals, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = top_vals[..., -1:]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
