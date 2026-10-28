"""Perplexity evaluation on held-out token streams.

Streams a packed corpus through the model and computes
``exp(mean(token-level CE loss))`` — the standard LM perplexity. Lower
is better.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from spiking_ternary_lm.training.data import PackedTokenDataset, TokenizedCorpus


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    corpus: TokenizedCorpus,
    seq_len: int = 256,
    batch_size: int = 4,
    device: torch.device | str = "cuda",
    max_tokens: int | None = None,
    vocab_size: int | None = None,
    detach_between_windows: bool = True,
) -> dict[str, float]:
    """Compute token-level CE and perplexity over ``corpus``.

    Returns ``{"loss": float, "perplexity": float, "n_tokens": int}``.

    If ``detach_between_windows`` is True (default), the model's
    recurrent state is detached between windows so the eval doesn't blow
    autograd memory. The model is put in ``eval()`` mode for the call
    and restored afterwards.
    """
    if vocab_size is None:
        vocab_size = int(model.config.vocab_size)
    device = torch.device(device)
    was_training = model.training
    model.eval()
    if hasattr(model, "reset_state"):
        model.reset_state()

    ds = PackedTokenDataset(corpus, seq_len=seq_len, shuffle_chunks=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        if detach_between_windows and hasattr(model, "detach_state"):
            model.detach_state()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum"
        )
        total_loss += float(loss)
        total_tokens += y.numel()
        if max_tokens is not None and total_tokens >= max_tokens:
            break

    if was_training:
        model.train()

    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss), "n_tokens": total_tokens}


# DECISIONS #15 eval protocol, shared by every rung of the Stage A ladder.
PROTOCOL_SEQ_LEN = 128
PROTOCOL_MAX_TOKENS = 1_000_000


def protocol_eval_corpus(corpus: TokenizedCorpus) -> TokenizedCorpus:
    """The eval tail: last 1 % of the packed corpus (same cut as ``eval_teacher.py``)."""
    n = len(corpus.ids)
    return TokenizedCorpus(
        ids=corpus.ids[int(n * 0.99):], vocab_size=corpus.vocab_size,
        eos_token_id=corpus.eos_token_id,
    )


@torch.no_grad()
def protocol_perplexity(
    model: nn.Module,
    corpus: TokenizedCorpus,
    device: torch.device | str = "cuda",
    batch_size: int = 16,
) -> dict[str, float]:
    """DECISIONS #15 protocol: eval tail, seq 128, first 1 M tokens, stateless.

    Every window starts from a fresh recurrent state, like the HF teacher
    called without ``state``. ``corpus`` is the full packed corpus.
    """
    device = torch.device(device)
    was_training = model.training
    model.eval()
    ds = PackedTokenDataset(
        protocol_eval_corpus(corpus), seq_len=PROTOCOL_SEQ_LEN, shuffle_chunks=False
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    vocab_size = int(model.config.vocab_size)
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        model.reset_state()
        logits = model(x).float()
        total_loss += float(F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum"
        ))
        total_tokens += y.numel()
        if total_tokens >= PROTOCOL_MAX_TOKENS:
            break
    model.reset_state()
    if was_training:
        model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss), "n_tokens": total_tokens}
