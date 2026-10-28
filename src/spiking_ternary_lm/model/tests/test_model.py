"""Tests for the top-level SpikingTernaryRWKV model."""
from __future__ import annotations

import torch

from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV


def test_forward_shape():
    cfg = SpikingTernaryRWKVConfig.tiny()
    model = SpikingTernaryRWKV(cfg)
    model.reset_state()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backward_pass_runs():
    cfg = SpikingTernaryRWKVConfig.tiny()
    model = SpikingTernaryRWKV(cfg)
    model.reset_state()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 4), dtype=torch.long)
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), input_ids.reshape(-1)
    )
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert has_grad > 0


def test_num_parameters_positive():
    cfg = SpikingTernaryRWKVConfig.tiny()
    model = SpikingTernaryRWKV(cfg)
    assert model.num_parameters() > 0


def test_tie_word_embeddings():
    cfg = SpikingTernaryRWKVConfig.tiny()
    cfg.tie_word_embeddings = True
    model = SpikingTernaryRWKV(cfg)
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
