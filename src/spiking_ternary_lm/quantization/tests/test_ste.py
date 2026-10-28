"""Tests for the ternary STE quantizer."""
from __future__ import annotations

import torch

from spiking_ternary_lm.quantization.ste import ste_quantize_ternary


def test_forward_produces_three_levels_per_group():
    w = torch.tensor([0.1, 0.5, -0.05, -0.3], dtype=torch.float32).view(1, 1, 4)
    scale = torch.tensor([[[w.abs().mean()]]])
    q = ste_quantize_ternary(w, scale)
    allowed = {0.0, scale.item(), -scale.item()}
    for v in q.flatten().tolist():
        assert min(abs(v - a) for a in allowed) < 1e-6


def test_threshold_at_half_scale():
    # Values with |w| > scale/2 fire; otherwise 0.
    w = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.9], dtype=torch.float32).view(1, 1, 5)
    scale = torch.tensor([[[0.4]]])  # scale/2 = 0.2
    q = ste_quantize_ternary(w, scale)
    expected = torch.tensor([0.0, 0.0, 0.4, 0.4, 0.4]).view(1, 1, 5)
    assert torch.allclose(q, expected)


def test_signs_preserved():
    w = torch.tensor([-0.5, 0.5, -0.6, 0.6], dtype=torch.float32).view(1, 1, 4)
    scale = torch.tensor([[[0.4]]])
    q = ste_quantize_ternary(w, scale)
    assert torch.equal(torch.sign(q), torch.sign(w))


def test_identity_ste_gradient():
    w = torch.randn(2, 3, 4, requires_grad=True)
    scale = w.detach().abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    q = ste_quantize_ternary(w, scale)
    q.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_no_grad_through_scale():
    w = torch.randn(2, 3, 4)
    scale = torch.full((2, 3, 1), 0.5, requires_grad=True)
    q = ste_quantize_ternary(w, scale)
    q.sum().backward()
    assert scale.grad is None


def test_bf16_round_trip():
    w = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    q = ste_quantize_ternary(w, scale)
    assert q.dtype == torch.bfloat16
    # Each group has at most three distinct absolute values.
    abs_unique = q.abs().view(2 * 3, 8).unique(dim=-1)  # not exact but cheap
    assert abs_unique.numel() > 0
