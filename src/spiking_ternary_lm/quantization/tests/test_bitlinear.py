"""Tests for the BitLinear layer."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from spiking_ternary_lm.quantization.bitlinear import BitLinear


def test_construct_and_shape():
    bl = BitLinear(128, 64, group_size=32)
    x = torch.randn(8, 128, dtype=torch.float32)
    y = bl(x)
    assert y.shape == (8, 64)


def test_group_divisibility_check():
    with pytest.raises(ValueError):
        BitLinear(in_features=130, out_features=64, group_size=32)


def test_weight_distribution_is_ternary_per_group():
    bl = BitLinear(128, 64, group_size=32)
    q = bl.quantized_weight()
    # For each row+group, only up to three unique levels (one of them
    # symmetric about zero) — i.e. {-s, 0, +s}. Take abs() BEFORE unique
    # so that -s and +s collapse to the single magnitude.
    grouped = q.view(64, 4, 32)
    for r in range(64):
        for g in range(4):
            magnitudes = grouped[r, g].abs().unique()
            non_zero = magnitudes[magnitudes > 0]
            assert non_zero.numel() <= 1


def test_gradient_flows_to_shadow_weight():
    bl = BitLinear(64, 16, group_size=16, weight_dtype=torch.float32)
    x = torch.randn(4, 64, requires_grad=False)
    y = bl(x)
    y.sum().backward()
    assert bl.shadow_weight.grad is not None
    assert bl.shadow_weight.grad.abs().sum().item() > 0


def test_all_zero_group_handled_gracefully():
    bl = BitLinear(64, 16, group_size=16, weight_dtype=torch.float32)
    # Zero out a whole group of the shadow weight.
    with torch.no_grad():
        bl.shadow_weight[0, :16].zero_()
    x = torch.randn(2, 64)
    y = bl(x)  # should not produce nan
    assert torch.isfinite(y).all()


def test_quantization_levels_count_after_pack():
    bl = BitLinear(64, 16, group_size=16, weight_dtype=torch.float32)
    ternary, scales = bl.pack_weights()
    assert ternary.shape == (16, 64)
    assert ternary.dtype == torch.int8
    assert set(ternary.unique().tolist()).issubset({-1, 0, 1})
    assert scales.shape == (16, 4)


def test_dist_sums_to_one():
    bl = BitLinear(128, 32, group_size=32, weight_dtype=torch.float32)
    d = bl.weight_distribution()
    s = d["neg"] + d["zero"] + d["pos"]
    assert math.isclose(s, 1.0, abs_tol=1e-6)


def test_forward_matches_explicit_quantization():
    """Forward should equal F.linear(x, quantized_weight)."""
    bl = BitLinear(64, 16, group_size=16, weight_dtype=torch.float32)
    x = torch.randn(4, 64)
    y = bl(x)
    w_q = bl.quantized_weight()
    y_ref = F.linear(x, w_q)
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_bf16_weight_storage():
    bl = BitLinear(64, 16, group_size=16, weight_dtype=torch.bfloat16)
    assert bl.shadow_weight.dtype == torch.bfloat16
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    y = bl(x)
    assert y.dtype == torch.bfloat16


def test_bias_optional():
    bl = BitLinear(32, 8, group_size=16, bias=True, weight_dtype=torch.float32)
    assert bl.bias is not None
    x = torch.randn(2, 32)
    y = bl(x)
    assert y.shape == (2, 8)


@pytest.mark.parametrize("qmin,qmax,zp", [(0, 15, 6.0), (-8, 7, -2.0), (0, 255, 0.0)])
def test_forward_integer_matches_dequantized(qmin, qmax, zp):
    """Folded integer path == BitLinear on the dequantized activation (DECISIONS #15)."""
    torch.manual_seed(0)
    bl = BitLinear(128, 48, group_size=32)
    q = torch.randint(qmin, qmax + 1, (5, 7, 128)).to(torch.float32)
    scale, zero_point = torch.tensor(0.0731), torch.tensor(zp)
    folded = bl.forward_integer(q, scale, zero_point)
    dequant = bl(scale * (q - zero_point))
    assert torch.allclose(folded, dequant, rtol=1e-5, atol=1e-4)


def test_quant_mix_blends_fp_and_ternary_with_identity_gradient():
    """Progressive ternarization (DECISIONS #15 plan B): 0 = FP, 1 = ternary."""
    torch.manual_seed(0)
    bl = BitLinear(64, 16, group_size=16)
    x = torch.randn(4, 64)
    fp = F.linear(x, bl.shadow_weight)
    ternary = bl(x)
    bl.quant_mix = 0.0
    assert torch.allclose(bl(x), fp, atol=1e-6)
    bl.quant_mix = 0.25
    assert torch.allclose(bl(x), fp + 0.25 * (ternary - fp), atol=1e-5)
    bl(x).sum().backward()
    assert torch.allclose(bl.shadow_weight.grad, x.sum(0).expand(16, 64), atol=1e-5)
