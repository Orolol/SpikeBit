"""Sanity tests for surrogate gradient Functions."""
from __future__ import annotations

import math

import pytest
import torch

from spiking_ternary_lm.neurons.surrogate import (
    ArcTanSurrogate,
    FastSigmoidSurrogate,
    RectangularSurrogate,
)


def _forward(fn, x, alpha):
    return fn.apply(x, alpha)


@pytest.mark.parametrize(
    "fn,alpha", [(FastSigmoidSurrogate, 4.0), (ArcTanSurrogate, 2.0), (RectangularSurrogate, 1.0)]
)
def test_forward_is_heaviside(fn, alpha):
    x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
    y = _forward(fn, x, alpha)
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    assert torch.equal(y, expected)


@pytest.mark.parametrize("fn,alpha", [(FastSigmoidSurrogate, 4.0), (ArcTanSurrogate, 2.0)])
def test_gradient_peaks_at_threshold(fn, alpha):
    x = torch.linspace(-2.0, 2.0, steps=401, requires_grad=True)
    y = _forward(fn, x, alpha)
    y.sum().backward()
    grad = x.grad
    assert torch.argmax(grad).item() == 200  # x == 0
    assert grad[0] < grad[200]
    assert grad[-1] < grad[200]


def test_rectangular_gradient_is_boxcar():
    x = torch.tensor([-0.6, -0.49, 0.0, 0.49, 0.6], requires_grad=True)
    y = RectangularSurrogate.apply(x, 1.0)
    y.sum().backward()
    g = x.grad
    assert g[0].item() == 0.0
    assert g[-1].item() == 0.0
    assert g[1].item() == 1.0
    assert g[2].item() == 1.0
    assert g[3].item() == 1.0


def test_fast_sigmoid_gradient_formula():
    # surrogate'(x) = 1 / (1 + alpha * |x|) ** 2
    x = torch.tensor([0.0, 0.25, -0.5], requires_grad=True)
    alpha = 4.0
    y = FastSigmoidSurrogate.apply(x, alpha)
    y.sum().backward()
    expected = 1.0 / (1.0 + alpha * x.detach().abs()) ** 2
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_arctan_gradient_formula():
    x = torch.tensor([0.0, 0.25, -0.5], requires_grad=True)
    alpha = 2.0
    y = ArcTanSurrogate.apply(x, alpha)
    y.sum().backward()
    pa = math.pi * alpha
    expected = (alpha / math.pi) / (1.0 + (pa * x.detach()) ** 2)
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_alpha_is_per_call():
    # Same function used twice with different alphas: gradient must differ.
    x1 = torch.tensor([0.3], requires_grad=True)
    x2 = torch.tensor([0.3], requires_grad=True)
    FastSigmoidSurrogate.apply(x1, 2.0).sum().backward()
    FastSigmoidSurrogate.apply(x2, 8.0).sum().backward()
    assert x1.grad.item() > x2.grad.item()
