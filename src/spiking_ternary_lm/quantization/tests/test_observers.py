"""Tests for the MovingAverageMinMaxObserver."""
from __future__ import annotations

import torch

from spiking_ternary_lm.quantization.observers import MovingAverageMinMaxObserver
from spiking_ternary_lm.quantization.ste import ste_quantize_integer


def test_observer_converges_to_true_range():
    """Running stats should converge to the data's true range."""
    obs = MovingAverageMinMaxObserver(num_bits=4, signed=False, momentum=0.1)
    obs.train()
    true_lo, true_hi = 0.0, 5.0
    torch.manual_seed(0)
    for _ in range(500):
        x = torch.empty(100).uniform_(true_lo, true_hi)
        obs(x)
    assert abs(obs.running_min.item() - true_lo) < 0.5
    assert abs(obs.running_max.item() - true_hi) < 0.5


def test_observer_first_call_initialises_directly():
    """The very first batch should set running_min/max exactly (no EMA bias)."""
    obs = MovingAverageMinMaxObserver(num_bits=4, signed=False, momentum=0.01)
    obs.train()
    x = torch.tensor([-3.0, 7.0])
    obs(x)
    assert obs.running_min.item() == -3.0
    assert obs.running_max.item() == 7.0
    assert int(obs.num_updates) == 1


def test_observer_freezes_in_eval():
    """Eval-mode forward must not update running stats."""
    obs = MovingAverageMinMaxObserver(num_bits=4, signed=False, momentum=0.5)
    obs.train()
    for _ in range(100):
        obs(torch.empty(100).uniform_(0, 1))
    saved_min = obs.running_min.clone()
    saved_max = obs.running_max.clone()
    obs.eval()
    for _ in range(100):
        obs(torch.empty(100).uniform_(10, 20))
    assert torch.allclose(obs.running_min, saved_min)
    assert torch.allclose(obs.running_max, saved_max)


def test_observer_signed_qrange():
    obs = MovingAverageMinMaxObserver(num_bits=4, signed=True)
    assert obs.qmin == -8
    assert obs.qmax == 7


def test_observer_unsigned_qrange():
    obs = MovingAverageMinMaxObserver(num_bits=4, signed=False)
    assert obs.qmin == 0
    assert obs.qmax == 15


def test_quantization_round_trip_bounded_error():
    """Quantize then dequantize should be within one quantization step."""
    obs = MovingAverageMinMaxObserver(num_bits=4, signed=False, momentum=0.5)
    obs.train()
    torch.manual_seed(1)
    for _ in range(200):
        obs(torch.empty(1000).uniform_(0, 5))
    scale, zp = obs.get_scale_and_zero_point()
    x_test = torch.empty(1000).uniform_(0, 5)
    x_int = ste_quantize_integer(x_test, scale, zp, obs.qmin, obs.qmax)
    x_recon = (x_int - zp) * scale
    max_err = (5.0 - 0.0) / (obs.qmax - obs.qmin)
    assert (x_test - x_recon).abs().max().item() <= max_err * 1.6


def test_state_dict_persists_running_stats():
    obs = MovingAverageMinMaxObserver(num_bits=4)
    obs.train()
    for _ in range(10):
        obs(torch.randn(20))
    sd = obs.state_dict()
    # Buffers are part of state_dict and survive reload.
    obs2 = MovingAverageMinMaxObserver(num_bits=4)
    obs2.load_state_dict(sd)
    assert torch.equal(obs2.running_min, obs.running_min)
    assert torch.equal(obs2.running_max, obs.running_max)
    assert int(obs2.num_updates) == int(obs.num_updates)


def test_zero_range_handled_by_eps():
    """A constant-input observer should not divide by zero."""
    obs = MovingAverageMinMaxObserver(num_bits=4)
    obs.train()
    obs(torch.zeros(100))
    scale, _ = obs.get_scale_and_zero_point()
    assert torch.isfinite(scale).all()
    assert scale.item() > 0
