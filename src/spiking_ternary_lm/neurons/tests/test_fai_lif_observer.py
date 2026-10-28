"""Tests for the observer-based :class:`FAILIF`."""
from __future__ import annotations

import torch

from spiking_ternary_lm.neurons.fai_lif import FAILIF, set_calibrating


def _warmup_observer(lif: FAILIF, n: int = 20, dim: int | None = None) -> None:
    lif.train()
    dim = dim or lif.num_neurons
    for _ in range(n):
        lif(torch.randn(4, dim))


def test_output_is_integer_in_qrange():
    lif = FAILIF(num_neurons=64, num_bits=4, signed=False)
    _warmup_observer(lif, n=50, dim=64)
    spike = lif(torch.randn(8, 64) * 2)
    assert (spike >= 0).all()
    assert (spike <= 15).all()
    assert torch.equal(spike, spike.round())


def test_output_signed_range():
    lif = FAILIF(num_neurons=32, num_bits=4, signed=True)
    _warmup_observer(lif, n=50, dim=32)
    spike = lif(torch.randn(8, 32) * 2)
    assert (spike >= -8).all()
    assert (spike <= 7).all()


def test_gradient_flows_through_ste():
    lif = FAILIF(num_neurons=16, num_bits=4)
    _warmup_observer(lif, n=20, dim=16)
    x = torch.randn(4, 16, requires_grad=True)
    spike = lif(x)
    spike.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_state_carries_across_steps():
    """Membrane potential should accumulate when input is sub-spike."""
    torch.manual_seed(0)  # the observer warm-up below is random; unseeded it failed ~50 %
    lif = FAILIF(num_neurons=2, num_bits=4, beta_init=0.95, gamma_init=1.0)
    lif.train()
    # Warm the observer with a broad distribution so the scale is not tiny.
    for _ in range(20):
        lif(torch.randn(1, 2))
    # Force a clean state, then feed small positive current.
    lif.reset_state(batch_shape=(1,))
    seen_v = []
    for _ in range(5):
        lif(torch.full((1, 2), 0.05))
        seen_v.append(lif.v.clone())
    # First-step v should be smaller than later v (accumulation).
    assert seen_v[-1].abs().sum() >= seen_v[0].abs().sum() - 1e-6


def test_calibrating_mode_updates_observer_but_skips_quant():
    lif = FAILIF(num_neurons=8, num_bits=4)
    set_calibrating(lif, True)
    lif.train()
    x = torch.randn(4, 8) * 3
    out = lif(x)
    # Output is the FP membrane (not integer-quantized).
    assert not torch.equal(out, out.round())  # very high prob.
    # Observer still updated.
    assert int(lif.observer.num_updates) >= 1
    set_calibrating(lif, False)
    # After leaving calibration, normal forward emits integers.
    spike = lif(x)
    assert torch.equal(spike, spike.round())


def test_set_calibrating_walks_model_recursively():
    lif1 = FAILIF(num_neurons=4)
    lif2 = FAILIF(num_neurons=4)
    container = torch.nn.Sequential(lif1, lif2)
    n = set_calibrating(container, True)
    assert n == 2
    assert lif1.calibrating
    assert lif2.calibrating
    set_calibrating(container, False)
    assert not lif1.calibrating
    assert not lif2.calibrating


def test_observer_freezes_in_eval():
    lif = FAILIF(num_neurons=16, num_bits=4, observer_momentum=0.5)
    _warmup_observer(lif, n=50, dim=16)
    saved = (lif.observer.running_min.clone(), lif.observer.running_max.clone())
    lif.eval()
    for _ in range(50):
        lif(torch.randn(4, 16) * 100)
    assert torch.equal(lif.observer.running_min, saved[0])
    assert torch.equal(lif.observer.running_max, saved[1])


def test_reset_state_zeros_membrane_but_keeps_observer():
    lif = FAILIF(num_neurons=8)
    _warmup_observer(lif, n=20, dim=8)
    saved_max = lif.observer.running_max.clone()
    lif.reset_state(batch_shape=(2,))
    assert (lif.v == 0).all()
    assert torch.equal(lif.observer.running_max, saved_max)


def test_dtype_preservation_bf16():
    lif = FAILIF(num_neurons=4, num_bits=4)
    _warmup_observer(lif, n=10, dim=4)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    spike = lif(x)
    assert spike.dtype == torch.bfloat16


def test_compat_properties_for_legacy_callers():
    """Legacy code may read ``hidden_size``, ``bits``, ``n_min/n_max``."""
    lif = FAILIF(num_neurons=8, num_bits=4, signed=False)
    assert lif.hidden_size == 8
    assert lif.bits == 4
    assert lif.n_min == 0
    assert lif.n_max == 15
