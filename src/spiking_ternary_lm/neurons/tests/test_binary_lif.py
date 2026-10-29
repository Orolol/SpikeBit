"""Tests for the integer→binary spike train decomposition."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.neurons.binary_lif import (
    BinaryLIF,
    decompose_integer_to_binary,
    decompose_signed_to_spikes,
)
from spiking_ternary_lm.neurons.fai_lif import FAILIFFixedThreshold as FAILIF
from spiking_ternary_lm.quantization.bitlinear import BitLinear


@pytest.mark.parametrize("scheme", ["even", "sequential"])
def test_sum_equals_integer_value(scheme):
    n = torch.tensor([0, 1, 5, 7, 12, 15], dtype=torch.int64)
    out = decompose_integer_to_binary(n, T=15, scheme=scheme, bits=4)
    assert out.shape == (15, 6)
    assert torch.equal(out.sum(dim=0), n)


@pytest.mark.parametrize("scheme", ["even", "sequential"])
def test_output_is_binary(scheme):
    n = torch.tensor([0, 3, 7, 15], dtype=torch.int64)
    out = decompose_integer_to_binary(n, T=15, scheme=scheme, bits=4)
    unique = out.unique().tolist()
    assert set(unique).issubset({0, 1})


def test_even_distribution_spreads_spikes():
    # For N=4, T=8: spikes at t=1,3,5,7 (or close to). Not bunched at front.
    n = torch.tensor([4])
    out = decompose_integer_to_binary(n, T=8, scheme="even", bits=4)
    # Indices of spikes.
    idx = (out[:, 0] == 1).nonzero(as_tuple=True)[0].tolist()
    assert len(idx) == 4
    # Spikes should not all be in the first 4 positions (that would be sequential).
    assert idx != [0, 1, 2, 3]


def test_sequential_packs_at_front():
    n = torch.tensor([3])
    out = decompose_integer_to_binary(n, T=8, scheme="sequential", bits=4)
    assert out[:, 0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]


def test_default_T_is_2_to_A_minus_1():
    n = torch.tensor([7])
    out = decompose_integer_to_binary(n, bits=4)
    assert out.shape[0] == 15  # 2**4 - 1


def test_rejects_oversized_values():
    n = torch.tensor([16])  # > 15 for bits=4
    with pytest.raises(ValueError):
        decompose_integer_to_binary(n, bits=4)


def test_rejects_negative_values():
    n = torch.tensor([-1])
    with pytest.raises(ValueError):
        decompose_integer_to_binary(n, bits=4)


def test_preserves_input_dtype_for_float():
    n = torch.tensor([3.0, 5.0], dtype=torch.bfloat16)
    out = decompose_integer_to_binary(n, T=15, bits=4)
    assert out.dtype == torch.bfloat16


def test_module_equivalent_to_function():
    n = torch.tensor([[0, 3], [7, 15]], dtype=torch.int64)
    mod = BinaryLIF(bits=4, T=15, scheme="even")
    fn = decompose_integer_to_binary(n, T=15, scheme="even", bits=4)
    assert torch.equal(mod(n), fn)


def test_round_trip_with_fai_lif():
    """End-to-end sanity: FAI-LIF integer output decomposed should sum back."""
    torch.manual_seed(0)
    lif = FAILIF(hidden_size=8, bits=4, init_v_thr=1.0)
    with torch.no_grad():
        lif.gamma_logit.fill_(50.0)
        lif.beta_logit.fill_(50.0)
    x = torch.rand(4, 8) * 14.0  # currents driving counts in [0, 14]
    n = lif(x).detach()
    n_int = n.round().to(torch.int64)
    binary = decompose_integer_to_binary(n_int, T=15, scheme="even", bits=4)
    assert torch.equal(binary.sum(dim=0), n_int)


@pytest.mark.parametrize("scheme", ["even", "sequential"])
@pytest.mark.parametrize("signed", [True, False])
def test_spike_reading_is_exact_through_a_ternary_bitlinear(scheme, signed):
    """DECISIONS #19 rung 4a: a ternary BitLinear fed the spikes, summed over T,
    gives its integer-path output; ±1 spikes for signed grids, T = 8 (signed
    4-bit) or 15 (unsigned 4-bit)."""
    torch.manual_seed(0)
    if signed:
        q = torch.randint(-8, 8, (4, 5, 64)).float()
        spikes = decompose_signed_to_spikes(q, scheme=scheme, bits=4)
    else:
        q = torch.randint(0, 16, (4, 5, 64)).float()
        spikes = decompose_integer_to_binary(q, scheme=scheme, bits=4)
    assert spikes.shape[0] == (8 if signed else 15)
    assert set(spikes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.equal(spikes.sum(0), q) and torch.equal(spikes.abs().sum(0), q.abs())

    bl = BitLinear(64, 24, group_size=32)
    ternary = bl.pack_weights()[0].to(torch.float64)
    # Integer accumulations: exact (float64 holds these sums exactly).
    per_step = torch.stack([s.to(torch.float64) @ ternary.T for s in spikes]).sum(0)
    assert torch.equal(per_step, q.to(torch.float64) @ ternary.T)
    # The folded BitLinear path, with the activation step in the group scales.
    s, zp = torch.tensor(0.0731), torch.tensor(0.0)
    via_spikes = sum(bl.forward_integer(sp, s, zp) for sp in spikes)
    assert torch.allclose(via_spikes, bl.forward_integer(q, s, zp), rtol=1e-5, atol=1e-5)


def test_signed_spikes_reject_out_of_range_counts():
    with pytest.raises(ValueError):
        decompose_signed_to_spikes(torch.tensor([8.0]), bits=4)
    with pytest.raises(ValueError):
        decompose_signed_to_spikes(torch.tensor([-8.0]), T=4, bits=4)
