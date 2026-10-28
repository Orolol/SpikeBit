"""Tests for the integer→binary spike train decomposition."""
from __future__ import annotations

import pytest
import torch

from spiking_ternary_lm.neurons.binary_lif import BinaryLIF, decompose_integer_to_binary
from spiking_ternary_lm.neurons.fai_lif import FAILIFFixedThreshold as FAILIF


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
