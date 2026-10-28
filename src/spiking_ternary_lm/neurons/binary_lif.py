"""Binary LIF — inference-time decomposition of integer spike counts.

At training, FAI-LIF emits a single integer-valued spike per call. At
inference, that integer is replayed as a sequence of *binary* spikes
across ``T`` virtual timesteps. This is the NSLLM trick that recovers
binary-spike energy efficiency without paying the BPTT cost of binary
training.

Two decomposition schemes are provided:

  * ``"sequential"`` — ``N`` spikes followed by ``T - N`` zeros. Simplest.
  * ``"even"``        — Bresenham distribution: ``spike[t] = floor((t+1) * N / T)
                        - floor(t * N / T)``. Sum equals ``N`` and spikes
                        are spread out evenly across the window.

The default is ``"even"`` (better matches energy distribution in real
neuromorphic substrates).

The decomposition is **not** used during Stage A/C training. It exists
to validate equivalence with FAI-LIF integer output (sum-preservation
test) and for energy-mode inference experiments.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

Scheme = Literal["sequential", "even"]


def _validate_input(n: Tensor, n_max: int) -> Tensor:
    if not torch.is_floating_point(n) and n.dtype not in (torch.int32, torch.int64, torch.int8):
        raise TypeError(f"unsupported dtype {n.dtype} for binary decomposition")
    n_int = n.round().to(torch.int64)
    if (n_int < 0).any():
        raise ValueError("binary decomposition currently assumes unsigned spike counts (n >= 0)")
    if (n_int > n_max).any():
        raise ValueError(f"spike count exceeds n_max={n_max}")
    return n_int


def _even_decomposition(n: Tensor, T: int) -> Tensor:
    """Bresenham-style distribution: floor((t+1)*N/T) - floor(t*N/T).

    Shape contract: ``n`` of shape ``S``; returns tensor of shape ``(T, *S)``
    with the same dtype as ``n`` (will be cast back from int64 at the end).
    """
    n64 = n.to(torch.int64)
    # `cumulative[t] = floor((t+1) * N / T)` for t = 0..T-1.
    # Compute with int math to stay exact.
    out = torch.empty((T, *n64.shape), dtype=torch.int64, device=n64.device)
    prev = torch.zeros_like(n64)
    for t in range(T):
        cur = ((t + 1) * n64) // T
        out[t] = cur - prev
        prev = cur
    return out


def _sequential_decomposition(n: Tensor, T: int) -> Tensor:
    """N ones followed by zeros.

    ``spike[t] = 1 if t < N else 0``.
    """
    n64 = n.to(torch.int64)
    t = torch.arange(T, device=n.device).view((T,) + (1,) * n64.dim())
    return (t < n64.unsqueeze(0)).to(torch.int64)


def decompose_integer_to_binary(
    n: Tensor,
    T: int | None = None,
    scheme: Scheme = "even",
    bits: int = 4,
) -> Tensor:
    """Decompose integer spike counts into ``T`` binary virtual timesteps.

    Parameters
    ----------
    n:
        Integer-valued tensor (any shape). May be float-typed, will be
        rounded.
    T:
        Number of virtual timesteps. Default is ``2**bits - 1`` (the
        smallest window that accommodates the maximum possible value).
    scheme:
        ``"even"`` (default) or ``"sequential"``.
    bits:
        Used to compute the default ``T`` and to validate the max range.

    Returns
    -------
    Tensor of shape ``(T, *n.shape)`` with values in ``{0, 1}``, dtype
    matching ``n.dtype`` (or int64 if ``n`` is integer-typed).

    Guarantees: ``out.sum(dim=0) == n`` (as integers).
    """
    if T is None:
        T = 2**bits - 1
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    n_max = 2**bits - 1
    n_int = _validate_input(n, n_max=n_max)
    if scheme == "even":
        out = _even_decomposition(n_int, T)
    elif scheme == "sequential":
        out = _sequential_decomposition(n_int, T)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    # Cast back to caller dtype (or keep int64 if input was int).
    if torch.is_floating_point(n):
        out = out.to(n.dtype)
    return out


class BinaryLIF(nn.Module):
    """Stateless wrapper around :func:`decompose_integer_to_binary`.

    Provided as a module for symmetry with :class:`FAILIF` and to make it
    easy to plug into a model and toggle inference modes. It carries no
    learnable parameters and no state.
    """

    def __init__(self, bits: int = 4, T: int | None = None, scheme: Scheme = "even") -> None:
        super().__init__()
        self.bits = int(bits)
        self.T = int(T) if T is not None else 2**self.bits - 1
        self.scheme: Scheme = scheme

    def forward(self, n: Tensor) -> Tensor:
        return decompose_integer_to_binary(n, T=self.T, scheme=self.scheme, bits=self.bits)

    def extra_repr(self) -> str:
        return f"bits={self.bits}, T={self.T}, scheme={self.scheme!r}"
