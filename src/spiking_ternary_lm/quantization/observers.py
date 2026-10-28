"""Moving-Average MinMax activation observers.

Replaces the fixed ``v_thr`` hyperparameter in FAI-LIF with a per-tensor
running-stat estimate of the activation range, EMA-smoothed across
training steps. NSLLM-style activation quantization (DECISIONS.md #6,
#7 — activation quant is the most fragile component of the stack).

Behaviour:
  * ``train()`` mode: every forward call updates ``running_min`` and
    ``running_max`` via exponential moving average of the per-batch
    min/max. First call initialises the running stats *directly* (no
    EMA bias toward whatever zero-init we started with).
  * ``eval()`` mode: stats are frozen. Eval batches do NOT contaminate
    the running statistics.
  * ``calibrating`` mode (extra flag): same as train for the update,
    but the caller is expected to skip the quantization step in the
    enclosing module so the model output is meaningless during the
    warm-up pass. The observer itself does not enforce this — it's a
    contract between the observer and its caller (see ``FAILIF``).

Per-tensor reduction only in v0. Per-channel is possible but rarely
needed for activations and not used by NSLLM.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MovingAverageMinMaxObserver(nn.Module):
    """Track per-tensor min / max with an exponential moving average.

    Parameters
    ----------
    num_bits:
        Activation bit-width. With ``signed=False`` the quantized range
        is ``[0, 2**num_bits - 1]``; with ``signed=True`` it is
        ``[-2**(num_bits-1), 2**(num_bits-1) - 1]``.
    signed:
        Whether to use a signed quantized range.
    momentum:
        EMA momentum. Lower = slower update = more robust to outlier
        batches. NSLLM-style start value is 0.01.
    eps:
        Floor on the observed range; prevents zero-scale when the
        observed distribution is constant.
    """

    def __init__(
        self,
        num_bits: int = 4,
        signed: bool = False,
        momentum: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")
        self.num_bits = int(num_bits)
        self.signed = bool(signed)
        self.momentum = float(momentum)
        self.eps = float(eps)

        # Buffers (saved with state_dict; survive .to(device)).
        self.register_buffer("running_min", torch.tensor(0.0))
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        if self.signed:
            self.qmin = -(2 ** (self.num_bits - 1))
            self.qmax = 2 ** (self.num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**self.num_bits - 1

    # ----------------------------------------------------------------------
    # Stats update
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def update_stats(self, x: Tensor) -> None:
        """EMA-update running_min/running_max from the current batch."""
        cur_min = x.min().detach().to(self.running_min.dtype).to(self.running_min.device)
        cur_max = x.max().detach().to(self.running_max.dtype).to(self.running_max.device)
        if int(self.num_updates) == 0:
            # First batch: initialise directly to avoid EMA bias against the
            # zero-initialised running buffers.
            self.running_min.copy_(cur_min)
            self.running_max.copy_(cur_max)
        else:
            m = self.momentum
            self.running_min.mul_(1 - m).add_(cur_min, alpha=m)
            self.running_max.mul_(1 - m).add_(cur_max, alpha=m)
        self.num_updates += 1

    @torch.no_grad()
    def update_stats_sequence(self, step_mins: Tensor, step_maxes: Tensor) -> None:
        """Run T sequential EMA updates from pre-reduced per-step (min, max).

        Equivalent to calling :meth:`update_stats` on each timestep slice of
        a ``(B, T, H)`` activation tensor, but without ever materialising the
        per-step tensors in the caller. ``step_mins`` and ``step_maxes`` are
        ``(T,)`` FP tensors of per-step min/max values reduced over the
        ``(B, H)`` plane.

        The first call to the observer (``num_updates == 0``) takes the
        first timestep's stats directly (no EMA bias against the zero-init
        buffers), matching the per-step :meth:`update_stats` semantic.
        Subsequent steps EMA-blend with ``momentum``. Buffers stay on their
        original device/dtype.
        """
        if step_mins.shape != step_maxes.shape:
            raise ValueError(
                f"step_mins/step_maxes shape mismatch: {tuple(step_mins.shape)} "
                f"vs {tuple(step_maxes.shape)}"
            )
        if step_mins.dim() != 1:
            raise ValueError(
                f"step_mins/step_maxes must be 1-D (T,), got {tuple(step_mins.shape)}"
            )
        T = int(step_mins.numel())
        if T == 0:
            return

        # Bring both into the buffer's dtype/device once.
        mins = step_mins.detach().to(
            dtype=self.running_min.dtype, device=self.running_min.device
        )
        maxes = step_maxes.detach().to(
            dtype=self.running_max.dtype, device=self.running_max.device
        )

        # Closed form of T sequential EMA steps, evaluated as one dot
        # product instead of 2T scalar kernels (launch-bound otherwise):
        #   new = (1-m)^T * old + sum_t m (1-m)^(T-1-t) x_t
        # The first-call init (take x_0 exactly, then EMA over t >= 1)
        # collapses to the same expression with ``old := x_0``, so the
        # branch is a device-side ``where`` -- no host sync on
        # ``num_updates``.
        decay, weights = self._sequence_weights(T, mins)
        is_first = self.num_updates == 0
        old_min = torch.where(is_first, mins[0], self.running_min)
        old_max = torch.where(is_first, maxes[0], self.running_max)
        self.running_min.copy_(decay * old_min + torch.dot(weights, mins))
        self.running_max.copy_(decay * old_max + torch.dot(weights, maxes))
        self.num_updates += T

    def _sequence_weights(self, T: int, like: Tensor) -> tuple[Tensor, Tensor]:
        """``((1-m)^T, [m (1-m)^(T-1-t)]_t)`` cached per (T, device, dtype)."""
        key = (T, like.device, like.dtype)
        cached = getattr(self, "_seq_weights_cache", None)
        if cached is None or cached[0] != key:
            m = self.momentum
            exps = torch.arange(T - 1, -1, -1, device=like.device, dtype=torch.float64)
            weights = (m * (1.0 - m) ** exps).to(like.dtype)
            decay = torch.tensor((1.0 - m) ** T, device=like.device, dtype=like.dtype)
            cached = (key, decay, weights)
            self._seq_weights_cache = cached
        return cached[1], cached[2]

    def get_scale_and_zero_point(self) -> tuple[Tensor, Tensor]:
        """Return ``(scale, zero_point)`` from current running stats.

        Both are scalar tensors. The standard affine quantization mapping
        is::

            q = round(x / scale + zero_point).clamp(qmin, qmax)
            x_recon = (q - zero_point) * scale
        """
        range_val = (self.running_max - self.running_min).clamp(min=self.eps)
        n_levels = self.qmax - self.qmin
        scale = range_val / n_levels
        # Solve so x = running_min maps to q = qmin:
        #   qmin = round(running_min / scale + zero_point)
        #   zero_point = qmin - round(running_min / scale)
        zero_point = self.qmin - (self.running_min / scale).round()
        return scale, zero_point

    # ----------------------------------------------------------------------
    # Forward (pass-through, only updates stats)
    # ----------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self.update_stats(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"num_bits={self.num_bits}, signed={self.signed}, momentum={self.momentum}, "
            f"range=[{self.qmin},{self.qmax}], updates={int(self.num_updates)}"
        )
