"""Flexibly Adjustable Integer LIF neuron (FAI-LIF).

Two implementations live in this module:

  * :class:`FAILIF` — the **default**. Uses a per-instance
    :class:`MovingAverageMinMaxObserver` to track the activation range
    online and quantize spikes accordingly. This matches NSLLM's
    activation-quantization approach. See DECISIONS.md #6 / #7 / #9
    for why the fixed-threshold variant doesn't suffice for Stage A.
  * :class:`FAILIFFixedThreshold` — the legacy implementation with a
    single fixed ``v_thr``. Kept so the Phase 1 MNIST sanity gate
    (which passes with the fixed threshold) stays reproducible. New
    code paths should use :class:`FAILIF`.

Both classes preserve the same outward contract:

  * Single-timestep emission of an integer spike count.
  * Stateful membrane potential ``V`` (non-persistent buffer).
  * Per-neuron learnable ``beta`` (leak) and ``gamma`` (hard/soft
    reset mix).
  * Output dtype matches the input dtype.
  * ``reset_state``, lazy state init, and a ``forward`` that accepts
    either ``(B, H)`` or ``(B, T, H)`` (looped internally).

Reset semantics: ``V_new = γ * (V - dequant) + (1 - γ) * 0`` where
``dequant = scale * (spike - zero_point)`` for the observer variant (the
affine dequantized value; subtracting ``scale * spike`` alone leaves a
``scale * zp`` error per step that runs the membrane away when the
zero-point is large, e.g. per-channel grids). This is one
deliberate simplification over a fully-detailed neuromorphic LIF: γ=0
zeroes ``V`` every step (not only on fire). It matches the Phase 3.5
spec. If we later want fire-conditional hard reset, gate the term by
``(spike != 0)``.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn

from spiking_ternary_lm.neurons.fai_lif_op import fai_lif_apply, lsq_fai_lif_apply
from spiking_ternary_lm.quantization.observers import MovingAverageMinMaxObserver
from spiking_ternary_lm.quantization.ste import ste_quantize_integer


# =========================================================================
# Helpers
# =========================================================================


def _ste_round_clip(x: Tensor, lo: float, hi: float) -> Tensor:
    """``clip(round(x), lo, hi)`` forward, identity backward.

    Used by :class:`FAILIFFixedThreshold` for the legacy path.
    """
    target = x.round().clamp(lo, hi)
    return x + (target - x).detach()


def _dequant_ste(
    spike: Tensor, scale: Tensor, zero_point: Tensor, in_range: Tensor | None = None,
) -> Tensor:
    """``scale * (spike - zero_point)`` whose gradient w.r.t. the membrane is 1.

    ``spike`` comes out of an identity STE (d spike / d v_pre = 1), so
    multiplying it by ``scale`` would give a gradient of ``scale`` instead of
    the straight-through 1 of the dequantized value. The value is taken
    detached and the zero-valued ``spike - spike.detach()`` carries the
    gradient. ``in_range`` (bool mask, optional) zeroes that gradient where
    the membrane fell outside the grid (clipped STE).
    """
    through = spike - spike.detach()
    if in_range is not None:
        through = through * in_range.to(through.dtype)
    return (scale * (spike - zero_point)).detach() + through


def _grid_mask(v_pre: Tensor, scale: Tensor, zero_point: Tensor, qmin: int, qmax: int) -> Tensor:
    """True where ``round(v_pre / scale + zp)`` is not clipped to ``[qmin, qmax]``."""
    u = v_pre.detach().float() / scale.float() + zero_point.float()
    return (u >= qmin - 0.5) & (u <= qmax + 0.5)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


# =========================================================================
# Legacy FAI-LIF with a single fixed v_thr (Phase 1 MNIST gate)
# =========================================================================


class FAILIFFixedThreshold(nn.Module):
    """Integer-output LIF with a fixed ``v_thr`` (legacy).

    Single-timestep training; per-neuron learnable ``beta`` and
    ``gamma``; ``v_thr`` is either a buffer or a learnable parameter.
    Identity-STE through ``clip(round(V/v_thr), n_min, n_max)``. The
    reset term is detached from autograd (NSLLM pitfall #1).
    """

    def __init__(
        self,
        hidden_size: int,
        bits: int = 4,
        signed: bool = False,
        init_beta: float = 0.9,
        init_gamma: float = 0.5,
        init_v_thr: float = 1.0,
        learnable_v_thr: bool = False,
    ) -> None:
        super().__init__()
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        self.hidden_size = int(hidden_size)
        self.bits = int(bits)
        self.signed = bool(signed)
        if self.signed:
            self.n_min = -(2 ** (self.bits - 1))
            self.n_max = 2 ** (self.bits - 1) - 1
        else:
            self.n_min = 0
            self.n_max = 2**self.bits - 1

        self.beta_logit = nn.Parameter(
            torch.full((self.hidden_size,), _logit(init_beta), dtype=torch.float32)
        )
        self.gamma_logit = nn.Parameter(
            torch.full((self.hidden_size,), _logit(init_gamma), dtype=torch.float32)
        )
        v_thr_init = torch.full((self.hidden_size,), float(init_v_thr), dtype=torch.float32)
        if learnable_v_thr:
            self.v_thr = nn.Parameter(v_thr_init)
        else:
            self.register_buffer("v_thr", v_thr_init, persistent=False)

        self.register_buffer("v", torch.zeros(0), persistent=False)

    @property
    def beta(self) -> Tensor:
        return torch.sigmoid(self.beta_logit)

    @property
    def gamma(self) -> Tensor:
        return torch.sigmoid(self.gamma_logit)

    def reset_state(
        self,
        batch_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device = device or (self.v.device if self.v.numel() else self.beta_logit.device)
        dtype = dtype or (self.v.dtype if self.v.numel() else torch.float32)
        if batch_shape is None:
            self.v = torch.zeros(0, device=device, dtype=dtype)
        else:
            self.v = torch.zeros((*batch_shape, self.hidden_size), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected last dim {self.hidden_size}, got {x.shape[-1]} (full shape {x.shape})"
            )
        if self.v.numel() == 0 or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x)

        beta = self.beta.to(x.dtype)
        gamma = self.gamma.to(x.dtype)
        v_thr = self.v_thr.to(x.dtype)

        v_pre = beta * self.v + x
        n_float = v_pre / v_thr
        n = _ste_round_clip(n_float, self.n_min, self.n_max)

        fire = (n != 0).to(v_pre.dtype)
        soft_term = n * v_thr
        hard_term = v_pre
        reset_term = fire * ((1.0 - gamma) * hard_term + gamma * soft_term)

        new_v = v_pre - reset_term.detach()
        self.v = new_v.detach()
        return n

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, bits={self.bits}, "
            f"signed={self.signed}, range=[{self.n_min},{self.n_max}]"
        )


# =========================================================================
# Default FAI-LIF with moving-average MinMax observer
# =========================================================================


class FAILIF(nn.Module):
    """Integer-output LIF with online activation-range calibration.

    The membrane potential ``V`` is integrated in FP. A
    :class:`MovingAverageMinMaxObserver` tracks its running range and
    derives a per-tensor ``(scale, zero_point)`` that re-maps ``V`` into
    the integer range ``[qmin, qmax]`` via STE-quantization. The reset
    is done in continuous space using ``scale * spike``.

    Parameters
    ----------
    num_neurons:
        Channel dim (per-neuron learnable params have this size).
    num_bits:
        Activation bit-width. Default 4 (NSLLM target).
    signed:
        Quantized-range sign. Default False.
    beta_init:
        Initial leak factor (sigmoid logit fit to this). 0.9 ≈ slow
        leak; many papers use higher.
    gamma_init:
        Initial hard/soft mix. ``1.0`` ⇒ pure soft reset (subtract
        ``scale * spike``); ``0.0`` ⇒ V always zeroed. Default 1.0 per
        Phase 3.5 spec (soft reset is the safe choice; pure-hard
        ``γ=0`` zeroes V every step which kills integration).
    learnable_beta, learnable_gamma:
        Whether the per-neuron leak / reset-mix are learnable.
    observer_momentum:
        EMA momentum for the activation observer. 0.01 is conservative.
    """

    def __init__(
        self,
        num_neurons: int,
        num_bits: int = 4,
        signed: bool = False,
        beta_init: float = 0.9,
        gamma_init: float = 1.0,
        learnable_beta: bool = True,
        learnable_gamma: bool = True,
        observer_momentum: float = 0.01,
        seq_chunk_size: int = 0,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")
        self.num_neurons = int(num_neurons)
        self.num_bits = int(num_bits)
        self.signed = bool(signed)
        self.learnable_beta = bool(learnable_beta)
        self.learnable_gamma = bool(learnable_gamma)
        # CUDA sequence-mode chunk size: bound the kernel's within-call
        # (scale, zp) staleness to this many timesteps before the
        # observer is refreshed and the kernel is invoked again. 0 ⇒
        # one call for the whole sequence (max staleness).
        self.seq_chunk_size = int(seq_chunk_size)
        # When True the observer keeps its current running stats even in
        # train mode (quantization grid frozen). Set by the trainer after a
        # warm-up to rule out grid drift as a source of late instability.
        self.observer_frozen = False
        # Activation quantizer: "minmax" (observer-tracked grid, default),
        # "lsq" (learned step size + offset, see :meth:`enable_lsq`) or
        # "none" (identity: no quantization, no membrane; DECISIONS #15).
        self.act_quant = "minmax"
        # MinMax output: "int" (legacy: the raw spike count q) or "dequant"
        # (``scale * (q - zp)`` with an identity STE w.r.t. the membrane;
        # DECISIONS #15). LSQ always outputs the dequantized value.
        self.output_mode = "int"
        # Dequantized outputs (MinMax "dequant" and LSQ): zero the STE
        # gradient where the membrane is clipped by the grid (DEC-17;
        # DEC-4's unmasked STE otherwise).
        self.ste_clip = False
        # True ⇒ beta = 0: no membrane carried across tokens, the LIF is a
        # plain quantizer (DECISIONS #15 rungs 0-3).
        self.stateless = False
        self.lsq_init_steps = 1
        # Last integer spike tensor (LSQ mode only; output is dequantized).
        self.last_spike: Tensor | None = None
        # Calibration: if True, observer updates stats but the spike
        # output is meaningless (we still produce one for shape parity,
        # but don't reset the membrane potential). Set via
        # `set_calibrating(True/False)`; never the in-place modification
        # of `.training` because eval-mode is still needed independently.
        self.calibrating: bool = False

        # Per-neuron beta (leak).
        if self.learnable_beta:
            self.beta_logit = nn.Parameter(
                torch.full((self.num_neurons,), _logit(beta_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "beta_const",
                torch.full((self.num_neurons,), float(beta_init), dtype=torch.float32),
                persistent=False,
            )

        # Per-neuron gamma (hard/soft reset mix).
        if self.learnable_gamma:
            self.gamma_logit = nn.Parameter(
                torch.full((self.num_neurons,), _logit(gamma_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "gamma_const",
                torch.full((self.num_neurons,), float(gamma_init), dtype=torch.float32),
                persistent=False,
            )

        # Activation observer. NOTE: it must be a sub-module so its
        # buffers travel with the FAI-LIF on ``.to(device)``.
        self.observer = MovingAverageMinMaxObserver(
            num_bits=num_bits, signed=signed, momentum=observer_momentum,
            symmetric=symmetric,
        )

        # Membrane potential. Lazy init on first forward.
        self.register_buffer("v", torch.zeros(0), persistent=False)

    # ----------------------------------------------------------------------
    # LSQ mode (DECISIONS #14): the grid becomes two learnable scalars.
    # ----------------------------------------------------------------------

    def enable_lsq(self, init_steps: int = 1) -> None:
        """Switch this LIF to a learned step size (LSQ, Esser et al. 2020).

        ``lsq_log_scale`` (``s = exp(.)``) and ``lsq_zp`` (integer units) are
        per-tensor parameters. They are initialised from the MinMax observer
        after ``init_steps`` training sequence-forwards (the observer keeps
        tracking during that warm-up, then stops), so a MinMax checkpoint can
        be loaded as-is. ``lsq_initialized`` is a persistent buffer so a saved
        LSQ checkpoint does not re-init.
        """
        if self.act_quant == "lsq":
            return
        self.act_quant = "lsq"
        self.lsq_init_steps = int(init_steps)
        self.lsq_log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.lsq_zp = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # Symmetric grids keep zp = 0 (DECISIONS #15): the offset is not learnt.
        self.lsq_zp.requires_grad_(not self.observer.symmetric)
        self.register_buffer("lsq_initialized", torch.tensor(0, dtype=torch.long))
        self.register_buffer("lsq_seen", torch.tensor(0, dtype=torch.long), persistent=False)

    def set_observer_scope(self, scope: str) -> None:
        """Replace the observer by a per-tensor or per-channel one (fresh stats).

        Per-channel = one (scale, zero-point) per neuron, i.e. a per-neuron
        threshold. The old scalar buffers stay in the state_dict so a
        per-tensor checkpoint loads; the per-channel grid is then initialised
        from the first training batch.
        """
        if scope not in ("tensor", "channel"):
            raise ValueError(f"observer scope must be 'tensor' or 'channel', got {scope!r}")
        want_channel = scope == "channel"
        if want_channel == self.observer.per_channel:
            return
        self.observer = MovingAverageMinMaxObserver(
            num_bits=self.observer.num_bits, signed=self.observer.signed,
            momentum=self.observer.momentum, eps=self.observer.eps,
            channel_dim=self.num_neurons if want_channel else None,
            symmetric=self.observer.symmetric,
        ).to(self.observer.running_min.device)

    def _lsq_init_from_observer(self) -> None:
        with torch.no_grad():
            scale, zp = self.observer.get_scale_and_zero_point()
            self.lsq_log_scale.copy_(scale.to(torch.float32).clamp(min=self.observer.eps).log())
            self.lsq_zp.copy_(zp.to(torch.float32))
            self.lsq_initialized.fill_(1)

    def lsq_scale_and_zero_point(self) -> tuple[Tensor, Tensor]:
        """Current learned grid as ``(scale, zero_point)`` tensors (no grad)."""
        return self.lsq_log_scale.detach().exp(), self.lsq_zp.detach()

    # ----------------------------------------------------------------------
    # Compatibility shims so the rest of the codebase can still use
    # the old field names (hidden_size, bits, n_min, n_max).
    # ----------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.num_neurons

    @property
    def bits(self) -> int:
        return self.num_bits

    @property
    def n_min(self) -> int:
        return int(self.observer.qmin)

    @property
    def n_max(self) -> int:
        return int(self.observer.qmax)

    @property
    def beta(self) -> Tensor:
        if self.stateless:
            return torch.zeros_like(self.beta_logit if self.learnable_beta else self.beta_const)
        if self.learnable_beta:
            return torch.sigmoid(self.beta_logit)
        return self.beta_const

    @property
    def gamma(self) -> Tensor:
        if self.learnable_gamma:
            return torch.sigmoid(self.gamma_logit)
        return self.gamma_const

    # ----------------------------------------------------------------------
    # State / mode control
    # ----------------------------------------------------------------------

    def reset_state(
        self,
        batch_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device = device or (self.v.device if self.v.numel() else self.observer.running_min.device)
        dtype = dtype or (self.v.dtype if self.v.numel() else torch.float32)
        if batch_shape is None:
            self.v = torch.zeros(0, device=device, dtype=dtype)
        else:
            self.v = torch.zeros((*batch_shape, self.num_neurons), device=device, dtype=dtype)

    def set_calibrating(self, flag: bool) -> None:
        """Toggle calibration mode for this LIF.

        While calibrating, the observer still updates its stats (since
        ``self.training`` is True during calibration), but the forward
        pass skips the reset and returns the FP membrane value cast to
        the input dtype. The caller is expected to discard the output.
        """
        self.calibrating = bool(flag)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.num_neurons:
            raise ValueError(
                f"expected last dim {self.num_neurons}, got {x.shape[-1]} (full shape {x.shape})"
            )
        if self.act_quant == "none":
            return x
        # Sequence-shaped input ``(B, T, H)``: route to the fused CUDA
        # kernel when available. Falls back to a per-step loop on CPU /
        # if the kernel isn't loaded.
        if x.dim() == 3:
            return self._forward_sequence(x)
        if x.dim() != 2:
            raise ValueError(
                f"FAILIF expects (B, H) or (B, T, H), got {tuple(x.shape)}"
            )
        return self._forward_step(x)

    # ----------------------------------------------------------------------
    # Forward — single timestep (B, H). Original Python path; preserved
    # exactly for callers that still loop FAI-LIF per-step.
    # ----------------------------------------------------------------------

    def _forward_step(self, x: Tensor) -> Tensor:
        if self.v.numel() == 0 or self.v.shape != x.shape or self.v.device != x.device:
            self.v = torch.zeros_like(x)

        beta = self.beta.to(x.dtype)
        gamma = self.gamma.to(x.dtype)

        v_pre = beta * self.v + x

        # Observer pass-through (stats update if in training mode).
        if not self.observer_frozen and not (
            self.act_quant == "lsq" and int(self.lsq_initialized) == 1
        ):
            self.observer(v_pre)

        if self.calibrating:
            # Skip quantization & reset. Membrane is detached so we don't
            # accumulate autograd graph across calibration batches.
            self.v = v_pre.detach()
            # Return the FP value so the caller's shape/dtype contract
            # is preserved; values are NOT meaningful integer spikes.
            return v_pre.to(x.dtype)

        if self.act_quant == "lsq" and int(self.lsq_initialized) == 1:
            scale, zero_point = self.lsq_scale_and_zero_point()
        else:
            scale, zero_point = self.observer.get_scale_and_zero_point()
        # Cast to x.dtype for the multiplication below.
        scale_dt = scale.to(x.dtype)
        zp_dt = zero_point.to(x.dtype)
        spike = ste_quantize_integer(
            v_pre, scale_dt, zp_dt, self.observer.qmin, self.observer.qmax
        )

        # Reset in continuous space. ``scale * spike`` is the
        # dequantized "magnitude" of the emitted spike. We detach from
        # autograd (per the same pitfall #1 reasoning as the
        # fixed-threshold variant). γ=1 ⇒ pure soft reset; γ=0 ⇒ V → 0
        # every step.
        spike_continuous = (scale_dt * (spike - zp_dt)).detach()
        new_v = gamma * (v_pre - spike_continuous)
        self.v = new_v.detach()
        if self.act_quant == "lsq" and int(self.lsq_initialized) == 1:
            self.last_spike = spike.detach()
            return scale_dt * (spike - zp_dt)
        if self.output_mode == "dequant":
            self.last_spike = spike.detach()
            mask = (
                _grid_mask(v_pre, scale, zero_point, self.observer.qmin, self.observer.qmax)
                if self.ste_clip else None
            )
            return _dequant_ste(spike, scale_dt, zp_dt, mask)
        return spike

    # ----------------------------------------------------------------------
    # Forward — full sequence (B, T, H). Fused-kernel path on CUDA.
    #
    # Observer-equivalence:
    #   The kernel takes a SINGLE (scale, zp) and uses it for all T steps
    #   (the within-call scale is "frozen" relative to the per-step Python
    #   reference). That's a fixed cost we accept for fusion speed.
    #
    #   What we DO match exactly is the per-step Python reference's
    #   observer EMA UPDATE STREAM. After the kernel returns we reduce
    #   ``v_pre_seq`` to (T,) per-step min/max over the (B, H) plane and
    #   feed those T values to ``observer.update_stats_sequence(...)``,
    #   which runs T sequential EMA updates of running_min/max — exactly
    #   what the per-step path would have done. The reduction is in FP32
    #   for numerical stability; the kernel's ``v_pre_seq`` retains its
    #   FP32 storage but the spike output is cast back to ``x.dtype`` to
    #   honour the outward dtype contract.
    # ----------------------------------------------------------------------

    def _forward_sequence(self, x: Tensor) -> Tensor:
        B, T, H = x.shape

        # Lazy-init membrane to shape (B, H). If a stale shape is present
        # from prior calls (e.g. a different B, or the 2D path), reset.
        target_v_shape = (B, H)
        if (
            self.v.numel() == 0
            or tuple(self.v.shape) != target_v_shape
            or self.v.device != x.device
        ):
            self.v = torch.zeros(target_v_shape, device=x.device, dtype=torch.float32)

        if self.calibrating:
            # Calibration: per-step fallback (kernel can't express the
            # "v_new = v_pre, no reset" calibrating semantic). Cheap because
            # calibration is a brief one-shot pass.
            outs = []
            for t in range(T):
                outs.append(self._forward_step(x[:, t]))
            return torch.stack(outs, dim=1)

        # --- LSQ path: one fused call, grid = learnable (scale, offset) ----
        if self.act_quant == "lsq":
            if int(self.lsq_initialized) == 0:
                # Warm-up: MinMax path (observer tracking) for lsq_init_steps
                # training calls, then freeze the grid into the parameters.
                out = self._forward_sequence_minmax(x)
                if self.training:
                    self.lsq_seen += 1
                    if int(self.lsq_seen) >= self.lsq_init_steps:
                        self._lsq_init_from_observer()
                return out
            x_hat, spike_seq, _v_pre, v_final = lsq_fai_lif_apply(
                x.to(torch.float32).contiguous(),
                self.beta.to(torch.float32).contiguous(),
                self.gamma.to(torch.float32).contiguous(),
                self.v.to(torch.float32),
                self.lsq_log_scale, self.lsq_zp, self.observer.qmin, self.observer.qmax,
                self.ste_clip,
            )
            self.v = v_final.detach()
            # Integer spikes kept for the spike-stats hooks (the module
            # output is the dequantized value in LSQ mode).
            self.last_spike = spike_seq.detach()
            return x_hat.to(x.dtype)

        return self._forward_sequence_minmax(x)

    def _forward_sequence_minmax(self, x: Tensor) -> Tensor:
        B, T, H = x.shape
        # CPU path (no kernel): fall back to looping the per-step path.
        # This preserves bit-exact CPU behaviour with the legacy code.
        if not x.is_cuda:
            outs = []
            for t in range(T):
                outs.append(self._forward_step(x[:, t]))
            return torch.stack(outs, dim=1)

        # --- CUDA chunked fused path -----------------------------------------
        # The kernel uses a SINGLE (scale, zp) per call, frozen at call
        # start. Calling once for the whole sequence makes the quant grid
        # ``T`` steps stale by the end — empirically costly (~55% of the
        # ppl regression vs the per-step Python reference). To bound that
        # staleness, split T into ``T_chunk``-sized chunks, call the
        # kernel once per chunk, and refresh (scale, zp) between chunks.
        # Within a chunk the grid is still frozen, but only over
        # ``T_chunk`` steps. We additionally feed per-step (min, max)
        # back to the observer between chunks so the EMA update stream
        # matches the per-step reference's bit-for-bit.
        x32 = x.to(torch.float32).contiguous()
        beta32 = self.beta.to(torch.float32).contiguous()
        gamma32 = self.gamma.to(torch.float32).contiguous()
        v_curr = self.v.to(torch.float32)

        T_chunk = min(self.seq_chunk_size, T) if self.seq_chunk_size > 0 else T
        spike_chunks: list[Tensor] = []
        int_chunks: list[Tensor] = []
        for t0 in range(0, T, T_chunk):
            t1 = min(t0 + T_chunk, T)
            x_chunk = x32[:, t0:t1].contiguous()

            # Refresh (scale, zp) from current observer state.
            scale, zero_point = self.observer.get_scale_and_zero_point()
            scale_f = scale.to(torch.float32)
            zp_f = zero_point.to(torch.float32)

            spike_c, v_pre_c, v_final_c = fai_lif_apply(
                x_chunk, beta32, gamma32, v_curr,
                scale_f, zp_f,
                self.observer.qmin, self.observer.qmax,
            )

            # Per-step EMA updates for this chunk's timesteps.
            if self.training and not self.observer_frozen:
                v_pre_fp32 = v_pre_c.detach()
                if v_pre_fp32.dtype != torch.float32:
                    v_pre_fp32 = v_pre_fp32.to(torch.float32)
                red = (0,) if self.observer.per_channel else (0, 2)
                step_mins = v_pre_fp32.amin(dim=red)
                step_maxes = v_pre_fp32.amax(dim=red)
                self.observer.update_stats_sequence(step_mins, step_maxes)

            if self.output_mode == "dequant":
                # Each chunk has its own grid: dequantize with it.
                int_chunks.append(spike_c.detach())
                mask = (
                    _grid_mask(v_pre_c, scale_f, zp_f, self.observer.qmin, self.observer.qmax)
                    if self.ste_clip else None
                )
                spike_c = _dequant_ste(spike_c, scale_f, zp_f, mask)
            spike_chunks.append(spike_c)
            # Keep v_curr in the autograd graph across chunks so BPTT
            # works through the whole sequence (caller detaches between
            # training windows via reset_state / detach_state).
            v_curr = v_final_c

        # Persist membrane state across calls.
        self.v = v_curr.detach()
        if int_chunks:
            self.last_spike = torch.cat(int_chunks, dim=1)
        spike_seq = torch.cat(spike_chunks, dim=1)
        return spike_seq.to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"num_neurons={self.num_neurons}, num_bits={self.num_bits}, "
            f"signed={self.signed}, range=[{self.n_min},{self.n_max}], "
            f"calibrating={self.calibrating}"
        )


# =========================================================================
# Model-wide helpers
# =========================================================================


def set_observer_scope(model: nn.Module, scope: str) -> int:
    """Set the observer scope ("tensor" | "channel") on every :class:`FAILIF`."""
    n = 0
    for m in model.modules():
        if isinstance(m, FAILIF):
            m.set_observer_scope(scope)
            n += 1
    return n


@torch.no_grad()
def calibrate_observers(model: nn.Module, batches: Iterable[Tensor]) -> int:
    """Set every MinMax observer to the global (min, max) of its FAI-LIF input.

    The model runs ``batches`` (token-id tensors) with FP activations and a
    fresh state per batch; each observer then starts from the range its site
    actually sees, instead of the first timestep of the first training batch
    (which clips the later timesteps: 23 % logit error at 8 bits on a tiny
    RWKV-4 vs 2.8 % calibrated). Only meaningful with dequantized outputs,
    where the FP pass sees the scales the quantized model will see (with the
    legacy integer outputs it inflates the deep ranges, DECISIONS #11).
    LSQ LIFs not initialised yet get their step and offset from the
    calibrated range. Per-tensor observers only. Returns the number of
    observers set.
    """
    lifs = [
        m for m in model.modules()
        if isinstance(m, FAILIF) and (
            m.act_quant == "minmax"
            or (m.act_quant == "lsq" and int(m.lsq_initialized) == 0)
        )
    ]
    modes = {id(m): m.act_quant for m in lifs}
    if any(m.observer.per_channel for m in lifs):
        raise ValueError("calibrate_observers supports per-tensor observers only")
    lo: dict[int, Tensor] = {}
    hi: dict[int, Tensor] = {}

    def record(mod: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        x = inputs[0].detach().float()
        k = id(mod)
        lo[k] = torch.minimum(lo[k], x.min()) if k in lo else x.min()
        hi[k] = torch.maximum(hi[k], x.max()) if k in hi else x.max()

    was_training = model.training
    handles = [m.register_forward_pre_hook(record) for m in lifs]
    for m in lifs:
        m.act_quant = "none"
    model.eval()
    try:
        for ids in batches:
            model.reset_state()
            model(ids)
    finally:
        for h in handles:
            h.remove()
        for m in lifs:
            m.act_quant = modes[id(m)]
        model.reset_state()
        model.train(was_training)
    for m in lifs:
        obs = m.observer
        obs.running_min.copy_(lo[id(m)])
        obs.running_max.copy_(hi[id(m)])
        obs.num_updates.fill_(1)
        if m.act_quant == "lsq":
            m._lsq_init_from_observer()
    return len(lifs)


def set_output_mode(model: nn.Module, mode: str) -> int:
    """MinMax output convention on every :class:`FAILIF`: "int" or "dequant"."""
    if mode not in ("int", "dequant"):
        raise ValueError(f"output mode must be 'int' or 'dequant', got {mode!r}")
    n = 0
    for m in model.modules():
        if isinstance(m, FAILIF):
            m.output_mode = mode
            n += 1
    return n


def set_ste_clip(model: nn.Module, flag: bool = True) -> int:
    """Clipped STE on every :class:`FAILIF` (dequant output only)."""
    n = 0
    for m in model.modules():
        if isinstance(m, FAILIF):
            m.ste_clip = bool(flag)
            n += 1
    return n


def set_stateless(model: nn.Module, flag: bool = True) -> int:
    """beta = 0 on every :class:`FAILIF` (no membrane carried across tokens)."""
    n = 0
    for m in model.modules():
        if isinstance(m, FAILIF):
            m.stateless = bool(flag)
            n += 1
    return n


def disable_act_quant(model: nn.Module) -> int:
    """Turn every :class:`FAILIF` into an identity (FP activations)."""
    n = 0
    for m in model.modules():
        if isinstance(m, FAILIF):
            m.act_quant = "none"
            n += 1
    return n


def enable_lsq(model: nn.Module, init_steps: int = 1) -> int:
    """Switch every :class:`FAILIF` in ``model`` to LSQ. Returns the count."""
    n = 0
    for m in model.modules():
        if isinstance(m, FAILIF):
            m.enable_lsq(init_steps=init_steps)
            n += 1
    return n


def set_calibrating(model: nn.Module, flag: bool) -> int:
    """Set ``calibrating`` on every observer-based :class:`FAILIF` in a model.

    Returns the number of FAILIFs flipped.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, FAILIF):
            m.set_calibrating(flag)
            n += 1
    return n
