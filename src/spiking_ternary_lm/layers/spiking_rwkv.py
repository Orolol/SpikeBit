"""Spiking-Ternary RWKV-4 time-mixing block.

Implements the time-mixing (recurrent linear attention) part of one
RWKV-4 layer with:

  * Token-shift via per-channel learnable interpolation factors
    ``mu_r``, ``mu_k``, ``mu_v``.
  * Three projections through :class:`BitLinear` (ternary weights):
    receptance, key, value.
  * :class:`FAILIF` activations between projections and the WKV update.
  * Output projection through :class:`BitLinear` after a final FAI-LIF.
  * Recurrent WKV update kept in **FP32** for numerical stability,
    using the BlinkDL stable formulation that tracks a running
    log-max ``p`` to keep ``exp(...)`` finite.

WKV equations (per channel, per timestep), matching the official
RWKV-4 reference (BlinkDL/RWKV-LM):

    # Output for the current token: bonus ``u`` on the current key.
    ww = u + k_t
    qq = max(p, ww)
    e1 = exp(p - qq)
    e2 = exp(ww - qq)
    wkv_t = (e1 * a + e2 * v_t) / (e1 * b + e2)

    # State update for the NEXT token: decay ``w`` applied to state.
    decay = exp(time_decay)          # positive
    ww2  = p - decay                  # subtract from log-magnitude
    qq2  = max(ww2, k_t)
    e1   = exp(ww2 - qq2)
    e2   = exp(k_t - qq2)
    a_new = e1 * a + e2 * v_t
    b_new = e1 * b + e2
    p_new = qq2

    out = W_o @ FAI-LIF(sigmoid(r_t) * wkv_t)

State carried across timesteps: ``(x_prev, a, b, p)`` per batch element.
State is detached at the END of a ``forward`` call so the next call
doesn't extend the autograd graph. Within a single forward call over a
sequence, state IS grad-tracked → standard BPTT.

Open points (see DECISIONS.md):

  * Applying ``FAI-LIF`` to ``k`` (range {0..15} unsigned or {-8..7}
    signed) feeds an *exponentiated* activation in WKV. We default to
    **signed** for ``k`` and ``v`` to keep ``exp(k)`` in a manageable
    range; the spec's strict reading (unsigned) is reachable via
    ``signed_kv=False``. This is flagged for revisit in Phase 3.
  * ``sigmoid(r)`` applied to a FAI-LIF integer output collapses to ≈1
    for any non-zero spike count. We follow the spec literally; an
    alternative is to keep ``r`` as FP (skip FAI-LIF on R) and let
    sigmoid do the gating natively. Toggle via ``spiking_r``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spiking_ternary_lm.layers.wkv_op import wkv_forward as _wkv_fused
from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.quantization.bitlinear import BitLinear


def _init_token_shift(hidden_size: int, layer_idx: int, n_layers: int) -> Tensor:
    """RWKV-4 reference init for token-shift mixing factor ``mu``.

    Returns a tensor of shape ``(hidden_size,)`` in (0, 1). Earlier
    channels favour the current token, later channels favour the previous
    token. Layer depth biases the curve slightly so deeper layers see
    more of the past.
    """
    ratio_0_to_1 = layer_idx / max(n_layers - 1, 1)
    ratio_1_to_almost0 = 1.0 - layer_idx / n_layers
    x = torch.arange(hidden_size, dtype=torch.float32) / hidden_size
    return torch.pow(x, ratio_1_to_almost0)


def _init_time_decay(hidden_size: int, layer_idx: int, n_layers: int) -> Tensor:
    """RWKV-4 reference init for ``time_decay`` (the log of decay magnitude).

    Effective per-step state multiplier is ``exp(-exp(time_decay))``,
    so smaller (more negative) ``time_decay`` → slower decay (long
    memory), larger → faster decay (short memory). The init spreads
    channels across a frequency range.
    """
    ratio_0_to_1 = layer_idx / max(n_layers - 1, 1)
    decay_speed = torch.empty(hidden_size, dtype=torch.float32)
    for h in range(hidden_size):
        # Logarithmic spacing in (0, 5] with a per-layer bias.
        decay_speed[h] = -5.0 + 8.0 * (h / max(hidden_size - 1, 1)) ** (
            0.7 + 1.3 * ratio_0_to_1
        )
    return decay_speed  # raw parameter; effective per-step decay = exp(decay_speed) (positive)


def _init_time_first(hidden_size: int) -> Tensor:
    """RWKV-4 reference init for the per-channel bonus ``u``."""
    zigzag = (torch.arange(hidden_size, dtype=torch.float32) % 3 - 1) * 0.5
    return zigzag + math.log(0.3)  # ≈ -1.20 with channel-dependent jitter


class SpikingRWKVTimeMix(nn.Module):
    """One RWKV-4 time-mixing block in spiking-ternary form.

    Parameters
    ----------
    hidden_size:
        Channel dim of the residual stream.
    layer_idx, n_layers:
        Used by the RWKV init scheme. Pass ``layer_idx=0`` and
        ``n_layers=1`` for a standalone block.
    bits:
        Activation bit-width for FAI-LIF.
    group_size:
        BitLinear weight group size for ternary scale.
    weight_dtype:
        Shadow weight dtype for BitLinear (FP32 default for stability;
        BF16 elsewhere is fine, see DECISIONS.md #4).
    init_v_thr:
        Initial firing threshold for FAI-LIF. The activation quantizer is
        the most fragile component (DECISIONS.md #7); revisit in Phase 3
        with moving-average MinMax calibration.
    signed_kv:
        If True, use signed FAI-LIF for ``k`` and ``v`` (range
        ``[-2^(A-1), 2^(A-1)-1]``). Default True to control the dynamic
        range of ``exp(k)`` inside the WKV update.
    spiking_r:
        If True, apply FAI-LIF to ``r`` (then sigmoid the integer). If
        False, keep ``r`` as FP and let sigmoid gate natively. Default
        True (literal spec).
    """

    def __init__(
        self,
        hidden_size: int,
        layer_idx: int = 0,
        n_layers: int = 1,
        bits: int = 4,
        group_size: int = 128,
        weight_dtype: torch.dtype = torch.float32,
        observer_momentum: float = 0.01,
        signed_kv: bool = True,
        spiking_r: bool = True,
        # ``init_v_thr`` is retained as a kwarg only for back-compat with
        # older configs; the new observer-based FAILIF derives the scale
        # online, so this value is **ignored** by the default path.
        init_v_thr: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_idx = int(layer_idx)
        self.n_layers = int(n_layers)
        self.spiking_r = bool(spiking_r)

        # Token-shift mixing factors. RWKV-4 uses one per channel per
        # projection (R/K/V), in FP32 (per DECISIONS.md #2).
        # Init slightly different for r vs k/v: receptance favours the
        # current token more (smaller mu shift), keys/values blend.
        self.mu_r = nn.Parameter(_init_token_shift(hidden_size, layer_idx, n_layers))
        self.mu_k = nn.Parameter(_init_token_shift(hidden_size, layer_idx, n_layers))
        self.mu_v = nn.Parameter(_init_token_shift(hidden_size, layer_idx, n_layers) + 0.3)
        # Clamp init range to (0, 1) at definition time.
        with torch.no_grad():
            self.mu_v.clamp_(0.0, 1.0)

        # WKV parameters (FP32, small — one per channel).
        self.time_decay = nn.Parameter(_init_time_decay(hidden_size, layer_idx, n_layers))
        self.time_first = nn.Parameter(_init_time_first(hidden_size))

        # BitLinear projections (ternary weights).
        self.W_r = BitLinear(hidden_size, hidden_size, group_size=group_size, weight_dtype=weight_dtype)
        self.W_k = BitLinear(hidden_size, hidden_size, group_size=group_size, weight_dtype=weight_dtype)
        self.W_v = BitLinear(hidden_size, hidden_size, group_size=group_size, weight_dtype=weight_dtype)
        self.W_o = BitLinear(hidden_size, hidden_size, group_size=group_size, weight_dtype=weight_dtype)

        # FAI-LIF after each projection (observer-based; scale calibrated online).
        self.lif_r = (
            FAILIF(hidden_size, num_bits=bits, observer_momentum=observer_momentum)
            if spiking_r
            else nn.Identity()
        )
        self.lif_k = FAILIF(
            hidden_size, num_bits=bits, signed=signed_kv, observer_momentum=observer_momentum
        )
        self.lif_v = FAILIF(
            hidden_size, num_bits=bits, signed=signed_kv, observer_momentum=observer_momentum
        )
        self.lif_out = FAILIF(
            hidden_size, num_bits=bits, observer_momentum=observer_momentum
        )

        # Recurrent state buffers. Non-persistent: re-init on each
        # `reset_state` and lazily on the first forward.
        self.register_buffer("x_prev", torch.zeros(0), persistent=False)
        self.register_buffer("aa", torch.zeros(0), persistent=False)
        self.register_buffer("bb", torch.zeros(0), persistent=False)
        # ``pp`` is the running log-magnitude max. Init to a very large
        # negative value so the first step is dominated by the new token.
        self.register_buffer("pp", torch.zeros(0), persistent=False)

    # ----------------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------------

    def reset_state(
        self,
        batch_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        device = device or (self.x_prev.device if self.x_prev.numel() else self.time_first.device)
        if batch_shape is None:
            self.x_prev = torch.zeros(0, device=device, dtype=torch.float32)
            self.aa = torch.zeros(0, device=device, dtype=torch.float32)
            self.bb = torch.zeros(0, device=device, dtype=torch.float32)
            self.pp = torch.zeros(0, device=device, dtype=torch.float32)
        else:
            shape = (*batch_shape, self.hidden_size)
            self.x_prev = torch.zeros(shape, device=device, dtype=torch.float32)
            self.aa = torch.zeros(shape, device=device, dtype=torch.float32)
            self.bb = torch.zeros(shape, device=device, dtype=torch.float32)
            # -1e30 is the BlinkDL canonical "log of 0" sentinel.
            self.pp = torch.full(shape, -1e30, device=device, dtype=torch.float32)
        # Also reset child LIF states so they re-init on next call.
        for m in self.modules():
            if isinstance(m, FAILIF):
                m.reset_state()

    def _ensure_state(self, x: Tensor) -> None:
        target_shape = x.shape[:-1] + (self.hidden_size,)
        if (
            self.x_prev.numel() == 0
            or self.x_prev.shape != target_shape
            or self.x_prev.device != x.device
        ):
            self.x_prev = torch.zeros(target_shape, device=x.device, dtype=torch.float32)
            self.aa = torch.zeros(target_shape, device=x.device, dtype=torch.float32)
            self.bb = torch.zeros(target_shape, device=x.device, dtype=torch.float32)
            self.pp = torch.full(target_shape, -1e30, device=x.device, dtype=torch.float32)

    # ----------------------------------------------------------------------
    # Forward — per-token step
    # ----------------------------------------------------------------------

    def _step(self, x_t: Tensor) -> Tensor:
        """One token of time-mixing.

        ``x_t`` shape: ``(B, hidden_size)``. Returns same shape.
        """
        # Token shift in the activation dtype.
        x_prev = self.x_prev.to(x_t.dtype)
        mu_r = self.mu_r.to(x_t.dtype)
        mu_k = self.mu_k.to(x_t.dtype)
        mu_v = self.mu_v.to(x_t.dtype)
        xr = mu_r * x_t + (1.0 - mu_r) * x_prev
        xk = mu_k * x_t + (1.0 - mu_k) * x_prev
        xv = mu_v * x_t + (1.0 - mu_v) * x_prev

        # Project + FAI-LIF.
        r_raw = self.W_r(xr)
        r = self.lif_r(r_raw)  # FAI-LIF (or Identity if spiking_r=False)
        k = self.lif_k(self.W_k(xk))
        v = self.lif_v(self.W_v(xv))

        # WKV update in FP32. We cast k, v to FP32 here and keep them in
        # FP32 for the recurrent math — the spec mandates FP32 for the
        # recurrent state.
        k32 = k.to(torch.float32)
        v32 = v.to(torch.float32)

        time_first = self.time_first  # FP32 by construction
        time_decay = self.time_decay  # FP32

        # Output for the current token (bonus u applied to current k).
        ww = time_first + k32
        qq = torch.maximum(self.pp, ww)
        e1 = torch.exp(self.pp - qq)
        e2 = torch.exp(ww - qq)
        wkv = (e1 * self.aa + e2 * v32) / (e1 * self.bb + e2)

        # State update for the next token (decay applied to state).
        decay = torch.exp(time_decay)
        ww2 = self.pp - decay
        qq2 = torch.maximum(ww2, k32)
        e1u = torch.exp(ww2 - qq2)
        e2u = torch.exp(k32 - qq2)
        aa_new = e1u * self.aa + e2u * v32
        bb_new = e1u * self.bb + e2u
        pp_new = qq2

        # Output: sigmoid(r) gates wkv (cast back to x's dtype for the
        # downstream FAI-LIF/BitLinear), then FAI-LIF, then BitLinear_O.
        wkv_dt = wkv.to(x_t.dtype)
        gate = torch.sigmoid(r.to(x_t.dtype))
        gated = gate * wkv_dt
        out_spikes = self.lif_out(gated)
        out = self.W_o(out_spikes)

        # Update state. We DO NOT detach state here — within a forward
        # over a sequence, gradients flow through time. The model's outer
        # `forward` will detach at the end of the sequence to keep the
        # graph from growing across calls.
        self.x_prev = x_t.to(torch.float32)
        self.aa = aa_new
        self.bb = bb_new
        self.pp = pp_new
        return out

    # ----------------------------------------------------------------------
    # Forward — sequence or single token
    # ----------------------------------------------------------------------

    def _forward_sequence(self, x: Tensor) -> Tensor:
        """Fully-batched forward over a sequence ``(B, T, H)``.

        All three sub-pieces are batched across time:

          * Token-shift via a single ``cat`` (previous-token version of
            ``x[:, t]`` is ``x[:, t-1]`` for t≥1, ``x_prev`` for t=0).
          * BitLinear R/K/V/output and the per-tensor FAI-LIF on the
            full ``(B, T, H)`` tensor — the FAI-LIF fused CUDA kernel
            handles the recurrent membrane in one call (CPU falls back
            to per-step internally).
          * WKV recurrence via the fused CUDA kernel (CPU fallback in
            ``wkv_op._wkv_python``).
        """
        self._ensure_state(x[:, 0])

        # ---- Token shift (batched) ---------------------------------------
        x_prev = self.x_prev.to(x.dtype).unsqueeze(1)       # (B, 1, H)
        x_shifted = torch.cat([x_prev, x[:, :-1]], dim=1)   # (B, T, H)

        mu_r = self.mu_r.to(x.dtype)
        mu_k = self.mu_k.to(x.dtype)
        mu_v = self.mu_v.to(x.dtype)
        xr = mu_r * x + (1.0 - mu_r) * x_shifted
        xk = mu_k * x + (1.0 - mu_k) * x_shifted
        xv = mu_v * x + (1.0 - mu_v) * x_shifted

        # ---- Projections + FAI-LIF on (B, T, H) -------------------------
        r = self.lif_r(self.W_r(xr))          # FAI-LIF or Identity
        k = self.lif_k(self.W_k(xk))
        v = self.lif_v(self.W_v(xv))

        # ---- WKV recurrence (fused) -------------------------------------
        k32 = k.to(torch.float32)
        v32 = v.to(torch.float32)
        w_eff = torch.exp(self.time_decay)    # (H,) positive
        wkv, aa_new, bb_new, pp_new = _wkv_fused(
            w_eff, self.time_first, k32, v32, self.aa, self.bb, self.pp
        )
        # State stays attached within a forward (BPTT); caller detaches
        # between training windows.
        self.aa = aa_new
        self.bb = bb_new
        self.pp = pp_new

        # ---- Gate, output FAI-LIF, output projection --------------------
        gate = torch.sigmoid(r.to(x.dtype))   # (B, T, H)
        gated = gate * wkv.to(x.dtype)
        out_spikes = self.lif_out(gated)
        out = self.W_o(out_spikes)

        self.x_prev = x[:, -1].to(torch.float32)
        return out

    def forward(self, x: Tensor) -> Tensor:
        """Process a single token ``(B, H)`` or a sequence ``(B, T, H)``.

        State is updated in-place. Caller is responsible for
        :meth:`reset_state` between independent sequences and for
        :meth:`detach_state` at the end of a training-window if needed.
        """
        if x.dim() == 2:
            self._ensure_state(x)
            return self._step(x)
        if x.dim() == 3:
            return self._forward_sequence(x)
        raise ValueError(f"expected (B, H) or (B, T, H), got {x.shape}")

    def detach_state(self) -> None:
        """Detach recurrent state from the autograd graph in-place.

        Call between training windows of a long sequence to prevent the
        graph from growing unboundedly. Within a single window, leave
        state attached so BPTT works.
        """
        if self.x_prev.numel():
            self.x_prev = self.x_prev.detach()
            self.aa = self.aa.detach()
            self.bb = self.bb.detach()
            self.pp = self.pp.detach()

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, layer_idx={self.layer_idx}, "
            f"n_layers={self.n_layers}, spiking_r={self.spiking_r}"
        )
