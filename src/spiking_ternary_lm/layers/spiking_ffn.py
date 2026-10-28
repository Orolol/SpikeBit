"""Spiking-Ternary RWKV-4 channel-mixing block (FFN).

RWKV-4 channel-mixing in standard form:

    x_k = mu_k * x + (1 - mu_k) * x_prev
    x_r = mu_r * x + (1 - mu_r) * x_prev

    k = relu(W_k @ x_k) ** 2        # squared-ReLU on the up-projection
    v = W_v @ k                      # down-projection
    r = sigmoid(W_r @ x_r)           # gating signal
    output = r * v

Spiking-ternary substitutions (spec):

  * ``W_k`` (hidden → ffn_dim), ``W_v`` (ffn_dim → hidden), ``W_r``
    (hidden → hidden): all :class:`BitLinear` ternary.
  * ``squared_relu`` on K → replaced by :class:`FAILIF` (the integer
    spike count IS the up-projected nonlinearity).
  * Final ``v`` is followed by another :class:`FAILIF` so the value path
    is fully integer-spike before being multiplied by the gate.
  * ``sigmoid(r)`` kept as FP (cheap, no need to quantize a 0-1 gate).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.quantization.bitlinear import BitLinear


def _init_token_shift_ffn(hidden_size: int, layer_idx: int, n_layers: int) -> Tensor:
    """Same form as time-mix init but with a slightly different exponent."""
    ratio_1_to_almost0 = 1.0 - layer_idx / max(n_layers, 1)
    x = torch.arange(hidden_size, dtype=torch.float32) / hidden_size
    return torch.pow(x, ratio_1_to_almost0)


class SpikingRWKVChannelMix(nn.Module):
    """One RWKV-4 channel-mixing block in spiking-ternary form.

    Parameters
    ----------
    hidden_size:
        Channel dim of the residual stream.
    ffn_dim:
        Inner dimension of the FFN. Default ``4 * hidden_size`` per
        RWKV-4 convention.
    layer_idx, n_layers:
        Used by the init scheme.
    bits, group_size, weight_dtype, init_v_thr:
        Same semantics as in :class:`SpikingRWKVTimeMix`.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int | None = None,
        layer_idx: int = 0,
        n_layers: int = 1,
        bits: int = 4,
        group_size: int = 128,
        weight_dtype: torch.dtype = torch.float32,
        observer_momentum: float = 0.01,
        # See SpikingRWKVTimeMix: ignored by the observer-based FAILIF.
        init_v_thr: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else 4 * self.hidden_size
        self.layer_idx = int(layer_idx)
        self.n_layers = int(n_layers)

        if self.hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        if self.ffn_dim % group_size != 0:
            raise ValueError(
                f"ffn_dim ({self.ffn_dim}) must be divisible by group_size ({group_size})"
            )

        self.mu_k = nn.Parameter(_init_token_shift_ffn(hidden_size, layer_idx, n_layers))
        self.mu_r = nn.Parameter(_init_token_shift_ffn(hidden_size, layer_idx, n_layers))

        self.W_k = BitLinear(
            hidden_size, self.ffn_dim, group_size=group_size, weight_dtype=weight_dtype
        )
        self.W_v = BitLinear(
            self.ffn_dim, hidden_size, group_size=group_size, weight_dtype=weight_dtype
        )
        self.W_r = BitLinear(
            hidden_size, hidden_size, group_size=group_size, weight_dtype=weight_dtype
        )

        self.lif_k = FAILIF(
            self.ffn_dim, num_bits=bits, observer_momentum=observer_momentum
        )
        self.lif_v = FAILIF(
            hidden_size, num_bits=bits, observer_momentum=observer_momentum
        )

        # Token-shift state — one (B, hidden) buffer per block instance.
        # Non-persistent: re-init on each reset_state and lazily on first
        # forward (matches FAI-LIF semantics).
        self.register_buffer("x_prev", torch.zeros(0), persistent=False)

    # -- state management -------------------------------------------------------

    def reset_state(
        self,
        batch_shape: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        device = device or (self.x_prev.device if self.x_prev.numel() else self.mu_k.device)
        if batch_shape is None:
            self.x_prev = torch.zeros(0, device=device, dtype=torch.float32)
        else:
            self.x_prev = torch.zeros(
                (*batch_shape, self.hidden_size), device=device, dtype=torch.float32
            )
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

    # -- forward ---------------------------------------------------------------

    def _step(self, x_t: Tensor) -> Tensor:
        x_prev = self.x_prev.to(x_t.dtype)
        mu_k = self.mu_k.to(x_t.dtype)
        mu_r = self.mu_r.to(x_t.dtype)
        xk = mu_k * x_t + (1.0 - mu_k) * x_prev
        xr = mu_r * x_t + (1.0 - mu_r) * x_prev

        k = self.lif_k(self.W_k(xk))  # (B, ffn_dim) spike counts
        v = self.lif_v(self.W_v(k))   # (B, hidden) spike counts
        r = torch.sigmoid(self.W_r(xr))  # (B, hidden) FP gate in (0, 1)

        out = r * v

        self.x_prev = x_t.to(torch.float32)
        return out

    def _forward_sequence(self, x: Tensor) -> Tensor:
        """Fully-batched forward over a sequence ``(B, T, H)``.

        Token-shift is done with a single ``cat`` (the previous-token
        version of ``x[:, t]`` is ``x[:, t-1]`` for t≥1 and the persistent
        ``x_prev`` for t=0). BitLinear and FAI-LIF then run on the full
        ``(B, T, H)`` tensor — the FAI-LIF fused kernel handles the
        recurrent membrane in one CUDA call (CPU falls back to a per-step
        loop internally).
        """
        self._ensure_state(x[:, 0])
        x_prev = self.x_prev.to(x.dtype).unsqueeze(1)        # (B, 1, H)
        x_shifted = torch.cat([x_prev, x[:, :-1]], dim=1)    # (B, T, H)

        mu_k = self.mu_k.to(x.dtype)
        mu_r = self.mu_r.to(x.dtype)
        xk = mu_k * x + (1.0 - mu_k) * x_shifted
        xr = mu_r * x + (1.0 - mu_r) * x_shifted

        k = self.lif_k(self.W_k(xk))      # (B, T, ffn_dim) integer spikes
        v = self.lif_v(self.W_v(k))       # (B, T, hidden)  integer spikes
        r = torch.sigmoid(self.W_r(xr))   # (B, T, hidden)  FP gate

        out = r * v

        self.x_prev = x[:, -1].to(torch.float32)
        return out

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            self._ensure_state(x)
            return self._step(x)
        if x.dim() == 3:
            return self._forward_sequence(x)
        raise ValueError(f"expected (B, H) or (B, T, H), got {x.shape}")

    def detach_state(self) -> None:
        if self.x_prev.numel():
            self.x_prev = self.x_prev.detach()

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, ffn_dim={self.ffn_dim}, "
            f"layer_idx={self.layer_idx}, n_layers={self.n_layers}"
        )
