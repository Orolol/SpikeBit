"""Micro-benchmark for the fused WKV op vs the Python reference.

Times forward + backward of:
  1. ``_wkv_python`` — the reference loop in pure PyTorch.
  2. ``wkv_forward`` (fused CUDA kernel) — the new fast path.

Both are run on the same shapes (default B=8, T=64, C=768 = SpikeBit
training config). Reports tokens/sec for each path and the speedup.

Usage:
    uv run python scripts/bench_wkv.py [--B 8 --T 64 --C 768 --warmup 5 --iters 20]
"""
from __future__ import annotations

import argparse
import time

import torch

from spiking_ternary_lm.layers.wkv_op import _wkv_python, wkv_forward


def _make_inputs(B: int, T: int, C: int, device: str = "cuda", seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w_eff = (
        torch.exp(torch.randn(C, generator=g) * 0.3 - 2.0)
        .to(device=device, dtype=torch.float32)
        .clone()
    )
    u = (torch.randn(C, generator=g) * 0.5).to(device=device, dtype=torch.float32).clone()
    k = (
        (torch.randn(B, T, C, generator=g) * 0.5)
        .to(device=device, dtype=torch.float32)
        .clone()
    )
    v = (
        (torch.randn(B, T, C, generator=g) * 0.5)
        .to(device=device, dtype=torch.float32)
        .clone()
    )
    aa0 = torch.zeros(B, C, device=device, dtype=torch.float32)
    bb0 = torch.zeros(B, C, device=device, dtype=torch.float32)
    pp0 = torch.full((B, C), -1e30, device=device, dtype=torch.float32)
    return w_eff, u, k, v, aa0, bb0, pp0


def _bench(fn, name: str, warmup: int, iters: int, B: int, T: int):
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tokens = B * T
    tps = tokens / dt
    print(f"  {name:<22s}  {dt*1000:8.3f} ms/iter   {tps:12.1f} tokens/s")
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--C", type=int, default=768)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    w0, u0, k0, v0, aa0, bb0, pp0 = _make_inputs(args.B, args.T, args.C)
    gout_template = torch.randn(args.B, args.T, args.C, device="cuda", dtype=torch.float32)

    # ---- Python reference (forward + backward) ---------------------------
    def step_py():
        w = w0.detach().clone().requires_grad_(True)
        u = u0.detach().clone().requires_grad_(True)
        k = k0.detach().clone().requires_grad_(True)
        v = v0.detach().clone().requires_grad_(True)
        a = aa0.detach().clone().requires_grad_(True)
        b = bb0.detach().clone().requires_grad_(True)
        p = pp0.detach().clone().requires_grad_(True)
        wkv, _, _, _ = _wkv_python(w, u, k, v, a, b, p)
        (wkv * gout_template).sum().backward()

    # ---- Fused CUDA op (forward + backward) ------------------------------
    def step_cuda():
        w = w0.detach().clone().requires_grad_(True)
        u = u0.detach().clone().requires_grad_(True)
        k = k0.detach().clone().requires_grad_(True)
        v = v0.detach().clone().requires_grad_(True)
        a = aa0.detach().clone().requires_grad_(True)
        b = bb0.detach().clone().requires_grad_(True)
        p = pp0.detach().clone().requires_grad_(True)
        wkv, _, _, _ = wkv_forward(w, u, k, v, a, b, p)
        (wkv * gout_template).sum().backward()

    print(f"WKV bench  B={args.B} T={args.T} C={args.C}  warmup={args.warmup} iters={args.iters}")
    print()
    print("WKV recurrence only (forward + backward)")
    dt_py = _bench(step_py, "python reference", args.warmup, args.iters, args.B, args.T)
    dt_cuda = _bench(step_cuda, "fused CUDA", args.warmup, args.iters, args.B, args.T)
    print()
    print(f"speedup  {dt_py / dt_cuda:5.1f}x")
    print()

    # ---- Full SpikingRWKVTimeMix layer (R/K/V projections + FAI-LIF +
    # WKV + output) — measures the end-to-end layer-level effect of the
    # fused kernel. Two variants: original Python loop vs the fused path.
    from spiking_ternary_lm.layers.spiking_rwkv import SpikingRWKVTimeMix

    block = SpikingRWKVTimeMix(
        hidden_size=args.C, layer_idx=0, n_layers=12, bits=4, group_size=128
    ).cuda()
    x = torch.randn(args.B, args.T, args.C, device="cuda", dtype=torch.float32)
    gout_layer = torch.randn_like(x)

    def step_layer_fused():
        block.reset_state()
        x_ = x.detach().clone().requires_grad_(True)
        y = block(x_)
        (y * gout_layer).sum().backward()

    # Monkey-patch the forward to force the per-step path for comparison.
    def _force_per_step(self, x):
        if x.dim() == 2:
            self._ensure_state(x)
            return self._step(x)
        B, T, H = x.shape
        self._ensure_state(x[:, 0])
        outs = [self._step(x[:, t]) for t in range(T)]
        return torch.stack(outs, dim=1)

    import types

    block_legacy = SpikingRWKVTimeMix(
        hidden_size=args.C, layer_idx=0, n_layers=12, bits=4, group_size=128
    ).cuda()
    block_legacy.load_state_dict(block.state_dict())
    block_legacy.forward = types.MethodType(_force_per_step, block_legacy)

    def step_layer_python():
        block_legacy.reset_state()
        x_ = x.detach().clone().requires_grad_(True)
        y = block_legacy(x_)
        (y * gout_layer).sum().backward()

    print("Full SpikingRWKVTimeMix layer (forward + backward)")
    dt_layer_py = _bench(step_layer_python, "python loop", args.warmup, args.iters, args.B, args.T)
    dt_layer_cu = _bench(step_layer_fused, "fused WKV path", args.warmup, args.iters, args.B, args.T)
    print()
    print(f"layer speedup  {dt_layer_py / dt_layer_cu:5.2f}x")


if __name__ == "__main__":
    main()
