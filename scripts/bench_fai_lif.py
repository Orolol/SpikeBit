"""Micro-benchmark for the fused FAI-LIF op vs the Python per-step loop.

Times forward + backward of:
  1. Per-step Python ``FAILIF`` loop on ``(B, T, H)`` — the legacy path
     (calls ``lif(x[:, t])`` T times).
  2. ``FAILIF.forward(x)`` on a 3D tensor — the new fused CUDA path.

Both are run on the same shapes (default B=8, T=64, H=768 = SpikeBit
training config). Reports tokens/sec for each path and the speedup.

Usage:
    uv run python scripts/bench_fai_lif.py [--B 8 --T 64 --H 768 \
                                            --warmup 5 --iters 20]
"""
from __future__ import annotations

import argparse
import time

import torch

from spiking_ternary_lm.neurons.fai_lif import FAILIF


def _warmup_observer(lif: FAILIF, dim: int, n: int = 30) -> None:
    """Warm the observer so scale/zp are calibrated before benching."""
    lif.train()
    for _ in range(n):
        lif(torch.randn(4, dim, device="cuda"))
    # Reset membrane state so the bench starts from a clean (B, H) shape.
    lif.reset_state()


def _bench(fn, name: str, warmup: int, iters: int, B: int, T: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tps = (B * T) / dt
    print(f"  {name:<28s}  {dt*1000:8.3f} ms/iter   {tps:14.1f} tokens/s")
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--H", type=int, default=768)
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    lif_a = FAILIF(num_neurons=args.H, num_bits=args.num_bits).cuda()
    lif_b = FAILIF(num_neurons=args.H, num_bits=args.num_bits).cuda()
    _warmup_observer(lif_a, args.H, n=30)
    # Mirror the warmed observer into lif_b so both bench paths share state.
    lif_b.load_state_dict(lif_a.state_dict())
    lif_b.observer.running_min.copy_(lif_a.observer.running_min)
    lif_b.observer.running_max.copy_(lif_a.observer.running_max)
    lif_b.observer.num_updates.copy_(lif_a.observer.num_updates)
    lif_a.eval(); lif_b.eval()  # freeze observer for a fair fwd+bwd compare

    x_template = torch.randn(args.B, args.T, args.H, device="cuda", dtype=torch.float32)
    gout_template = torch.randn_like(x_template)

    # ---- Python per-step path -------------------------------------------
    def step_python():
        lif_a.reset_state()
        x = x_template.detach().clone().requires_grad_(True)
        outs = []
        for t in range(args.T):
            outs.append(lif_a(x[:, t]))
        y = torch.stack(outs, dim=1)
        (y * gout_template).sum().backward()

    # ---- Fused CUDA path -----------------------------------------------
    def step_cuda():
        lif_b.reset_state()
        x = x_template.detach().clone().requires_grad_(True)
        y = lif_b(x)
        (y * gout_template).sum().backward()

    print(
        f"FAI-LIF bench  B={args.B} T={args.T} H={args.H}  "
        f"num_bits={args.num_bits}  warmup={args.warmup} iters={args.iters}"
    )
    print()
    print("Forward + backward")
    dt_py = _bench(step_python, "python per-step loop", args.warmup, args.iters, args.B, args.T)
    dt_cu = _bench(step_cuda, "fused CUDA (3D path)", args.warmup, args.iters, args.B, args.T)
    print()
    print(f"speedup  {dt_py / dt_cu:5.2f}x")


if __name__ == "__main__":
    main()
