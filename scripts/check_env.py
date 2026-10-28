"""Phase 0 environment check.

Verifies that:
  - PyTorch imports.
  - CUDA is available.
  - The GPU is an RTX 5090 (SM 12.0 / Blackwell).
  - A trivial matmul on CUDA produces correct results.

Exits non-zero on any failure. Surface issues; do not silently fall back to CPU.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print(f"FAIL: torch not importable ({e}). Run `uv sync`.", file=sys.stderr)
        return 1

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() == False.", file=sys.stderr)
        return 2

    print(f"cuda runtime: {torch.version.cuda}")
    print(f"device count: {torch.cuda.device_count()}")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    print(f"device[{device}]: {name}, sm_{cap[0]}{cap[1]}")

    # Sanity matmul on GPU.
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        ref = (a.float() @ b.float()).bfloat16()
        max_err = (c - ref).abs().max().item()
        print(f"bf16 matmul max abs err vs fp32-ref-cast: {max_err:.3e}")
    except Exception as e:
        print(f"FAIL: CUDA matmul raised: {e}", file=sys.stderr)
        return 3

    # SM check: spec requires sm_120 for the 5090.
    if cap < (12, 0):
        print(
            f"WARN: device capability sm_{cap[0]}{cap[1]} < sm_120; "
            "this is not the 5090 path. Continuing.",
            file=sys.stderr,
        )
    elif cap >= (12, 0):
        print("ok: sm_120 detected (Blackwell).")

    # VRAM probe.
    free, total = torch.cuda.mem_get_info(device)
    print(f"vram: free {free / 1e9:.2f} GB / total {total / 1e9:.2f} GB")
    if free < 4e9:
        print(
            "WARN: less than 4 GB free; another process is using the GPU. "
            "Per CLAUDE.md, wait before launching training jobs.",
            file=sys.stderr,
        )

    print("env check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
