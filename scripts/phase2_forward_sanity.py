"""Phase 2 decision gate: SpikingTernaryRWKV forward + backward sanity.

The gate condition (from the project spec):

    "forward pass on random input runs without crash, output shapes correct"

We extend slightly so the script also exercises the backward pass on a
random cross-entropy target — that catches any autograd-graph or state
detachment bugs that pure forward would silently miss.

The script runs three configurations:
  1. tiny  — fast, CPU-friendly, asserts basic shapes
  2. small — Stage A target shape (RWKV-4-169M) — only built, not
              forward-passed unless GPU is free, to avoid hammering a
              shared GPU.
  3. tiny + sequence — verifies per-step and per-sequence calls agree.

Exits 0 on PASS, non-zero on FAIL.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F

from spiking_ternary_lm.layers.block import RWKVBlock
from spiking_ternary_lm.layers.spiking_ffn import SpikingRWKVChannelMix
from spiking_ternary_lm.layers.spiking_rwkv import SpikingRWKVTimeMix
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("phase2_sanity")


def _gpu_free_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("inf")
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1e9


def block_smoke(device: torch.device) -> None:
    """Each Phase-2 block runs forward+backward without crashing."""
    LOG.info("=== block smoke ===")
    H = 64
    for name, mk in [
        ("time", lambda: SpikingRWKVTimeMix(H, layer_idx=0, n_layers=2, group_size=16)),
        ("ffn", lambda: SpikingRWKVChannelMix(H, ffn_dim=256, layer_idx=0, n_layers=2, group_size=16)),
        ("block", lambda: RWKVBlock(H, ffn_dim=256, layer_idx=0, n_layers=2, group_size=16)),
    ]:
        m = mk().to(device)
        m.reset_state()
        x = torch.randn(3, 5, H, device=device, requires_grad=True)
        y = m(x)
        assert y.shape == (3, 5, H), f"{name} bad shape {y.shape}"
        loss = y.sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0, f"{name} grad failed"
        LOG.info("  %s: forward+backward OK", name)


def model_tiny(device: torch.device, batch: int = 4, seqlen: int = 16) -> None:
    LOG.info("=== model_tiny ===")
    cfg = SpikingTernaryRWKVConfig.tiny()
    model = SpikingTernaryRWKV(cfg).to(device)
    LOG.info("config: %s", cfg)
    n_params = model.num_parameters()
    LOG.info("trainable params: %d (%.2f K)", n_params, n_params / 1e3)
    model.reset_state()
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seqlen), device=device, dtype=torch.long)
    t0 = time.time()
    logits = model(input_ids)
    dt = time.time() - t0
    assert logits.shape == (batch, seqlen, cfg.vocab_size), f"bad logits shape {logits.shape}"
    assert torch.isfinite(logits).all(), "non-finite logits"
    LOG.info(
        "forward: B=%d T=%d wall %.3fs (%d tok, %.0f tok/s)",
        batch, seqlen, dt, batch * seqlen, batch * seqlen / max(dt, 1e-6),
    )

    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), input_ids.reshape(-1))
    LOG.info("CE loss on random init: %.4f (log(vocab)=%.4f for ref)",
             loss.item(), float(torch.log(torch.tensor(float(cfg.vocab_size)))))
    t0 = time.time()
    loss.backward()
    LOG.info("backward: wall %.3fs", time.time() - t0)
    grad_ok = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for _ in model.parameters())
    LOG.info("params with non-zero grad: %d / %d", grad_ok, total)
    assert grad_ok > total // 2, f"too few params got gradient ({grad_ok}/{total})"


def model_169m_build_only(device: torch.device) -> None:
    """Build the Stage A target shape and report parameter counts.

    We only run forward if there is enough free VRAM and CLI flag.
    """
    LOG.info("=== model_169m build ===")
    cfg = SpikingTernaryRWKVConfig.small_169m()
    model = SpikingTernaryRWKV(cfg).to(device)
    n_params = model.num_parameters()
    LOG.info("config: %s", cfg)
    LOG.info("trainable params: %d (%.2f M) — target ~169 M", n_params, n_params / 1e6)


def step_vs_sequence_equivalence(device: torch.device) -> None:
    """Calling forward token-by-token should equal forward on the sequence."""
    LOG.info("=== step ≡ sequence ===")
    H = 64
    block = SpikingRWKVTimeMix(H, layer_idx=0, n_layers=2, group_size=16).to(device)
    x = torch.randn(2, 4, H, device=device)
    block.reset_state()
    y_seq = block(x)
    block.reset_state()
    outs = [block(x[:, t]) for t in range(4)]
    y_step = torch.stack(outs, dim=1)
    err = (y_seq - y_step).abs().max().item()
    LOG.info("max abs diff: %.3e", err)
    assert err < 1e-4, f"step vs sequence diverged: {err}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-169m-forward", action="store_true",
                   help="Also run forward on the 169 M config (needs ~3 GB VRAM)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=16)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free_gb = _gpu_free_gb(device)
        LOG.info("GPU %s | %.2f GB free", torch.cuda.get_device_name(device), free_gb)
        if free_gb < 1.5:
            LOG.warning(
                "GPU has < 1.5 GB free. Per CLAUDE.md, wait for the dmtp agent "
                "to release VRAM before running Phase 2."
            )

    try:
        block_smoke(device)
        step_vs_sequence_equivalence(device)
        model_tiny(device, batch=args.batch, seqlen=args.seqlen)
        model_169m_build_only(device)
        if args.run_169m_forward:
            LOG.info("=== model_169m forward ===")
            cfg = SpikingTernaryRWKVConfig.small_169m()
            m = SpikingTernaryRWKV(cfg).to(device)
            m.reset_state()
            ids = torch.randint(0, cfg.vocab_size, (1, 8), device=device, dtype=torch.long)
            t0 = time.time()
            y = m(ids)
            LOG.info("169 M forward: %s in %.2fs", tuple(y.shape), time.time() - t0)
            assert torch.isfinite(y).all()
    except AssertionError as e:
        LOG.error("FAIL: %s", e)
        return 2
    except Exception as e:
        LOG.exception("CRASH: %s", e)
        return 3

    print("phase2_forward_sanity: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
