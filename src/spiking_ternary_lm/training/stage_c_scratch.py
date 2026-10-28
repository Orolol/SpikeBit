"""Stage C: from-scratch training of a randomly-initialised SpikingTernaryRWKV.

Same architecture as Stage A, but:
  * no teacher weights — random init on every parameter
  * all parameters trainable (embeddings + LM head included)
  * higher per-group LRs (~3-10× Stage A)
  * longer warmup, longer total schedule
  * optional **quantization warmup**: start with higher activation
    bit-width (e.g. ``A=8``) and anneal to the target (``A=4``) over the
    first 20 % of training. Hook: see ``apply_quant_warmup``.
  * optional **surrogate alpha schedule**: kept for API parity with the
    spec, but a no-op in our path because FAI-LIF uses pure STE through
    round+clip rather than a surrogate gradient.

**Status.** Code complete; not run. The full schedule (100 k+ steps) is
not feasible on a single 5090 with the explicit Python WKV loop —
revisit when either a custom WKV kernel or a parallel/chunked
implementation is in place (currently out of scope, see
``DECISIONS.md`` #9).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from spiking_ternary_lm.layers.spiking_ffn import SpikingRWKVChannelMix
from spiking_ternary_lm.layers.spiking_rwkv import SpikingRWKVTimeMix
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.quantization.bitlinear import BitLinear
from spiking_ternary_lm.training.data import build_train_eval_loaders, tokenize_tinystories
from spiking_ternary_lm.training.schedules import linear_warmup_cosine
from spiking_ternary_lm.training.stage_a_finetune import (
    _grad_norms,
    _SpikeStatsHook,
    _weight_dists,
    evaluate,
)
from spiking_ternary_lm.utils.logging import JsonlMetricsWriter, get_logger

LOG = get_logger("stage_c")


@dataclass
class StageCArgs:
    out_dir: str = "checkpoints/stage_c"
    metrics_path: str = "runs/stage_c/metrics.jsonl"
    tokenizer_id: str = "RWKV/rwkv-4-169m-pile"
    # Model
    config_preset: str = "tiny"  # "tiny" or "small_169m"
    # Data
    seq_len: int = 256
    batch_size: int = 8
    grad_accum: int = 4
    max_examples: int = 0  # 0 = full TinyStories
    eval_fraction: float = 0.01
    # Training
    total_steps: int = 100_000
    warmup_steps: int = 2_000
    min_lr_frac: float = 0.05
    grad_clip: float = 1.0
    # Per-group LRs (spec, higher than Stage A)
    lr_bitlinear: float = 3e-4
    lr_lif: float = 5e-4
    lr_rwkv: float = 1e-4
    lr_embed_head: float = 1e-4
    weight_decay: float = 0.01
    # Quantization warmup
    quant_warmup_steps: int = 0       # 0 = disabled
    quant_warmup_bits_start: int = 8  # start bit-width
    # Logging / checkpointing
    log_every: int = 100
    eval_every: int = 5_000
    ckpt_every: int = 10_000
    seed: int = 0
    device: str = "cuda"


def apply_quant_warmup(model: nn.Module, step: int, args: StageCArgs, target_bits: int) -> int:
    """Linearly anneal FAI-LIF ``bits`` from ``quant_warmup_bits_start`` to
    ``target_bits`` over ``quant_warmup_steps``. Returns the bit-width now in effect.
    """
    if args.quant_warmup_steps <= 0 or step >= args.quant_warmup_steps:
        return target_bits
    progress = step / args.quant_warmup_steps
    cur = round(
        args.quant_warmup_bits_start
        - (args.quant_warmup_bits_start - target_bits) * progress
    )
    for m in model.modules():
        if isinstance(m, FAILIF):
            if m.bits != cur:
                m.bits = cur
                m.n_max = 2**cur - 1 if not m.signed else 2 ** (cur - 1) - 1
                m.n_min = 0 if not m.signed else -(2 ** (cur - 1))
    return cur


def build_param_groups(model: SpikingTernaryRWKV, args: StageCArgs) -> list[dict]:
    bitlinear_params: list[nn.Parameter] = []
    lif_params: list[nn.Parameter] = []
    rwkv_recurrent_params: list[nn.Parameter] = []
    embed_head_params: list[nn.Parameter] = []
    norm_params: list[nn.Parameter] = []

    seen: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if name.startswith("embed.") or name.startswith("lm_head."):
            embed_head_params.append(p)
        elif "shadow_weight" in name:
            bitlinear_params.append(p)
        elif "beta_logit" in name or "gamma_logit" in name:
            lif_params.append(p)
        elif any(k in name for k in ("time_decay", "time_first", "mu_r", "mu_k", "mu_v")):
            rwkv_recurrent_params.append(p)
        elif "ln" in name:
            norm_params.append(p)
        else:
            bitlinear_params.append(p)
    return [
        {"params": bitlinear_params, "lr": args.lr_bitlinear, "name": "bitlinear"},
        {"params": lif_params, "lr": args.lr_lif, "name": "lif"},
        {"params": rwkv_recurrent_params, "lr": args.lr_rwkv, "name": "rwkv_recurrent"},
        {"params": embed_head_params, "lr": args.lr_embed_head, "name": "embed_head"},
        {"params": norm_params, "lr": args.lr_bitlinear, "name": "norm"},
    ]


def train(args: StageCArgs) -> int:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free_gb = torch.cuda.mem_get_info(device)[0] / 1e9
        LOG.info("GPU %s | %.2f GB free", torch.cuda.get_device_name(device), free_gb)

    # --- Build model from scratch (no teacher) -----------------------
    if args.config_preset == "tiny":
        cfg = SpikingTernaryRWKVConfig.tiny()
    elif args.config_preset == "small_169m":
        cfg = SpikingTernaryRWKVConfig.small_169m()
    else:
        raise ValueError(f"unknown preset {args.config_preset!r}")
    model = SpikingTernaryRWKV(cfg).to(device)
    LOG.info("model: %.1f M params", model.num_parameters() / 1e6)

    # --- Data --------------------------------------------------------
    LOG.info("tokenizing TinyStories...")
    corpus = tokenize_tinystories(
        tokenizer_id=args.tokenizer_id,
        max_examples=args.max_examples if args.max_examples > 0 else None,
    )
    train_loader, eval_loader = build_train_eval_loaders(
        corpus, seq_len=args.seq_len, batch_size=args.batch_size,
        eval_fraction=args.eval_fraction, shuffle_train=False,
    )

    # --- Optimizer ---------------------------------------------------
    groups = build_param_groups(model, args)
    for g in groups:
        LOG.info("  %-16s | %d params | lr %.2e",
                 g["name"], sum(p.numel() for p in g["params"]), g["lr"])
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    target_bits = cfg.bits

    hook = _SpikeStatsHook(model)
    writer = JsonlMetricsWriter(args.metrics_path)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    model.reset_state()
    t0 = time.time()
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    train_iter = iter(train_loader)

    while step < args.total_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(device); y = y.to(device)
        cur_bits = apply_quant_warmup(model, step, args, target_bits=target_bits)
        model.detach_state()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
        ) / args.grad_accum
        loss.backward()
        running += float(loss.detach()) * args.grad_accum
        micro += 1

        if micro % args.grad_accum == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            mult = linear_warmup_cosine(
                step, args.warmup_steps, args.total_steps, min_lr_frac=args.min_lr_frac
            )
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * mult
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            avg = running / args.grad_accum
            running = 0.0

            if step % args.log_every == 0:
                writer.log(step, loss=avg, lr_mult=mult, bits=cur_bits,
                           wall_s=time.time() - t0, spikes=dict(hook.stats),
                           weight_dist=_weight_dists(model),
                           grad_norms=_grad_norms(model))
                LOG.info(
                    "step %6d | loss %.4f | lr×%.2f | bits=%d | %.0fs",
                    step, avg, mult, cur_bits, time.time() - t0,
                )

            if step > 0 and step % args.eval_every == 0:
                el = evaluate(model, eval_loader, device, cfg.vocab_size)
                writer.log(step, eval_loss=el, eval_ppl=math.exp(el))
                LOG.info("step %6d | eval loss %.4f (ppl %.1f)",
                         step, el, math.exp(el))

            if step > 0 and step % args.ckpt_every == 0:
                p = Path(args.out_dir) / f"step_{step:07d}.pt"
                torch.save({"step": step, "model": model.state_dict(),
                            "config": asdict(cfg)}, p)
                LOG.info("saved %s", p)
            step += 1

    el = evaluate(model, eval_loader, device, cfg.vocab_size)
    LOG.info("FINAL | step %d eval %.4f (ppl %.1f)", step, el, math.exp(el))
    torch.save({"step": step, "model": model.state_dict(), "config": asdict(cfg)},
               Path(args.out_dir) / "final.pt")
    writer.close()
    hook.close()
    return 0


def _parse_args() -> StageCArgs:
    p = argparse.ArgumentParser(description=__doc__)
    defaults = StageCArgs()
    for name, default in defaults.__dict__.items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(default, bool):
            p.add_argument(flag, action="store_false" if default else "store_true")
        elif isinstance(default, int):
            p.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, default=default)
        else:
            p.add_argument(flag, type=str, default=default)
    ns = p.parse_args()
    return StageCArgs(**{k: getattr(ns, k) for k in defaults.__dict__})


def main() -> int:
    return train(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
