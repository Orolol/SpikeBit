"""Stage A: QAT-conversion fine-tune of the spiking-ternary student.

Loads a converted student checkpoint, fine-tunes on TinyStories with the
per-group optimizer config from the spec, and logs spike / quantization
stats every N steps. Saves checkpoints periodically.

Trade-off note: the spec calls for "no LIF reset between sequences,
single-epoch streaming." We approximate that with `shuffle_chunks=False`
and reset state ONLY at training start; the WKV state is detached
between training windows to bound autograd memory. With batch_size > 1
the state continuity is per-batch-element, not strictly contiguous
across batches — Stage C may need a more careful streaming dataloader.

Per DECISIONS.md #6/#7, ``init_v_thr`` is a placeholder; if Stage A
plateaus at ~94 % retention, we will replace it with NSLLM-style
moving-average MinMax calibration before Stage C. Hooks for that are
not yet wired in.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from spiking_ternary_lm.conversion.rwkv_loader import load_rwkv4_teacher
from spiking_ternary_lm.conversion.swap_layers import load_teacher_into_student
from spiking_ternary_lm.eval.perplexity import protocol_perplexity
from spiking_ternary_lm.layers.spiking_ffn import SpikingRWKVChannelMix
from spiking_ternary_lm.layers.spiking_rwkv import SpikingRWKVTimeMix
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.fai_lif import FAILIF, calibrate_observers
from spiking_ternary_lm.quantization.bitlinear import BitLinear, set_quant_mix
from spiking_ternary_lm.training.data import build_train_eval_loaders, tokenize_tinystories
from spiking_ternary_lm.training.schedules import linear_warmup_cosine
from spiking_ternary_lm.utils.logging import JsonlMetricsWriter, get_logger

LOG = get_logger("stage_a")


@dataclass
class StageAArgs:
    init_checkpoint: str = "checkpoints/stage_a_init.pt"
    out_dir: str = "checkpoints/stage_a"
    metrics_path: str = "runs/stage_a/metrics.jsonl"
    tokenizer_id: str = "RWKV/rwkv-4-169m-pile"
    # Data
    seq_len: int = 256
    batch_size: int = 8
    grad_accum: int = 4  # effective batch 32
    max_examples: int | None = None  # None = full TinyStories
    eval_fraction: float = 0.01
    # Training
    total_steps: int = 10000
    warmup_steps: int = 500
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    # Per-group LRs (spec)
    lr_bitlinear: float = 1e-4
    lr_lif: float = 5e-5
    lr_rwkv: float = 1e-5
    lr_embed_head: float = 0.0    # 0 = frozen, per spec
    weight_decay: float = 0.0
    # Logging / checkpointing
    log_every: int = 50
    eval_every: int = 1000
    # Cap on eval batches for the periodic in-training eval (the eval
    # split is ~4.7 M tokens; a full pass costs minutes). 0 = full split.
    # The final eval after training always uses the full split.
    eval_max_batches: int = 200
    ckpt_every: int = 2000
    seed: int = 0
    device: str = "cuda"
    # Override the checkpoint's stored ``weight_dtype`` for the BitLinear
    # shadow weights. "" means "use the dtype baked into the checkpoint
    # config". Useful for the FP32/BF16 smoke comparison.
    weight_dtype: str = ""
    # Mixed-precision policy for the forward + loss. "none" runs the
    # forward in the param dtype (FP32). "bf16" wraps the forward in
    # ``torch.amp.autocast(dtype=bfloat16)`` so matmuls/elementwise ops
    # take the tensor-core path while parameters, gradients, and
    # optimizer state stay in FP32 (see DECISIONS.md #13). The WKV CUDA
    # kernel is force-cast back to FP32 inside its Python wrapper.
    mixed_precision: str = "bf16"
    # Override the per-FAILIF observer EMA momentum. ``-1`` keeps the
    # value baked into the checkpoint's cfg. Useful when the fused
    # sequence-mode FAI-LIF kernel does 1 EMA update per (B, T, H) call
    # vs T updates in the per-step path: to compensate, set
    # ``observer_momentum ≈ 1 - (1 - m_step)^T`` (e.g. 0.47 for T=64,
    # m_step=0.01).
    observer_momentum: float = -1.0
    # Override ``FAILIF.seq_chunk_size`` on every LIF: the fused kernel
    # freezes (scale, zp) for one call, so ``1`` reproduces the per-step
    # Python reference (best PPL, ~2x slower), ``0`` calls once per
    # sequence (fastest, stalest grid). ``-1`` keeps the module default.
    lif_seq_chunk: int = -1
    # Reset the recurrent state (WKV aa/bb/pp, LIF membranes, token-shift)
    # every N optimizer steps. 0 = never (DEC-2 single-epoch streaming:
    # state carried across sequences, reset once at the start). 1 = every
    # sequence is independent. Diagnostic for state-drift instabilities.
    reset_state_every: int = 0
    # Freeze every FAI-LIF observer (quantization grid) from this optimizer
    # step on. 0 = never (observers track activations for the whole run).
    freeze_observers_at: int = 0
    # Override the checkpoint's activation quantizer: "" keeps the stored
    # config, "minmax" or "lsq" forces it (a MinMax checkpoint loads into an
    # LSQ model; the LSQ grids are initialised from the first batch).
    act_quant: str = ""
    # Override the observer scope: "" keeps the checkpoint's, "tensor" or
    # "channel" forces it (a per-tensor checkpoint loads into a per-channel
    # model; the per-channel grids initialise from the first batch).
    observer_scope: str = ""
    lsq_init_steps: int = 1
    # LR of the LSQ grid parameters (log-scale and zero-point). Adam's step is
    # ~lr per step in log-scale units, i.e. ~1 % of scale per step at 1e-2:
    # fast enough to track the x2-6 range drift seen in the MinMax runs.
    lr_lsq: float = 1e-2
    # Lay the training stream out so that batch row b at step n+1 continues
    # row b at step n (DEC-2 streaming as intended). Off = the historical
    # layout (consecutive chunks across the batch; the carried state then
    # comes from an unrelated chunk). Kept off by default for comparability
    # with the 2026-10-28 reference runs.
    contiguous_rows: bool = False
    # --- Stage A ladder (DECISIONS #15) ---------------------------------
    # Build the student from this HF teacher (conversion in memory) instead
    # of loading ``init_checkpoint``. "" = use ``init_checkpoint``.
    teacher_id: str = ""
    # Override the BitLinear weights: "" keeps the config's, else "ternary"
    # or "none" (FP shadow weights).
    weight_quant: str = ""
    # Keep the teacher's channel-mix relu² in front of the FFN FAI-LIF.
    ffn_relu_sq: bool = False
    # Logit distillation from a student-format checkpoint (rung 1 seed 0):
    # loss = CE + distill_weight * T^2 * KL(teacher || student) at T =
    # distill_temperature. The logged "loss" stays the CE.
    distill_checkpoint: str = ""
    distill_weight: float = 1.0
    distill_temperature: float = 1.0
    # Final eval on the DECISIONS #15 protocol (eval tail, seq 128, 1 M
    # tokens, stateless) instead of the full split at the training seq_len.
    protocol_eval: bool = False
    # Progressive ternarization (DECISIONS #15 plan B (a)): the weight blend
    # ramps linearly from FP to ternary over this fraction of total_steps.
    # 0 = ternary from the first step.
    ternary_ramp_frac: float = 0.0
    # Activation-quantization overrides for rung 3 (see the config fields).
    # "" / 0 / False keep the config's value.
    act_grid: str = ""
    act_output: str = ""
    act_ste: str = ""
    lif_stateless: bool = False
    lif_membrane_clamp: float = 0.0
    lif_membrane_grad: str = ""
    bits: int = 0
    bits_ffn_hidden: int = 0
    # Calibrate every MinMax observer on this many training batches (FP
    # pass, global min/max) before the first step. 0 = first-batch init.
    calibrate_batches: int = 0


def make_param_groups(model: SpikingTernaryRWKV, args: StageAArgs) -> list[dict]:
    """Per-spec parameter groups."""
    bitlinear_params: list[nn.Parameter] = []
    lif_params: list[nn.Parameter] = []
    rwkv_recurrent_params: list[nn.Parameter] = []
    embed_head_params: list[nn.Parameter] = []
    norm_params: list[nn.Parameter] = []
    lsq_params: list[nn.Parameter] = []

    seen: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if name.startswith("embed.") or name.startswith("lm_head."):
            embed_head_params.append(p)
        elif "shadow_weight" in name:
            bitlinear_params.append(p)
        elif "lsq_" in name:
            lsq_params.append(p)
        elif "beta_logit" in name or "gamma_logit" in name:
            lif_params.append(p)
        elif any(k in name for k in ("time_decay", "time_first", "mu_r", "mu_k", "mu_v")):
            rwkv_recurrent_params.append(p)
        elif "ln" in name:
            norm_params.append(p)
        else:
            # Anything else: treat as bitlinear-rate (rare — like BitLinear.bias).
            bitlinear_params.append(p)

    if args.lr_embed_head == 0.0:
        for p in embed_head_params:
            p.requires_grad_(False)

    groups = [
        {"params": bitlinear_params, "lr": args.lr_bitlinear, "name": "bitlinear"},
        {"params": lif_params, "lr": args.lr_lif, "name": "lif"},
        {"params": rwkv_recurrent_params, "lr": args.lr_rwkv, "name": "rwkv_recurrent"},
        {"params": norm_params, "lr": args.lr_bitlinear, "name": "norm"},
    ]
    if lsq_params:
        groups.append({"params": lsq_params, "lr": args.lr_lsq, "name": "lsq"})
    if args.lr_embed_head > 0.0:
        groups.append({"params": embed_head_params, "lr": args.lr_embed_head, "name": "embed_head"})
    return groups


# ------------------------------------------------------------------
# Spike-rate hooks for logging
# ------------------------------------------------------------------


class _SpikeStatsHook:
    """Attach forward-hooks to every FAI-LIF to track per-step fire rate."""

    def __init__(self, model: nn.Module):
        self.stats: dict[str, dict[str, float]] = {}
        # Each hook costs 3 device->host syncs; 60 LIFs x 3 per step
        # starves the GPU. The trainer flips this on for logged steps only.
        self.enabled = True
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, m in model.named_modules():
            if isinstance(m, FAILIF):
                self.handles.append(m.register_forward_hook(self._make_hook(name, m)))

    def _make_hook(self, name: str, lif: FAILIF):
        n_max = lif.n_max

        def hook(_module, _inputs, output):
            if not self.enabled:
                return
            with torch.no_grad():
                out = output.detach()
                if getattr(lif, "last_spike", None) is not None:
                    out = lif.last_spike  # LSQ mode: output is dequantized
                self.stats[name] = {
                    "fire_norm": float((out.float() / max(n_max, 1)).mean()),
                    "saturated": float((out >= n_max).float().mean()),
                    "dead": float((out.float().mean(dim=0) == 0).float().mean()),
                }
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _weight_dists(model: nn.Module) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, m in model.named_modules():
        if isinstance(m, BitLinear):
            out[name] = m.weight_distribution()
    return out


def _grad_norms(model: nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        out[name] = float(p.grad.detach().float().norm())
    return out


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------


def train(args: StageAArgs) -> int:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free_gb = torch.cuda.mem_get_info(device)[0] / 1e9
        LOG.info("GPU %s | %.2f GB free", torch.cuda.get_device_name(device), free_gb)
        if free_gb < 4.0:
            LOG.warning("low GPU memory (< 4 GB); Stage A may OOM")

    # --- Load init checkpoint (or convert the teacher in memory) ------
    teacher_info = None
    if args.teacher_id:
        LOG.info("building the student from teacher %s", args.teacher_id)
        teacher_info = load_rwkv4_teacher(args.teacher_id)
        cfg = SpikingTernaryRWKVConfig(
            vocab_size=teacher_info.vocab_size, hidden_size=teacher_info.hidden_size,
            n_layers=teacher_info.n_layers, ffn_dim=teacher_info.ffn_dim,
        )
    else:
        LOG.info("loading init checkpoint: %s", args.init_checkpoint)
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        cfg = SpikingTernaryRWKVConfig(**ckpt["config"])
    if args.weight_dtype:
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        if args.weight_dtype not in dtype_map:
            raise ValueError(f"--weight-dtype must be one of {list(dtype_map)}; got {args.weight_dtype}")
        new_dtype = dtype_map[args.weight_dtype]
        if new_dtype is not cfg.weight_dtype:
            LOG.info("overriding weight_dtype: %s -> %s", cfg.weight_dtype, new_dtype)
            cfg.weight_dtype = new_dtype
    if args.act_quant:
        if args.act_quant not in ("minmax", "lsq", "none"):
            raise ValueError(f"--act-quant must be minmax, lsq or none; got {args.act_quant}")
        LOG.info("overriding act_quant: %s -> %s", getattr(cfg, "act_quant", "minmax"), args.act_quant)
        cfg.act_quant = args.act_quant
        cfg.lsq_init_steps = int(args.lsq_init_steps)
    if args.observer_scope:
        if args.observer_scope not in ("tensor", "channel"):
            raise ValueError(f"--observer-scope must be tensor or channel; got {args.observer_scope}")
        LOG.info("overriding observer_scope: %s -> %s", getattr(cfg, "observer_scope", "tensor"), args.observer_scope)
        cfg.observer_scope = args.observer_scope
    if args.observer_momentum > 0:
        LOG.info(
            "overriding observer_momentum: %s -> %s",
            getattr(cfg, "observer_momentum", 0.01), args.observer_momentum,
        )
        cfg.observer_momentum = float(args.observer_momentum)
    if args.weight_quant:
        if args.weight_quant not in ("ternary", "none"):
            raise ValueError(f"--weight-quant must be ternary or none; got {args.weight_quant}")
        LOG.info("overriding weight_quant: %s -> %s", getattr(cfg, "weight_quant", "ternary"), args.weight_quant)
        cfg.weight_quant = args.weight_quant
    if args.ffn_relu_sq:
        LOG.info("ffn_relu_sq: the channel-mix keeps the teacher's relu²")
        cfg.ffn_relu_sq = True
    for name in ("act_grid", "act_output", "act_ste", "lif_stateless", "lif_membrane_clamp",
                 "lif_membrane_grad", "bits", "bits_ffn_hidden"):
        value = getattr(args, name)
        if value:
            LOG.info("overriding %s: %s -> %s", name, getattr(cfg, name), value)
            setattr(cfg, name, value)
    model = SpikingTernaryRWKV(cfg)
    if teacher_info is not None:
        report = load_teacher_into_student(teacher_info, model, strict=True)
        LOG.info("assigned %d teacher tensors", len(report.assigned))
        del teacher_info
    else:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            LOG.warning("state_dict gaps: missing=%d unexpected=%d", len(missing), len(unexpected))
    if args.lif_seq_chunk >= 0:
        n_lif = 0
        for m in model.modules():
            if isinstance(m, FAILIF):
                m.seq_chunk_size = int(args.lif_seq_chunk)
                n_lif += 1
        LOG.info("lif_seq_chunk=%d applied to %d FAILIF modules", args.lif_seq_chunk, n_lif)
    model.to(device)
    LOG.info("model: %d trainable params (%.1f M)", model.num_parameters(), model.num_parameters() / 1e6)

    distill_model = None
    if args.distill_checkpoint:
        LOG.info("distillation teacher: %s (weight %.2f, T %.2f)",
                 args.distill_checkpoint, args.distill_weight, args.distill_temperature)
        dckpt = torch.load(args.distill_checkpoint, map_location="cpu", weights_only=False)
        distill_model = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(**dckpt["config"]))
        distill_model.load_state_dict(dckpt["model"])
        distill_model.eval().requires_grad_(False).to(device)
        del dckpt

    # --- Data --------------------------------------------------------
    LOG.info("tokenizing TinyStories (cached after first run)...")
    corpus = tokenize_tinystories(
        tokenizer_id=args.tokenizer_id, max_examples=args.max_examples
    )
    LOG.info("corpus tokens: %.2f M", len(corpus.ids) / 1e6)
    train_loader, eval_loader = build_train_eval_loaders(
        corpus, seq_len=args.seq_len, batch_size=args.batch_size,
        eval_fraction=args.eval_fraction, shuffle_train=False,
        contiguous_rows=args.contiguous_rows, seed=args.seed,
    )

    # --- Optimizer ---------------------------------------------------
    groups = make_param_groups(model, args)
    for g in groups:
        LOG.info("  group %-16s | %d params | lr %.2e",
                 g["name"], sum(p.numel() for p in g["params"]), g["lr"])
    optimizer = torch.optim.AdamW(
        groups, weight_decay=args.weight_decay, fused=(device.type == "cuda")
    )

    # --- Mixed-precision policy (DECISIONS.md #13) -------------------
    mp_map = {"none": None, "bf16": torch.bfloat16}
    if args.mixed_precision not in mp_map:
        raise ValueError(
            f"--mixed-precision must be one of {list(mp_map)}; got {args.mixed_precision}"
        )
    autocast_dtype = mp_map[args.mixed_precision]
    use_autocast = autocast_dtype is not None and device.type == "cuda"
    LOG.info(
        "mixed_precision=%s (autocast %s)",
        args.mixed_precision,
        f"enabled[{autocast_dtype}]" if use_autocast else "disabled",
    )

    if args.calibrate_batches > 0:
        calib = []
        for i, (cx, _cy) in enumerate(train_loader):
            if i >= args.calibrate_batches:
                break
            calib.append(cx.to(device))
        n_obs = calibrate_observers(model, calib)
        LOG.info("calibrated %d observers on %d batches", n_obs, len(calib))
        del calib

    hook = _SpikeStatsHook(model)

    # --- Loop --------------------------------------------------------
    model.train()
    model.reset_state()  # spec: reset only at start
    if distill_model is not None:
        distill_model.reset_state()
    writer = JsonlMetricsWriter(args.metrics_path)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    running_loss = 0.0
    running_kl = 0.0

    train_iter = iter(train_loader)
    while step < args.total_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.detach_state()  # bound BPTT memory between training windows
        if args.reset_state_every > 0 and micro % (args.reset_state_every * args.grad_accum) == 0:
            model.reset_state()
            if distill_model is not None:
                distill_model.reset_state()
        if args.freeze_observers_at > 0 and step == args.freeze_observers_at and micro % args.grad_accum == 0:
            n_frozen = 0
            for m in model.modules():
                if isinstance(m, FAILIF) and not m.observer_frozen:
                    m.observer_frozen = True
                    n_frozen += 1
            LOG.info("step %d: froze %d FAI-LIF observers", step, n_frozen)
        hook.enabled = step % args.log_every == 0
        if args.ternary_ramp_frac > 0:
            set_quant_mix(model, min(1.0, step / (args.ternary_ramp_frac * args.total_steps)))
        # Autocast wraps ONLY the forward + loss. backward inherits the
        # dtype graph automatically and gradients accumulate into FP32
        # leaf .grad buffers; the optimizer.step() runs outside the
        # autocast region so the optim state stays FP32.
        with torch.amp.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=use_autocast
        ):
            logits = model(x)
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss = ce
            if distill_model is not None:
                with torch.no_grad():
                    t_logits = distill_model(x)
                temp = args.distill_temperature
                kl = F.kl_div(
                    F.log_softmax(logits.float().reshape(-1, cfg.vocab_size) / temp, dim=-1),
                    F.log_softmax(t_logits.float().reshape(-1, cfg.vocab_size) / temp, dim=-1),
                    log_target=True, reduction="batchmean",
                ) * temp * temp
                loss = ce + args.distill_weight * kl
                running_kl += float(kl.detach())
            loss = loss / args.grad_accum
        loss.backward()
        running_loss += float(ce.detach())
        micro += 1

        if micro % args.grad_accum == 0:
            total_grad_norm = float("nan")
            if args.grad_clip > 0:
                total_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                )
            # Per-parameter grad norms must be read before zero_grad (they were
            # logged after it since May and always came out empty).
            grad_norms = _grad_norms(model) if step % args.log_every == 0 else {}
            # LR schedule.
            mult = linear_warmup_cosine(
                step, args.warmup_steps, args.total_steps, min_lr_frac=args.min_lr_frac
            )
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * mult
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            avg_loss = running_loss / args.grad_accum
            avg_kl = running_kl / args.grad_accum
            running_loss = 0.0
            running_kl = 0.0

            if step % args.log_every == 0:
                metrics: dict[str, object] = {
                    "step": step,
                    "loss": avg_loss,
                    "lr_mult": mult,
                    "wall_s": time.time() - t0,
                    "spikes": dict(hook.stats),
                    "weight_dist": _weight_dists(model),
                    "grad_norms": grad_norms,
                    "grad_norm_total": total_grad_norm,
                }
                if distill_model is not None:
                    metrics["kl"] = avg_kl
                writer.log(step, **{k: v for k, v in metrics.items() if k != "step"})
                # Pick out one representative spike rate per layer-type.
                samples = list(hook.stats.items())[:4]
                spike_str = " ".join(f"{n.split('.')[-1]}={s['fire_norm']:.3f}" for n, s in samples)
                LOG.info(
                    "step %5d | loss %.4f | lr×%.2f | %s | %.1fs",
                    step, avg_loss, mult, spike_str, time.time() - t0,
                )

            if step > 0 and step % args.eval_every == 0:
                eval_loss = evaluate(
                    model, eval_loader, device, cfg.vocab_size,
                    max_batches=args.eval_max_batches,
                )
                writer.log(step, eval_loss=eval_loss, eval_ppl=math.exp(eval_loss))
                LOG.info("step %5d | eval loss %.4f (ppl %.1f)", step, eval_loss, math.exp(eval_loss))
                if distill_model is not None:
                    distill_model.reset_state()  # the student resumes from a fresh state too

            if step > 0 and step % args.ckpt_every == 0:
                ckpt_path = Path(args.out_dir) / f"step_{step:06d}.pt"
                torch.save({"step": step, "model": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
                LOG.info("saved %s", ckpt_path)

            step += 1

    # Final ckpt + eval.
    if args.protocol_eval:
        res = protocol_perplexity(model, corpus, device)
        eval_loss = res["loss"]
        writer.log(step, protocol_loss=eval_loss, protocol_ppl=res["perplexity"],
                   protocol_tokens=res["n_tokens"])
        LOG.info("PROTOCOL | %d tokens, seq 128, stateless: ppl %.3f",
                 res["n_tokens"], res["perplexity"])
    else:
        eval_loss = evaluate(model, eval_loader, device, cfg.vocab_size)
    LOG.info("FINAL | step %d eval loss %.4f (ppl %.1f)",
             step, eval_loss, math.exp(eval_loss))
    final_path = Path(args.out_dir) / "final.pt"
    torch.save({"step": step, "model": model.state_dict(), "config": asdict(cfg)}, final_path)
    LOG.info("saved final %s", final_path)
    writer.close()
    hook.close()
    print(f"stage_a: DONE step={step} eval_loss={eval_loss:.4f}")
    return 0


@torch.no_grad()
def evaluate(
    model: SpikingTernaryRWKV, loader: Iterable, device: torch.device, vocab_size: int,
    max_batches: int = 0,
) -> float:
    model.eval()
    total = 0
    total_loss = 0.0
    # Stateless per window, like scripts/eval_teacher.py (the teacher sees
    # each window from a fresh state), and no leakage of the training state
    # into the eval nor of the eval state back into training (review
    # 2026-10-28). Training resumes from a fresh state after an eval.
    model.reset_state()
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x = x.to(device); y = y.to(device)
        model.reset_state()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum"
        )
        total_loss += float(loss)
        total += y.numel()
    model.reset_state()
    model.train()
    return total_loss / max(total, 1)


def _parse_args() -> StageAArgs:
    """Minimal argparse: one --flag per dataclass field, types inferred."""
    p = argparse.ArgumentParser(description=__doc__)
    defaults = StageAArgs()
    for name, default in defaults.__dict__.items():
        flag = f"--{name.replace('_', '-')}"
        if name == "max_examples":
            p.add_argument(flag, type=int, default=default,
                           help="Limit TinyStories examples (None = full).")
        elif isinstance(default, bool):
            p.add_argument(flag, action="store_false" if default else "store_true")
        elif isinstance(default, int):
            p.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, default=default)
        else:
            p.add_argument(flag, type=str, default=default)
    ns = p.parse_args()
    return StageAArgs(**{k: getattr(ns, k) for k in defaults.__dict__})


def main() -> int:
    return train(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
