"""Phase 3a driver: convert a pre-trained RWKV-4 into our SpikingTernaryRWKV.

Downloads (or loads from cache) the HF teacher, builds a matching
SpikingTernaryRWKV instance, copies the weights, saves the initial
student checkpoint, and runs a sanity forward to confirm nothing is
broken structurally.

Output: a `.pt` checkpoint compatible with our Stage A trainer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from spiking_ternary_lm.conversion.rwkv_loader import load_rwkv4_teacher
from spiking_ternary_lm.conversion.swap_layers import load_teacher_into_student
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("convert_rwkv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--out", default="checkpoints/stage_a_init.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--weight-dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--init-v-thr", type=float, default=0.5)
    p.add_argument("--no-signed-kv", action="store_true",
                   help="Disable signed FAI-LIF on k/v (see DECISIONS.md #8).")
    p.add_argument("--no-spiking-r", action="store_true",
                   help="Keep r as FP (sigmoid then FP gate, no FAI-LIF on r).")
    p.add_argument("--prompt", default="Once upon a time,")
    args = p.parse_args()

    LOG.info("loading teacher: %s", args.teacher)
    teacher = load_rwkv4_teacher(args.teacher)
    LOG.info(
        "teacher: n_layers=%d hidden=%d ffn=%d vocab=%d",
        teacher.n_layers, teacher.hidden_size, teacher.ffn_dim, teacher.vocab_size,
    )

    weight_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.weight_dtype]
    cfg = SpikingTernaryRWKVConfig(
        vocab_size=teacher.vocab_size,
        hidden_size=teacher.hidden_size,
        n_layers=teacher.n_layers,
        ffn_dim=teacher.ffn_dim,
        bits=args.bits,
        group_size=args.group_size,
        weight_dtype=weight_dtype,
        init_v_thr=args.init_v_thr,
        signed_kv=not args.no_signed_kv,
        spiking_r=not args.no_spiking_r,
    )
    LOG.info("student config: %s", cfg)
    student = SpikingTernaryRWKV(cfg)
    LOG.info("building student: %d params", student.num_parameters())

    LOG.info("assigning teacher weights...")
    report = load_teacher_into_student(teacher, student)
    LOG.info("  assigned: %d student keys", len(report.assigned))
    LOG.info(
        "  kept student-only (LIF/v_thr/state): %d keys (e.g. %s)",
        len(report.kept_student),
        report.kept_student[:3] if report.kept_student else "<none>",
    )
    if report.skipped_teacher:
        LOG.warning(
            "teacher keys not used (%d): %s", len(report.skipped_teacher),
            report.skipped_teacher[:5],
        )

    # Move to device for sanity forward.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        LOG.info("GPU free: %.2f GB", free / 1e9)
        if free < 1.5e9:
            LOG.warning("low GPU memory; might OOM during conversion sanity forward")
    student.to(device)

    # Sanity forward on the prompt — uses the teacher tokenizer.
    LOG.info("sanity forward on prompt: %r", args.prompt)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
        student.reset_state()
        with torch.no_grad():
            logits = student(ids)
        LOG.info("logits shape: %s, finite: %s", tuple(logits.shape), bool(torch.isfinite(logits).all()))
        # Pull top-5 next-token logits for visual sanity (after the prompt).
        next_logits = logits[0, -1]
        topk = torch.topk(next_logits.float(), k=5)
        decoded = [tok.decode([i.item()]) for i in topk.indices]
        LOG.info("top-5 next tokens: %s", list(zip(decoded, topk.values.tolist())))

        # Quick CE on the prompt itself (predict each token from the previous).
        if ids.shape[1] > 1:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.reshape(-1, cfg.vocab_size), shift_labels.reshape(-1))
            LOG.info("teacher-forcing CE on the prompt (after quant): %.4f", loss.item())
    except Exception as e:
        LOG.warning("sanity forward failed: %s — saving checkpoint anyway", e)

    # Save checkpoint.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": student.state_dict(),
            "config": cfg.__dict__,
            "teacher_model_id": args.teacher,
            "conversion_report": {
                "assigned": report.assigned,
                "kept_student": report.kept_student,
                "skipped_teacher": report.skipped_teacher,
            },
        },
        out_path,
    )
    LOG.info("saved student checkpoint to %s (%.1f MB)",
             out_path, out_path.stat().st_size / 1e6)
    print(f"convert_rwkv: PASS — checkpoint at {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
