"""DECISIONS #15 rung 0: the converted student with quantization off is the teacher.

Builds the student from the HF RWKV-4 teacher with FP weights, identity
FAI-LIF and the teacher's channel-mix ``relu²`` kept, then runs both models
on the same eval windows (protocol of DECISIONS #15: last 1 % of the packed
TinyStories corpus, seq 128, stateless per window, first 1 M tokens).

Pre-registered gate:
  * logits allclose (rtol = atol = 1e-3, FP32) on the first ``--check-batches``
    batches;
  * student PPL equal to the teacher's within 0.1 %.

The teacher PPL printed here is also the re-measurement of the 13.34
reference on the DECISIONS #15 protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from spiking_ternary_lm.conversion.rwkv_loader import load_rwkv4_teacher
from spiking_ternary_lm.conversion.swap_layers import load_teacher_into_student
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.training.data import (
    PackedTokenDataset,
    TokenizedCorpus,
    tokenize_tinystories,
)
from spiking_ternary_lm.utils.logging import get_logger

LOG = get_logger("rung0")

LOGIT_RTOL = 1e-3
LOGIT_ATOL = 1e-3
PPL_REL_TOL = 1e-3


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="RWKV/rwkv-4-169m-pile")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=1_000_000)
    p.add_argument("--check-batches", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/rung0_equivalence/summary.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    LOG.info("loading teacher %s ...", args.model_id)
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, dtype=torch.float32
    ).eval().to(device)
    info = load_rwkv4_teacher(args.model_id)

    cfg = SpikingTernaryRWKVConfig(
        vocab_size=info.vocab_size, hidden_size=info.hidden_size,
        n_layers=info.n_layers, ffn_dim=info.ffn_dim,
        act_quant="none", weight_quant="none", ffn_relu_sq=True,
    )
    student = SpikingTernaryRWKV(cfg)
    report = load_teacher_into_student(info, student, strict=True)
    LOG.info("assigned %d teacher tensors, %d teacher keys unused",
             len(report.assigned), len(report.skipped_teacher))
    if report.skipped_teacher:
        LOG.error("unused teacher keys: %s", report.skipped_teacher)
        return 1
    student.eval().to(device)

    corpus = tokenize_tinystories(tokenizer_id=args.model_id)
    n = len(corpus.ids)
    eval_corpus = TokenizedCorpus(
        ids=corpus.ids[int(n * 0.99):], vocab_size=corpus.vocab_size,
        eos_token_id=corpus.eos_token_id,
    )
    ds = PackedTokenDataset(eval_corpus, seq_len=args.seq_len, shuffle_chunks=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    vocab = info.vocab_size
    loss_t = loss_s = 0.0
    n_tokens = 0
    max_abs_diff = 0.0
    logits_close = True
    for i, (x, y) in enumerate(loader):
        x = x.to(device); y = y.to(device)
        logits_t = teacher(input_ids=x, use_cache=False).logits.float()
        student.reset_state()  # stateless per window, like the teacher call
        logits_s = student(x).float()
        if i < args.check_batches:
            max_abs_diff = max(max_abs_diff, float((logits_s - logits_t).abs().max()))
            logits_close &= bool(torch.allclose(logits_s, logits_t, rtol=LOGIT_RTOL, atol=LOGIT_ATOL))
        loss_t += float(F.cross_entropy(logits_t.reshape(-1, vocab), y.reshape(-1), reduction="sum"))
        loss_s += float(F.cross_entropy(logits_s.reshape(-1, vocab), y.reshape(-1), reduction="sum"))
        n_tokens += y.numel()
        if n_tokens >= args.max_tokens:
            break

    ppl_t = math.exp(loss_t / n_tokens)
    ppl_s = math.exp(loss_s / n_tokens)
    ppl_rel = abs(ppl_s / ppl_t - 1.0)
    passed = logits_close and ppl_rel <= PPL_REL_TOL
    summary = {
        "teacher_ppl": ppl_t, "student_ppl": ppl_s, "ppl_rel_diff": ppl_rel,
        "n_tokens": n_tokens, "seq_len": args.seq_len,
        "check_batches": args.check_batches, "max_abs_logit_diff": max_abs_diff,
        "logits_allclose": logits_close, "passed": passed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    LOG.info("teacher PPL %.4f | student PPL %.4f | rel diff %.2e | max |dlogit| %.2e (first %d batches)",
             ppl_t, ppl_s, ppl_rel, max_abs_diff, args.check_batches)
    print(f"rung0: {'PASS' if passed else 'FAIL'} " + json.dumps(summary))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
