# Stage A full run — 4-bit, per-step kernel path (2026-10-28)

**Pre-registered criterion (written before results).**

- Metric: held-out TinyStories perplexity, `scripts/evaluate.py` defaults
  (last 1 % of the packed corpus, seq 128, batch 2, 20 k tokens).
- Reference: teacher RWKV-4-169M-Pile PPL on the same tail
  (`scripts/eval_teacher.py --seq-len 128 --batch 2 --max-tokens 20000`).
- Secondary reference: FP ablation `runs/stage_a_fp_baseline` reached
  PPL 37.8 at 1 450 steps (quantisation off, LIF = identity).
- Success (charter #1, "≥ 95 % of teacher PPL"): student PPL ≤ 1.05 × teacher PPL.
- Question answered either way: at 10 k steps, is the 4-bit student still
  ~4× the FP baseline (quantisation-limited) or does it close the gap
  (budget-limited)?

**Command** (commit: see `runs/stage_a_full_4b/pipeline.log` header; run from repo root).
A first attempt with `--lif-seq-chunk 1` (per-step kernel calls) ran at 2.2 s/step
(6 h ETA) and was aborted at step 400; its logs are in `runs/stage_a_full_4b/aborted_chunk1/`.
The run below uses chunks of 8 steps after the launch-overhead fixes (154 ms/step):

```
uv run python scripts/train_stage_a.py \
  --init-checkpoint checkpoints/stage_a_init_phase35.pt \
  --out-dir checkpoints/stage_a_full_4b --metrics-path runs/stage_a_full_4b/metrics.jsonl \
  --seq-len 64 --batch-size 8 --grad-accum 1 --total-steps 10000 --warmup-steps 500 \
  --lr-embed-head 1e-4 --mixed-precision bf16 --lif-seq-chunk 8 \
  --log-every 50 --eval-every 500 --ckpt-every 1000
```

seed 0 (default), other LRs per spec (bitlinear 1e-4, lif 5e-5, rwkv 1e-5).
Init: `checkpoints/stage_a_init_phase35.pt` (4-bit, spiking_r=False, empty observers).
Logs: `runs/stage_a_full_4b/{pipeline.log,train.log,teacher_ppl.txt,eval.txt}`.

**Teacher PPL** (measured 2026-10-28, same tail): **13.34** (loss 2.5906, 20 224 tokens) → success threshold: student PPL ≤ 14.0.

**Results.** _pending_
