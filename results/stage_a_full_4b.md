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
  --seq-len 64 --batch-size 32 --grad-accum 1 --total-steps 10000 --warmup-steps 300 \
  --lr-embed-head 1e-3 --mixed-precision bf16 --lif-seq-chunk 8 \
  --log-every 50 --eval-every 500 --ckpt-every 1000
```

seed 0 (default), other LRs per spec (bitlinear 1e-4, lif 5e-5, rwkv 1e-5).
Init: `checkpoints/stage_a_init_phase35.pt` (4-bit, spiking_r=False, empty observers).
Logs: `runs/stage_a_full_4b/{pipeline.log,train.log,teacher_ppl.txt,eval.txt}`.

**Teacher PPL** (measured 2026-10-28, same tail): **13.34** (loss 2.5906, 20 224 tokens) → success threshold: student PPL ≤ 14.0.


## Attempt 2 aborted: slow descent, root cause found (2026-10-28)

Attempt 2 (`--lif-seq-chunk 8`, warmup 500, spec LRs, 0.18 s/step, GPU 82 %)
reached train loss 14.5 / eval loss 15.5 at step 1000 — worse than uniform
(ln 50277 = 10.8) — where the May runs were at 5.3. Killed at step 1100.
Logs: `runs/stage_a_full_4b/attempt2_chunk8/` (checkpoint moved to
`checkpoints/stage_a_full_4b_attempt2/`).

Short A/B series (300 steps, seed 0, 64×8, chunk 8 unless noted, 20k-example
subset where noted — identical early trajectory to the full corpus),
train loss at steps 0 / 100 / 200. May reference (`stage_a_midrun_fast_fp32`):
46.2 / 13.4 / 7.1.

| variant (`runs/ab_early/<name>`) | step 0 | step 100 | step 200 |
|---|---|---|---|
| phase35_wu500_full (attempt-2 config) | 53.6 | 35.7 | 24.3 |
| phase35_wu150_full | 53.6 | 27.1 | 20.3 |
| phase35_wu150_head0 (head frozen) | 53.6 | 32.4 | 28.0 |
| calib_wu150_full (`stage_a_init_calibrated.pt`) | 53.6 | 29.0 | 21.9 |
| phase35_wu150_20k (data subset) | 53.6 | 27.1 | 20.3 |
| phase35_wu150_chunk0 (whole-seq kernel) | 44.7 | 26.2 | 19.1 |
| phase35_wu150_chunk16 | 53.1 | 27.1 | 20.1 |
| phase35_wu150_chunk2 | 50.4 | 26.9 | 21.4 |
| seq128_b4 | 55.0 | 26.3 | 15.7 |
| lr_x3 (all LRs ×3) | 53.6 | 11.7 | 7.2 |
| lr_x10 (all LRs ×10) | 53.6 | 7.9 | 6.9 |
| **lr_head_x10** (`--lr-embed-head 1e-3`, rest per spec) | 53.6 | **7.7** | **6.5** |

Conclusion: not the warmup, init checkpoint, data, kernel chunking, batch
shape, nor the Oct-28 perf patches (the pre-patch chunk-1 attempt was equally
slow). The lever is the LM-head/embedding learning rate: at the spec value
(1e-4) the FP head cannot re-adapt to the integer-spike hidden distribution
fast enough (same mechanism as DECISIONS #9, where a frozen head gave a dead
model). The May runs must have used a higher LR than the spec defaults; it
was not recorded. Decision: full run uses `--lr-embed-head 1e-3`.

Batch size (head LR 1e-3, 0 / 100 / 200): 64×8 53.6 / 7.7 / 6.5; 64×16 53.8 / 7.4 / 6.7;
64×32 52.9 / 7.3 / 6.45. Step time is launch-bound and identical (0.17 s) up to
batch 32 (GPU 88 %, peak 11.6 GB), so attempt 3 uses 64×32 (20 M tokens over 10 k steps).

**Results (attempt 3).** _pending_
