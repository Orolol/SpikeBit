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


## Long-run instability (2026-10-28, afternoon)

Every run past ~1.5 k steps at a LR that reproduces the May descent diverges:

| run (`runs/`) | config | train loss 1000 → 2000 | eval PPL (200-batch subset) |
|---|---|---|---|
| `stage_a_full_4b/attempt3_h1e3` | 64×32, head 1e-3, rest spec | 6.0 → 6.2 (plateau 370–520 PPL to step 5650) | 371–517 |
| `stage_a_full_4b_h3e4` | 64×32, head 3e-4 | 6.9 at 1000, rising | 1038 at 1000 |
| `stage_a_full_4b_lrx3_b8` | 64×8, all LRs ×3 (May-like) | 5.9 → 7.4 | 448 → 1386 |
| `stage_a_full_4b_lrx3_b8_reset1` | + state reset every step | stuck at ~40 from step 50 | — |
| `stage_a_full_4b_lrx3_b8_frz1k` | + observers frozen at 1000 | 5.9 → 8.7 at 1500 | 6355 at 1500 |

Facts established:
- The May-like run tracks `stage_a_midrun_fast_fp32` step by step (same loss
  bumps at 1100/1300: same data order) with a ~0.5-nat offset consistent with
  May's cosine already decaying (1 500-step schedule).
- Parameters move modestly by step 2000 (ternary distribution unchanged;
  |Δ| shadow 5 % rel.; LN weights up to 0.45 abs.); grads flow to every group.
- In eval mode (grids frozen) resetting the recurrent state per batch costs
  ~10 % PPL (413 → 459): the model does not depend on carried state.
- In train mode, resetting state per step stalls training (observer EMA sees
  cold-start membranes); freezing the observers makes it diverge faster.
  ⇒ the activation statistics are non-stationary under training and the
  observers' tracking is what keeps it alive; the divergence at ~2 k steps is
  the tracking losing the race, not a kernel bug (WKV grads verified with a
  fresh state).

**Results (attempt 3).** aborted at step 5650 (see table above).

**Reference run — `runs/stage_a_repro_may_1500` (closed 2026-10-28 16:35).**
Commit `65a3ae5` (+ results notes), seed 0, init `stage_a_init_phase35.pt`, 64×8,
1 500 steps, warmup 150, LRs ×3 (bitlinear 3e-4, lif 1.5e-4, rwkv 3e-5,
embed/head 3e-4), bf16 autocast, `--lif-seq-chunk 8`, full-tail periodic evals:

| step | train loss | eval PPL (full tail, seq 64) |
|---|---|---|
| 500 | 6.75 | 755.6 |
| 1000 | 5.91 | 390.1 |
| 1500 (final) | 5.51 (step 1400) | **309.2** |

Pre-registered metric (`scripts/evaluate.py`, seq 128, 20 224 tokens): **PPL 394.9**
(loss 5.979). Teacher on the same tail: 13.34. Gate (≤ 14.0): **not met**, by ~30×.
May's `stage_a_midrun_fast_fp32` reported 184 at the same step with an
unrecorded config; the shape of the loss curve is reproduced (same bumps), the
level is not (×1.5–2). Chapter closed: May's numbers are not a usable reference;
this run is. Checkpoint: `checkpoints/stage_a_repro_may_1500/final.pt`.

**Next (DECISIONS #14):** replace the moving-average MinMax observer by a
learned step size (LSQ) before any further Stage A budget.

## Quantizer variants (2026-10-28, 17:00–17:25), all at the reference config

| run (`runs/`) | change | train loss @100 / @300 | outcome |
|---|---|---|---|
| `stage_a_lsq_1500/attempt1_additive` | LSQ, additive step `s`, int output | 23 / 32 | grids could not follow the ×2–6 range drift; saturated |
| `stage_a_lsq_1500/attempt2_logscale_intout` | LSQ, `s = exp(θ)`, int output | 20 / 18.5 | saturation self-perpetuating: clipped elements give no gradient to `s` when the output is the raw integer |
| `stage_a_lsq_1500` (attempt 3) | LSQ, dequantized output `s(q − zp)` | 71 / 71 | membrane (×10 leak gain, two LIFs in series) explodes downstream; inf grads, frozen |
| `stage_a_chan_1500/attempt1_m001` | per-channel MinMax, m = 0.01 | 28 / 35.5 | never learns |
| `stage_a_chan_1500` (m = 0.001) | per-channel MinMax, slower EMA | 28 / 35.5 | idem |
| `stage_a_fixreset_tensor_1500` | **reset fixed** `s(q − zp)`, per-tensor | 14.8 / 6.8 | PPL **415** full tail / 555 evaluate.py (reference 309 / 395: within the run-to-run noise band) |
| `stage_a_fixreset_chan_1500` | reset fixed, per-channel | 29 / 34 | plateau at loss ~31; PPL 1e23 |

Reading (confirmed by the two external reviews): with the raw integer output,
every grid (scale, zero-point) that differs per channel or moves during
training is an affine distortion the next BitLinear must unlearn, and the
teacher's function is already lost at step 0 (loss 45–53 > uniform 10.8).
Changing the quantizer treats the symptom. The reset bug (`s·q` instead of
`s·(q − zp)`) was real and is fixed, but it was not what limited the
per-tensor runs.
