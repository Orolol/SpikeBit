# Stage A ladder (DECISIONS #15)

Protocol for every rung: eval tail = last 1 % of the packed TinyStories corpus,
seq 128, stateless per window, first 1 M tokens. Thresholds are the ones
pre-registered in DECISIONS #15.

## Rung 0 — equivalence (no training)

**Pre-registered gate.** With FP weights (`weight_quant="none"`), identity
FAI-LIF (`act_quant="none"`) and the teacher's channel-mix `relu²` kept
(`ffn_relu_sq=True`), the converted student must give logits allclose to the
HF teacher (FP32, rtol = atol = 1e-3) on the first 4 eval batches, and the
same PPL within 0.1 %. Second test: the folded integer path of `BitLinear`
(`forward_integer`: activation scale into the group scales, zero-point into an
FP bias) equals `BitLinear` on the dequantized activation.

**Command** (branch `rung0-equivalence`, run from the repo root):

```
uv run python scripts/rung0_equivalence.py --out runs/rung0_equivalence/summary.json
```

Defaults: `--seq-len 128 --batch 16 --max-tokens 1000000 --check-batches 4`.
Unit tests: `src/spiking_ternary_lm/conversion/tests/test_teacher_equivalence.py`
(same check on a tiny random RWKV-4, plus the legacy layout shown to differ),
`test_forward_integer_matches_dequantized` in
`src/spiking_ternary_lm/quantization/tests/test_bitlinear.py`.

**Result (2026-10-28, branch `rung0-equivalence` on top of `1370c09`).
PASS.** 222 teacher tensors assigned, none unused. On 1 001 472 tokens:

| | PPL |
|---|---|
| HF teacher | 13.9810 |
| student, quantization off | 13.9810 |

Relative PPL difference 4.9e-9; max |Δlogit| on the first 4 batches 8.0e-5
(gate 1e-3). Unit tests green. Log and JSON:
`runs/rung0_equivalence/{run.log,summary.json}`.

The teacher reference on the DECISIONS #15 protocol is therefore **13.98**
(the earlier 13.34 was on the first 20 k tokens of the tail only).

## Rung 1 — reference: the rung-0 student in FP, fine-tuned

**Config** (protocol of DECISIONS #15): `--teacher-id RWKV/rwkv-4-169m-pile
--weight-quant none --act-quant none --ffn-relu-sq --contiguous-rows
--protocol-eval`, 64 × 32, 10 k steps (20.5 M tokens), warmup 300, cosine to
0.1, AdamW without weight decay, grad clip 1, bf16 autocast over FP32 master
weights, CE only. (A first launch of the sweep forgot `--weight-quant none`,
i.e. trained ternary weights; it was killed within the first minutes and its
outputs deleted.)

**LR selection, written before the runs.** DECISIONS #15 does not fix the LR.
One uniform LR for every trained group (projections, norms, time parameters,
embedding and head), swept over {3e-5, 1e-4, 3e-4, 1e-3} at seed 0 and full
length. Criterion: mean train CE over steps 9 000–9 999, i.e. progressive
validation on text the run has not seen yet; the eval tail is not used for the
choice. The best LR is rerun at seed 1; its seed-0 sweep run is rung 1 seed 0.

## Rung 2 — ternary weights, FP activations (pre-registered before rung 1 results)

Config: rung-1 switches with `--weight-quant ternary`, plus
`--distill-checkpoint <rung 1 seed 0 final.pt>` (KL weight 1, T = 1). Same
LR rule as rung 1, grid {1e-4, 3e-4, 1e-3, 3e-3}, selection on the train CE
(the logged `loss`, without the KL) over steps 9 000–9 999 at seed 0, rerun
at seed 1. Thresholds of DECISIONS #15 on the protocol PPL ratio to rung 1
(mean of its two seeds): ≤ 1.5 continue; 1.5–2 continue and run plan B (a)
(`--ternary-ramp-frac 0.3`) in parallel; > 2 plan B before rung 3.

## Rung 3 — activations on top of rung 2 (pre-registered)

Config: rung-2 config and LR, plus `--act-quant minmax --act-grid faithful
--act-output dequant --lif-stateless --calibrate-batches 8`, at `--bits 8`,
then `--bits 4`, then the pre-registered variant `--bits 4
--bits-ffn-hidden 8`. Faithful grids: zp = 0, signed symmetric for time-mix
k, v and the gated wkv and for the FFN output, unsigned after relu². The
receptance stays FP (`spiking_r=False`, DEC-8), as in every run since May.
Thresholds (ratio to rung 2): 8-bit ≤ 1.1, 4-bit ≤ 1.3.

Implementation notes fixed before these runs: the dequantized output uses a
straight-through gradient of 1 (the legacy integer path and the LSQ path
both had it off by the step size s); the observers are calibrated on the
global min/max of an FP pass over 8 training batches, then follow the usual
EMA (momentum 0.01 per step).

## Rung 1 — sweep results (2026-10-28, 17:53–18:16) and a protocol flaw

Runs `runs/ladder/rung1_lr{LR}_s0` (64 × 32, state carried across contiguous
rows), commits in each `pipeline.log` (`5da5fc8`…`9ed94ed`; the code changes
in between do not touch the FP path). `uv run python scripts/ladder_summary.py`
and `uv run python scripts/_diagnose_state_regime.py teacher rung1_lr…_s0`
(`runs/ladder/rung1_warm_cold.txt`):

| LR | train CE, steps 9 000–9 999 | PPL cold (protocol) | PPL warm (streaming) |
|---|---|---|---|
| zero-shot teacher | — | 13.98 | 15.41 |
| 3e-5 | 1.729 | **8.86** | 5.83 |
| 1e-4 | 1.630 | 9.49 | 5.41 |
| 3e-4 | **1.578** | 10.68 | **5.11** |
| 1e-3 | 1.616 | 9.60 | 5.23 |

The pre-registered rule (lowest train CE) picks 3e-4, which has the *worst*
protocol PPL: the ranking by train CE matches the warm eval and is reversed
on the cold one. CE by position inside the 128-token window shows the cold
penalty is not a cold-start transient: at positions 32–127 the fine-tuned
models are 0.27–0.53 nats worse cold than warm (teacher: 2.45 cold vs 2.74
warm, it was pre-trained from a fresh state). Training with the state carried
across contiguous rows teaches the models to rely on a warm state, and the
more so the higher the LR; the protocol then evaluates them from a cold one.
DECISIONS #15 put both in the protocol without seeing that they conflict.

The ladder is paused on this point (decision for Gaétan). As a diagnostic,
not a replacement, the same sweep runs with stateless training matched to
the eval: seq 128 × batch 16 (same 2 048 tokens per step), state reset every
step (`--reset-state-every 1`), runs `runs/ladder/rung1c_lr{LR}_s0`.

**Diagnostic sweep, stateless training matched to the eval** (18:19–18:42,
commit `88b28dd`): seq 128 × batch 16, `--reset-state-every 1`.

| LR | train CE, steps 9 000–9 999 | PPL cold (protocol) |
|---|---|---|
| 3e-5 | 1.855 | 6.26 |
| 1e-4 | 1.756 | 5.66 |
| 3e-4 | 1.681 | 5.24 |
| 1e-3 | **1.640** | **5.01** |

Train CE and protocol PPL now rank the LRs the same way, and every LR beats
the best warm-trained model on the protocol (5.0–6.3 vs 8.9). The best LR
sits on the edge of the grid, so 3e-3 is added (decided before its result).
The ladder continues under this regime (DECISIONS #16, accepted the same
evening; the runs started before sign-off count): rung-1 seed 1 at the
selected LR, then the rung-2 sweep with KL to the selected rung-1 seed-0
run, runs `runs/ladder/rung{1,2}c_*`. The warm-trained `rung1_lr*_s0` runs
stay as the record of the flaw.

**Rung 1 result.** 3e-3 (train CE 1.656, PPL 5.08) does not beat 1e-3, so the
selected LR is **1e-3**. Seed 0 (`rung1c_lr1e-3_s0`): PPL 5.005; seed 1
(`rung1c_lr1e-3_s1`, disjoint text): PPL 4.967. **Reference = 4.986**
(mean of the two seeds), seed spread 0.8 %, so the noise rule is set by its
5 % floor. KL teacher for rungs ≥ 2: `checkpoints/ladder/rung1c_lr1e-3_s0/final.pt`.

## Rung 2 — results (2026-10-28, 18:53–19:43)

Runs `runs/ladder/rung2c_lr{LR}_s{seed}`: `--weight-quant ternary --act-quant
none --ffn-relu-sq`, KL to rung 1 seed 0, regime of DECISIONS #16.
"Ternary" covers the 84 BitLinear projections (92.0 M parameters); the
embedding and LM head (77.2 M) and the norms stay FP, as in DEC-2.

| LR | train CE, steps 9 000–9 999 | PPL (protocol) | ratio to rung 1 (4.986) |
|---|---|---|---|
| 1e-4 | 1.869 | 6.436 | 1.29 |
| 3e-4 | 1.747 | 5.721 | 1.15 |
| 1e-3 | 1.669 | 5.315 | 1.07 |
| 3e-3 | **1.653** | 5.222 | 1.05 |
| 3e-3, seed 1 | 1.620 | 5.093 | 1.02 |

KL to the rung-1 teacher: 6.12 nats at step 0, 0.09–0.10 at the end.
3e-3 is on the edge of the grid, so 1e-2 is added (DECISIONS #16 rule) before
rung 2 is closed; the seed-1 run at 3e-3 above came from the queue and
counts only if 3e-3 stays selected.

**Eval check** (suspicious-result rule): on `rung2c_lr3e-3_s0/final.pt` the
config says `weight_quant="ternary"`, all 84 BitLinear have `quantize=True`,
each group has a single non-zero magnitude (ternary, 31.7 % zeros), the
forward equals `x @ quantized_weight().T`, and the protocol PPL is 5.222 with
the ternary weights vs 38.7 with the FP shadow weights swapped in.

**Grid extension and closure.** 1e-2 (`rung2c_lr1e-2_s0`): train CE 1.696,
PPL 5.425, worse than 3e-3, so the selected LR stays **3e-3** and the seed-1
run above counts. **Rung 2 = 5.157** (mean of 5.222 and 5.093), seed spread
2.5 %, noise rule at its 5 % floor. Ratio to rung 1: **1.03**, under 1.5:
continue, no plan B.

## Rung 3 — results with the pre-registered quantizer (2026-10-28, 19:54–21:14)

Runs `runs/ladder/rung3c_{a8,a4,a4h8}_s{0,1}`, the pre-registered config at
LR 3e-3 (commit `61aa574`), KL to rung 1 seed 0.

| config | PPL seed 0 | PPL seed 1 | ratio to rung 2 (5.157) | threshold |
|---|---|---|---|---|
| 8-bit | **679.9** (diverged) | 5.982 | 131.8 / 1.16 | ≤ 1.1 |
| 4-bit | 10.118 | 9.385 | 1.89 (mean) | ≤ 1.3 |
| 4-bit, relu² output 8-bit | 10.456 | 10.556 | 2.04 (mean) | ≤ 1.3 |

Taken at face value every configuration fails, including 8-bit on its
healthy seed. The 8-bit seed-0 divergence is a result to explain before any
verdict (suspicious-result rule): 8 bits should be close to lossless.

**What happened.** Total grad norm, logged every 50 steps (pre-clip; the
clip at 1 bounds the step, not the direction): rung 2 has a median of 0.41
and a maximum of 26 (step 0). `rung3c_a8_s0` has 99 of 200 samples above 10,
the first at step 550, a maximum of 2.7e8, and its loss never recovers after
step 5 800. `rung3c_a4h8_s1` reaches 1.3e7 and recovers. The healthy
`rung3c_a8_s1` stays at a median of 0.56.

`scripts/_diagnose_act_scale.py` on the final checkpoints (first 16 windows
of the eval tail, train mode) shows an activation-scale runaway in the
channel-mix under activation quantization, on both 8-bit runs, healthy
seed included (absmax over the batch; 4-bit runs not checked yet):

| block 0 channel-mix | rung 1 | rung 2 | rung 3, 8-bit s1 | rung 3, 8-bit s0 |
|---|---|---|---|---|
| `W_k` output (pre-relu²) | 6.75 | 8.19 | 724 | 310 |
| relu² output (`lif_k` in) | 24.7 | 65 | 3.4e4 | 1.5e4 |
| `W_v` output | 135 | 222 | 9.5e6 | 4.0e6 |

In block 11 the `W_v` output reaches 6.4e3 at rung 2 and 1.3e9 / 2.1e9 at
rung 3. The time-mix stays within a few × of rung 2. The residual stream
grows with it; the final LayerNorm hides it in the forward, but the
backward profile across blocks is no longer smooth (diverged run: residual
grad 1.9e-7 at block 11, 9.1e2 at block 0).

**Hypothesis.** The FAI-LIF STE is the identity outside the grid as well
(DECISIONS #4: "we do *not* mask the gradient outside [0, 2^A − 1] … If
saturation becomes a problem in Phase 3, revisit"). With β = 0 the clipped
excess is lost rather than carried by the membrane. A clipped output is
smaller than its input; the loss pushes it up; the identity STE hands that
push to the pre-activation, whose output cannot move; the pre-activation
grows, the EMA max follows, the grid widens, and relu² squares the growth.
Saturated elements are few (1e-5 to 3e-3 per LIF at qmax) but they are the
largest ones, where relu²'s backward (2z) is largest.

**Diagnostic in progress.** The same six runs with
`--act-ste clipped` (gradient 0 where the membrane is outside the grid, commit
`0bcd51c`), same LR and seeds: runs `runs/ladder/rung3m_*`. This changes the
quantizer of the pre-registered rung 3; DECISIONS #17 (accepted the same
evening) makes these runs rung 3, and the `rung3c_*` runs stay as the
record of the failure.

## Rung 3 — results with the clipped STE, and LSQ vs MinMax (2026-10-28 21:20 – 2026-10-29 12:03)

Runs `runs/ladder/rung3m_*` (MinMax, DECISIONS #17) and `runs/ladder/rung3l_*`
(LSQ): the pre-registered configs with `--act-ste clipped`, LR 3e-3, KL to
rung 1 seed 0, seeds 0 and 1. The LSQ runs are the comparison DEC-15
pre-registers at this rung ("LSQ vs MinMax is compared here only"). Before
they started, LSQ was made to differ from MinMax by the learned step only
(commit `1a5623a`, which leaves the MinMax path unchanged): same calibrated
starting grid (same step-0 loss, 9.9978 at seed 1), zp fixed at 0 on the
symmetric grids, clipped STE. One step per tensor, log-parameterised, LR 1e-2
(trainer default), gradient scaled by `1/sqrt(N·Q_P)` (Esser et al.).

Hardware: the local 5090 for the five runs started by 22:20. The sixth,
`rung3m_a4h8_s1`, was stopped at step 300 to free the GPU and rerun on a Colab
A100 with the four `rung3l_*` runs (marked A100 below), two at a time, via
`colab/ladder_rung3.py` at commit `5ca50e5` (KL teacher and packed data checked
by sha256). A first A100 attempt ran the per-step Python fallback (the image
has no ninja, so the CUDA extensions did not build: 6.7 s/step) and was
stopped at step 200; both aborted runs are in `runs/ladder_aborted/`. The code
is the same on both GPUs; bf16 numerics are not bit-identical (0.1 % on the
rung-2 checkpoints, below).

| quantizer | config | PPL seed 0 | PPL seed 1 | mean | ratio to rung 2 (5.157) | threshold |
|---|---|---|---|---|---|---|
| MinMax | 8-bit | 5.237 | 5.104 | 5.171 | **1.003** | ≤ 1.1: pass |
| MinMax | 4-bit | 10.371 | 9.702 | 10.037 | 1.946 | ≤ 1.3: fail |
| MinMax | 4-bit, relu² output 8-bit | 9.498 | 10.082 (A100) | 9.790 | 1.898 | ≤ 1.3: fail |
| LSQ | 4-bit | 5.389 (A100) | 5.267 (A100) | 5.328 | **1.033** | ≤ 1.3: pass |
| LSQ | 4-bit, relu² output 8-bit | 5.358 (A100) | 5.241 (A100) | 5.300 | 1.028 | ≤ 1.3: pass |

The clipped STE fixes 8 bits: no divergence on either seed, and the MinMax
8-bit model is within 0.3 % of rung 2. At 4 bits MinMax fails on every seed
and config, by about the same margin as with the identity STE (`rung3c_*`,
1.89 and 2.04). LSQ at 4 bits is within 3.3 % of rung 2, and keeping the relu²
output at 8 bits buys only 0.5 % more; the gap between the two quantizers
(×1.9) is far outside the noise rule (5 % floor).

**Eval check** (suspicious-result rule, `runs/ladder/check_quant_*_colab.txt`,
first 32 windows of the eval tail, eval mode). On `rung3l_a4_s{0,1}` and
`rung3l_a4h8_s1`, all 60 FAI-LIFs are in LSQ mode and every output is on its
grid (`max |q − round(q)|` ≤ 2.7e-2, bf16 rounding); in blocks 0 and 11 the
4-bit LIFs use 10–16 levels, the 8-bit relu² output 181–184. With the same weights and
FP activations the CE gets worse (1.39 → 1.64 at seed 0, 1.38 → 1.67 at
seed 1): the model is fitted to its 4-bit grids, not running around them.

**Activation scale at 4 bits** (`scripts/_diagnose_act_scale.py`, run on the
A100, `runs/ladder/diag_act_scale_{rung3m_colab,rung3l_a4_s0_colab}.txt`;
grid steps of LSQ from the eval check, since the script prints the observer's
step, which LSQ does not use). The clipped STE does not stop the runaway at 4 bits:

| block 0 | rung 2 | MinMax 8-bit (a8 s0) | MinMax 4-bit (a4 s0 / a4 s1 / a4h8 s0) | LSQ 4-bit (a4 s0) |
|---|---|---|---|---|
| channel-mix `W_k` output absmax | 8.19 | 8.75 | 486 / 209 / 187 | 7.94 |
| channel-mix `W_v` output absmax | 222 | 204 | 3.1e7 / 2.8e6 / 2.5e6 | 138 |
| channel-mix `lif_v` grid step | – | 1.08 | 3.7e6 / 3.3e5 / 2.8e5 | 6.98 |
| time-mix `lif_v` grid step | – | 0.044 | 35.5 / 22.1 / 17.9 | 0.239 |
| residual-stream grad at block 0 | n/a | 3.3e-3 | 1.1e-8 / 1.1e-7 / 2.1e-7 | 3.9e-3 |

Block 11 channel-mix `W_v` output: 3.2e3 at 8 bits, 1e9 / 9.8e7 / 7.9e7 with
MinMax at 4 bits, 2.9e3 with LSQ at 4 bits (rung 2: 6.4e3). With MinMax at
4 bits the block-0 time-mix inflates too (`W_k` output ×5–9, `W_v` ×20–50 over
8 bits), which it did not at 8 bits.

**Reading.** A MinMax grid is set by the EMA of the activation max, so at
16 levels a wider range sends every non-extreme value to the zero bin; the
loss can then only push magnitudes up, and the grid follows. LSQ sets the
step from the loss: it clips instead (block 0 time-mix `lif_v`: 21–24 % of
the outputs at the grid ends on all three LSQ runs, vs 7e-5 for MinMax
`a4h8 s1`) and keeps its resolution.
This is the mechanism the data fit, not one tested in isolation.

**Warm (streaming) eval** (DEC-16 point 3, secondary; missing at rung 2 until
now). `scripts/_diagnose_state_regime.py` on the A100,
`runs/ladder/warm_cold_rung2_rung3_colab.txt`:

| run | cold (protocol) | warm (streaming) |
|---|---|---|
| rung 2, s0 / s1 | 5.217 / 5.088 | 4.566 / 4.456 |
| MinMax 8-bit, s0 | 5.232 | 4.546 |
| LSQ 4-bit, s0 / s1 | 5.384 / 5.262 | 4.616 / 4.508 |

LSQ 4-bit warm: 4.562 vs 4.511 for rung 2 (1.011). The cold column also
measures the hardware effect: the rung-2 checkpoints give 5.217 / 5.088 on the
A100 vs 5.222 / 5.093 on the 5090 (0.1 %).

**Verdict** (DECISIONS #18, accepted 2026-10-29): rung 3 passes at 4 bits with
LSQ (1.033 ≤ 1.3) and fails with MinMax (1.95); 8 bits passes with MinMax.
**Rung 3 = LSQ, all-4-bit activations, PPL 5.328**, the reference of rungs 4a
and 4b.
