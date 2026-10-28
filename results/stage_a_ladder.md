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
