# Stage A — summary (2026-10-30)

Closes Stage A (DECISIONS #1) as rebuilt by DECISIONS #15–#23. Every number
below comes from `results/stage_a_ladder.md`, where each one has its command,
commit, run directory and seed; this page adds no measurement.

## Result

RWKV-4-169M converted to ternary weights (the 84 BitLinear projections, 92.0 M
parameters) and 4-bit activations at every BitLinear input (120 stateless
FAI-LIFs with a learned LSQ step), fine-tuned on 20 M tokens of TinyStories with
KL distillation from the FP fine-tuned student: **PPL 5.391**, 1.081 × the FP
fine-tuned reference (4.986). Rung 1 / rung 5 = **92.5 %**: DEC-1's Stage A
target (≥ 95 %) is **missed**. At 2 × the budget the ratio is 94.3 %, still
under 95 % (DEC-23).

Two parts of DEC-1's Stage A do not hold as written; the claim was revised
(DEC-21), not the metric:

  * "Spiking": the LIFs end up as stateless 4-bit quantizers. A membrane carried
    across tokens did not help (rung 4b). The ±1 spike reading of their outputs
    is exact but takes more accumulations than INT4 × ternary (2.8 × at rung 5);
    it saves energy only on hardware where a 1-bit accumulation costs under
    1/2.8 of a 4-bit one.
  * "All activations integer-spike": at rung 5 every BitLinear input is a 4-bit
    LIF output, but the embeddings, the LM head (29.6 % of the dense matmul
    MACs), the norms, the WKV recurrence and the sigmoid gates stay FP, as DEC-2
    lists.

## The ladder

Protocol (DEC-15, DEC-16): eval on the first 1 M tokens of the last 1 % of
TinyStories, 128-token windows, fresh state per window; training 10 k steps of
128 × 16 (20.5 M tokens), state reset before every window, seeds 0 and 1 on
disjoint text; KL to rung 1 seed 0 from rung 2 on. A difference counts only
above max(2 × seed spread, 5 % PPL).

| rung | what changes | PPL s0 / s1 | mean | vs previous rung | vs rung 1 | pre-registered rule |
|---|---|---|---|---|---|---|
| 0 | converted student, quantization off | – | 13.98 (= HF teacher) | – | – | equivalence: pass |
| 1 | FP student fine-tuned (reference) | 5.005 / 4.967 | 4.986 | – | 1 | – |
| 2 | ternary weights | 5.222 / 5.093 | 5.157 | 1.034 | 1.034 | ≤ 1.5: pass |
| 3 | + 4-bit LSQ at the 60 LIF sites | 5.389 / 5.267 | 5.328 | 1.033 | 1.069 | ≤ 1.3: pass |
| 4a | ±1 spike reading of rung 3 | exact | – | – | – | measurement only |
| 4b | + membrane carried across tokens | 5.590 / 5.497 | 5.543 | 1.040 | 1.112 | kept if < 5.062: not kept |
| 5 | rung 3 + 4-bit LSQ on every BitLinear input | 5.444 / 5.338 | **5.391** | 1.012 | 1.081 | ≤ 1.1 × rung 3: pass |

Where the gap goes (rung 1 → rung 5 = 0.078 nats): ternary weights 0.034
(43 %), 4-bit activations at the LIF sites 0.033 (42 %), 4-bit BitLinear inputs
0.012 (15 %). Each step on its own is under the noise rule's 5 % floor; only the
cumulative gaps (rung 1 → 3: 6.9 %, rung 1 → 5: 8.1 %) count. The split is
indicative.

## Budget ×2 diagnostic (DEC-23)

Rung 1 and rung 5 replayed at 20 k steps (41 M tokens), rung 5 distilled from
rung 1 ×2 seed 0.

| | PPL s0 / s1 | mean | ×2 vs ×1 |
|---|---|---|---|
| rung 1 ×2 | 4.694 / 4.650 | 4.672 | −6.3 % |
| rung 5 ×2 | 5.005 / 4.905 | 4.955 | −8.1 % |

Ratio 94.3 % < 95 %: **not budget-limited at 2×** by the pre-registered binary
rule. The miss (0.75 % PPL) is smaller than rung 5's seed spread (2.0 %), so
rung 5 sits at the threshold at 2×, not clearly under it. The gap to rung 1 goes
from 8.1 % to 6.1 % (0.078 → 0.059 nats): doubling the budget closes a quarter
of it. The extra budget helps the quantized model more than the reference, but
not enough to reach 95 % at 2×.

## What was learned on the way

  1. **The May design could not convert** (DEC-14, DEC-15). The converted
     student started at loss 45–53 (uniform: 10.8) and diverged after ~1.5 k
     steps: FAI-LIF emitted raw integers whose scale was never folded into the
     next layer, the teacher's relu² was dropped, and a reset bug left an `s·zp`
     membrane offset. The ladder rebuilt the conversion so that rung 0 *is* the
     teacher (max |Δlogit| 8e-5, PPL equal within 5e-9 relative).
  2. **Train and eval must share the state regime** (DEC-16). Trained with the
     state carried across rows and evaluated cold, the LR ranking flipped: the
     best train CE gave the worst protocol PPL (10.7, vs 5.0 trained stateless).
  3. **Activation quantization is the fragile layer** (DEC-7, confirmed). With
     DEC-4's identity STE the clipped excess pushed the pre-activations up,
     relu² squared the growth and the MinMax grid followed it (block-0
     channel-mix `W_v` output 222 → 4e6–1e7; one 8-bit seed diverged). The
     clipped STE (DEC-17) fixes 8 bits. At 4 bits MinMax still runs away
     (1.95 × rung 2); LSQ, which sets its step from the loss and accepts
     clipping, holds (1.033 ×, DEC-18).
  4. **Membrane memory did not help at this scale** (rung 4b). The gradient
     through the carried membrane exploded (6.5e15, ×30 per block backward) and
     was detached (DEC-20). Trained that way, β stayed at its 0.9 init, the model
     ignored its membrane (β set to 0 at eval: −0.02 % / −0.4 %), and its
     weights were worse than rung 3's.
  5. **Spike reading** (rung 4a, rung 5). Exact by construction for BitLinear
     consumers. With ternary weights the INT4 baseline has no multiplication to
     remove and both readings skip zeros, so the accumulation ratio R ≥ 1:
     1.9 at rung 3 (on 27 % of the dense matmul MACs), 2.8 at rung 5 (on
     70.4 %; the input sites are dense, 83–87 % non-zero). If a 1-bit
     accumulation costs 1/4 of a 4-bit one, R / 4 = 0.47 and 0.70.
  6. **The quantized student is fitted to its grids.** Every LIF output is on
     its grid, and the same weights with FP activations are much worse (rung 5:
     CE 1.40 → 2.42 / 2.34; rung 5 ×2: 1.30 → 2.67 / 2.72).

## Protocol changes made after seeing results

Each is recorded in DECISIONS with its reason; they are listed here because they
bear on how much of the ladder is pre-registered. Thresholds and the noise rule
never changed. Rung 5 (DEC-22) and the ×2 diagnostic (DEC-23) were
pre-registered before any of their runs.

| DEC | change | trigger | runs started before sign-off |
|---|---|---|---|
| 16 | stateless 128 × 16 training (was stateful 64 × 32) | the rung-1 sweep ranked the LRs in reverse | counted (`rung1c_*`, `rung2c_*`) |
| 17 | clipped STE on the dequantized LIF output | rung-3 divergence, mechanism diagnosed | counted (`rung3m_*`) |
| 18 | LSQ kept for the later rungs | the comparison was pre-registered, which quantizer to keep was not | – |
| 20 | no gradient through the carried membrane | rung 4b did not train (infinite grad norm) | – |
| 21 | claim revised, rung 5 added | DEC-15's positioning note | – |

Hardware: rungs 1–2 and part of rung 3 ran on the local 5090, the rest on a
Colab A100; the same checkpoints evaluate 0.1 % apart on the two (rung 2).

## Limits

  * One model (RWKV-4-169M), one dataset (TinyStories, simple text), 20 M
    tokens; two seeds per rung, seed spreads 0.8–2.5 %.
  * The primary metric is cold (fresh state per 128-token window). The warm
    (streaming) eval is secondary, and was not measured for rung 1 at 1×.
  * The denominator is the FP *fine-tuned* student (DEC-15), not DEC-1's
    zero-shot teacher (13.98), which every trained rung beats by about ×2.6.
  * Rung 1 seed 0 is both the KL teacher of every quantized rung and half of the
    denominator.
  * No energy number: the spike-reading proxy counts accumulations; picojoules
    need a hardware cost model.

## Open questions and next steps (to decide)

  1. **Gate to Stage C.** DEC-1, as DEC-21 point 3b reads it, asks for a
     redesign before Stage C if Stage A fails; rung 5 is at 92.5 %. Options:
     accept rung 5 as the Stage A result Stage C is measured against (≥ 80 %
     of it), with a new DECISIONS entry; work on the gap first; or write
     Stage A up and stop.
  2. **Warm eval of rung 1 at 1×** (two checkpoints, a few minutes of GPU). At 2×
     the quantized student is better warm than the FP reference (4.194 vs 4.413,
     5 % apart, at the noise floor): the FP model gets worse late in a window
     when its state is carried, the quantized one gets better. Whether this
     appears with the budget or was there at 1× is unknown.
  3. **Budget or capacity.** At 2× rung 5 sits at the threshold. A ×4 replay
     (four runs, about 6 h of A100 two at a time) would settle it; worth it only
     if 95 % matters for point 1.
  4. **Stage C trainer.** `training/stage_c_scratch.py` predates the ladder (no
     LSQ, no input LIFs, no protocol eval or stateless windows); it needs a port
     before any run.
  5. **Quiet fallback.** When the CUDA extensions fail to build, training falls
     back to the Python path (6.7 s/step on the A100) with a one-line print and
     no error; the Colab drivers assert that the extensions load, the trainer
     does not.

## Reproduction

Branch `rung0-equivalence`. Colab drivers in `colab/`; analysis scripts in
`scripts/`: `rung0_equivalence.py`, `ladder_summary.py`,
`rung4a_spike_reading.py`, `_check_quant.py`, `_diagnose_state_regime.py`,
`_diagnose_act_scale.py`. Checkpoints are local and git-ignored
(`checkpoints/ladder/<run>/final.pt`):

| run | role | sha256 (first 16 hex) |
|---|---|---|
| `rung1c_lr1e-3_s0` / `_s1` | rung 1; s0 is the KL teacher at 1× | `04ea63897adf6dc7` / `cc99bc09e9c3b2f3` |
| `rung3l_a4_s0` / `_s1` | rung 3 | `bc4e9beb0344b466` / `c4baa5ef1db6c586` |
| `rung5_a4_s0` / `_s1` | rung 5, Stage A's result | `c19948348dc88553` / `182eeebb75edad6a` |
| `rung1x2_lr1e-3_s0` / `_s1` | rung 1 ×2; s0 is the KL teacher at 2× | `01c01e3f856cca71` / `f197e41359460f19` |
| `rung5x2_a4_s0` / `_s1` | rung 5 ×2 | `02a8f912721a68e4` / `7940d58e8de56ecc` |
