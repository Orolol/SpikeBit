# SpikeBit — Spiking Ternary Language Model

Experimental research project: RWKV-4 architecture, FAI-LIF integer-spike
activations (4-bit by default), BitNet b1.58-style ternary weights with
group-wise scale. The goal is a language model whose weights are all in
{-1, 0, +1} and whose activations are all small integers, so that it could
eventually run without floating-point multiplies.

Reference: Yao et al., *Neuromorphic spike-based large language model*,
NSR, Dec 2025 (4-bit weights). SpikeBit pushes the weights to ternary.

Two-stage validation plan:

- **Stage A** — QAT-conversion of the pre-trained RWKV-4-169M-Pile into a
  spiking-ternary student, fine-tuned on TinyStories. Validates the
  architecture. Target: ≥ 95 % of the teacher's perplexity.
- **Stage C** — train the same architecture from scratch on the same data
  budget. Isolates optimisation risk from architectural feasibility.
  Target: ≥ 80 % of Stage A.

All design decisions live in `DECISIONS.md` (numbered, immutable entries).

## Status (snapshot, 2026-10-28, evening)

Built in an autonomous three-day session (2026-05-19 → 21), paused, resumed
2026-10-28. See DECISIONS #14 and `results/stage_a_full_4b.md` for the resume
day: teacher PPL = 13.34; reference Stage A run `stage_a_repro_may_1500`
(PPL 309 full tail / 395 on the evaluate.py metric at 1 500 steps); every
longer run diverges because the per-tensor MinMax observers lose track of the
activation ranges. Next step: LSQ activation quantizer.

**Done.** Phases 0–4: neurons (FAI-LIF with moving-average MinMax
observer, binary LIF, surrogates), ternary STE + `BitLinear`, spiking
RWKV-4 blocks, teacher conversion (222 tensors), Stage A trainer, eval
pipeline (PPL, generation, spike stats). Fused CUDA kernels for the WKV
recurrence and for FAI-LIF (`layers/csrc`, `neurons/csrc`) make training
~25× faster than the Python loop (0.13 s/step at seq 64 × batch 8 on a
5090, i.e. ~22 min for 10 k steps). Stage C and the Phase 6 ablations are
code only, never run.

**Results so far** (all runs ≈ 1 500 steps, not the planned 10 k; curves
were still descending). Held-out TinyStories perplexity:

| run (`runs/`)                    | activations | path                      | PPL   |
|----------------------------------|-------------|---------------------------|-------|
| `stage_a_fp_baseline` (quant off) | FP          | python                    | 37.8  |
| `stage_a_a8`                     | 8-bit       | python                    | 100.8 |
| `stage_a_midrun_perstep_obs`     | 4-bit       | kernel, per-step scale    | 154.9 |
| `stage_a_midrun_m047`            | 4-bit       | kernel, whole-seq scale   | 176.2 |
| `stage_a_midrun_obs`             | 4-bit       | python                    | 229.0 |

Reading: quantisation costs ~4× in PPL at equal budget; 8-bit activations
halve the gap. The activation quantiser is the bottleneck (DECISIONS #7),
not the ternary weights nor the LIF dynamics. The teacher PPL on the same
eval tail has **not** been measured (`scripts/eval_teacher.py` exists), so
the 95 % gate is not yet evaluable — but it is almost certainly not met.

**Not yet recorded in `DECISIONS.md`** (journal stops at #13, 2026-05-20):
the CUDA kernels (charter #1 listed them as out of scope for v0), the FP
baseline ablation (`scripts/train_fp_baseline.py`, note it also removes
the LIF recurrence so it is an upper bound), the 8-bit run, and the
chunked kernel mode (`FAILIF(seq_chunk_size=...)`, run incomplete).

**Known issues.** `grad_norms` in the Stage A metrics are logged after
`zero_grad` and are always empty (DECISIONS #12). BF16 shadow weights
break vanilla AdamW; use FP32 master + autocast (DECISIONS #12, #13).

**Suggested next steps.**

1. Measure teacher PPL on the eval tail (`scripts/eval_teacher.py`).
2. Run a full 10 k-step Stage A with the fused per-step path to settle
   "budget-limited vs quantisation-limited".
3. If the plateau holds, implement LSQ (learned step size) for the
   activations before investing in Stage C (DECISIONS #6).

## Setup

```fish
uv sync --extra dev
uv run python scripts/check_env.py
```

Python 3.12 via `uv`; torch must ship sm_120 kernels for the RTX 5090.
The GPU is shared with other agents: check `nvidia-smi` before any run
(DECISIONS #5).

## Pipeline

```fish
uv run python scripts/test_rwkv_baseline.py        # Phase 0: teacher loads
uv run python scripts/sanity_mnist.py              # Phase 1 gate (see DECISIONS #6, #10)
uv run python scripts/phase2_forward_sanity.py     # Phase 2 gate
uv run python scripts/convert_rwkv.py              # teacher -> student checkpoint
uv run python scripts/calibrate_rwkv.py            # optional observer warm-up
uv run python scripts/train_stage_a.py --help      # Stage A fine-tune
uv run python scripts/evaluate.py --help           # PPL / generation / spikes
uv run python scripts/eval_teacher.py              # teacher PPL reference
```

Micro-benchmarks: `scripts/bench_wkv.py`, `scripts/bench_fai_lif.py`.
Diagnostics: `scripts/_diagnose_mlp.py`, `scripts/_diagnose_quantizer.py`.

`data/`, `checkpoints/` and `runs/` are git-ignored (≈ 20 GB of
checkpoints locally; `stage_a_init_phase35.pt` is the converted 4-bit
student, `stage_a_init_a8.pt` the 8-bit one).

## Tests

```fish
uv run pytest                      # 95 CPU tests + 11 CUDA tests
CUDA_VISIBLE_DEVICES= uv run pytest   # CPU only (CUDA tests skip)
```
