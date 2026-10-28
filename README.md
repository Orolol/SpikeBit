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

**Results.** The May runs are not a usable reference (DECISIONS #14: unrecorded
config, code changed since, ±30 % eval noise). Reference run on 2026-10-28:
`runs/stage_a_repro_may_1500` → PPL 309 (full tail, 1 500 steps) / 394.9 on the
`scripts/evaluate.py` metric, teacher 13.34 on the same tail. Full log and the
A/B tables in `results/stage_a_full_4b.md`. The quantisation stack, not the
budget, is the limiter; the per-tensor MinMax observers lose track of the
activation ranges past ~1.5 k steps.

**Known limitations (reviewed 2026-10-28, see DECISIONS #14/#15).** Stage A is
not yet a faithful conversion: FAI-LIF emits raw integers (scale/zero-point
never folded into the next BitLinear), the channel-mix `relu²` is dropped and
`lif_out` is unsigned, so the converted student starts worse than uniform.
Batch rows do not continue across steps (DEC-2 streaming not implemented),
the in-training eval shares state with training, and the teacher reference
is not fine-tuned on TinyStories. BF16 shadow weights break vanilla AdamW;
use FP32 master + autocast (DECISIONS #12, #13).

**Next steps:** DECISIONS #15 (faithful-conversion ladder with a fine-tuned
teacher reference, stateless-LIF ablation, KL distillation).

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
