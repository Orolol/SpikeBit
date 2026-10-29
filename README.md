# SpikeBit — ternary RWKV-4 with 4-bit LIF activations

Experimental research project: RWKV-4 architecture, ternary weights
(BitNet b1.58 style, group-wise scale) and FAI-LIF neurons that quantize
activations to 4-bit integers.

**Claim (DECISIONS #21).** SpikeBit converts RWKV-4-169M to ternary weights and
4-bit activations at the LIF sites at 1.07 × the FP fine-tuned perplexity on
TinyStories. The LIF neurons act as stateless 4-bit quantizers; their outputs
admit an exact ±1 spike reading, which costs less than an INT4 × ternary
baseline only where a 1-bit accumulation costs under ~half a 4-bit one, on
27 % of the matmul work. A membrane carried across tokens brought no gain at
this scale. No energy claim without a hardware cost model.

Reference: Yao et al., *Neuromorphic spike-based large language model*,
NSR, Dec 2025 (4-bit weights). SpikeBit pushes the weights to ternary.

Two-stage validation plan (DECISIONS #1):

- **Stage A** — QAT-conversion of the pre-trained RWKV-4-169M-Pile into the
  ternary student, fine-tuned on TinyStories. Validates the architecture.
  Target: ≥ 95 % of the teacher's perplexity.
- **Stage C** — train the same architecture from scratch on the same data
  budget. Isolates optimisation risk from architectural feasibility.
  Target: ≥ 80 % of Stage A.

All design decisions live in `DECISIONS.md` (numbered, immutable entries).

## Status (2026-10-29)

Stage A was rebuilt as a measured ladder (DECISIONS #15–#21); results in
`results/stage_a_ladder.md`. Protocol: last 1 % of TinyStories, seq 128,
stateless windows, 1 M tokens; two seeds per rung.

| rung | what changes | PPL | ratio |
|---|---|---|---|
| 0 | converted student, quantization off | 13.98 (= teacher) | exact |
| 1 | FP student fine-tuned (reference) | 4.986 | – |
| 2 | ternary weights | 5.157 | 1.03 × rung 1 |
| 3 | + 4-bit LSQ activations at the 60 LIF sites | **5.328** | 1.03 × rung 2 |
| 4a | ±1 spike reading of rung 3 | exact; 1.9 × the accumulations of INT4 × ternary | – |
| 4b | + membrane carried across tokens (β ≈ 0.9) | 5.543 | not kept |

Stage A's result is rung 3, at 93.6 % of rung 1 (target ≥ 95 %). Five of the
seven BitLinears per block still take the FP LayerNorm output; quantizing
those inputs is rung 5 (next, to be pre-registered). Embeddings, LM head and
norms stay FP (DECISIONS #2). Stage C and the Phase 6 ablations are code
only, never run.

**Done.** Phases 0–4: neurons (FAI-LIF with MinMax or LSQ grids, binary LIF,
surrogates), ternary STE + `BitLinear`, spiking RWKV-4 blocks, teacher
conversion (222 tensors), Stage A trainer, eval pipeline (PPL, generation,
spike stats). Fused CUDA kernels for the WKV recurrence and for FAI-LIF
(`layers/csrc`, `neurons/csrc`) make training ~25× faster than the Python
loop. Colab drivers for the ladder runs in `colab/`.

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
