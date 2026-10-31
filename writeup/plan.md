# Write-up plan (DECISIONS #24, after tiers 1 and 2)

Status 2026-10-31: plan only, no draft yet. Format (arXiv note, full paper or
blog post) to be decided; section 6 lists what each one needs. Every number
below is in `results/` with its command and commit.

## 1. Thesis

Reading a quantized LLM's activations as spikes does not save work against an
integer engine running the same network. SpikeBit is a case where the
comparison can be made exactly: its 4-bit activations admit an exact ±1 spike
reading, so a spiking engine and an integer engine compute identical logits,
and the only difference between them is how many accumulations each performs
and what each one costs.

## 2. Draft abstract

Spiking language models are often motivated by energy: activations become
binary events, multiplications become additions, and work happens only where
neurons fire. We test that motivation on a network where the comparison is
exact. We convert RWKV-4-169M to ternary weights and 4-bit activations produced
by stateless LIF neurons, at 1.08 × the perplexity of the FP fine-tuned model
on TinyStories. Each 4-bit activation can be read exactly as up to 8 or 15 ±1
spikes, so a spiking engine and an integer engine compute the same logits.
Counted on the same weights and inputs, the spike reading needs 1.7–3.4 × the
accumulations of an integer engine that skips zero activations, at every site;
it beats an engine that does not skip zeros only at two sparse sites. Coding
the change between consecutive tokens saves nothing. The energy formula common
in the SNN literature (0.9 pJ per accumulate against 4.6 pJ per FP32 MAC)
credits the spikes with a 3.3 × saving that decomposes into removed
multiplications and skipped zeros, both available to integer engines, times a
spike-count penalty. Published per-event energies of neuromorphic chips, and
the cost of fetching a weight, put a spike accumulation at no less than 0.82 of
an integer one, where spikes lose. A membrane carried across tokens, with its
leak fixed or learned, did not improve the model.

## 3. Claims and evidence

| # | claim | evidence | status |
|---|---|---|---|
| C1 | The ternary 4-bit conversion holds: rung 5, PPL 5.391, 1.081 × rung 1 (92.5 %) | `results/stage_a_ladder.md`, DEC-22 | measured, 2 seeds |
| C2 | The spike reading is exact for every BitLinear consumer | unit tests and real-count checks, rung 4a and tier 1 | measured |
| C3 | Against a zero-skipping integer engine, `R = S/Z ≥ 1` by construction; measured 1.7–3.4 per site, 1.9 (rung 3) and 2.8 (rung 5) pooled | `results/spike_accounting.md` | measured, 3 models × 2 seeds |
| C4 | Against a dense integer engine, spikes win only where activations are sparse: `ρ*` ≈ 3.3 and 15 at the two sparse sites, 0.33–0.47 at the input sites, 0.65 pooled at rung 5 | same | measured |
| C5 | Delta coding between tokens saves nothing: `ΔZ/Z` 0.97–1.01 at dense sites, 1.24–1.58 at sparse ones | same | measured |
| C6 | The SNN papers' 3.3 × on rung 5 = 5.11 (multiplications removed) × 1.82 (zeros skipped) × 0.36 (spike count); a zero-skipping integer engine gets 9.3 × with the same constants | same, decomposition table | measured; constants sourced |
| C7 | Real hardware prices a spike accumulation at `ρ ≥ 0.82` of an integer one: adds priced by accumulator width, weight fetch, per-event energies of Loihi, TrueNorth and SpiNNaker 2, Loihi 2 graded spikes | `results/hardware_cost_note.md` | sourced; no measurement of ours |
| C8 | A membrane brings no gain at this scale, leak fixed (4b, 5.543) or learned (4c, 5.523); β = 0 at eval does not hurt | `results/stage_a_ladder.md` | measured, 2 seeds each |
| C9 | Design lesson: if spikes at all, only at sparse sites (relu² output, time-mix output), and a zero-skipping integer engine still wins there by `R` (1.7–2.2) | C3, C4 | follows from C3–C4 |

C3 is a short theorem plus a measurement. A reviewer may call it obvious; the
paper's weight is in C4–C7 (the dense-baseline split, the decomposition of
the literature's formula, the hardware placement) and in C2 (the same network
on both sides, which most SNN-vs-ANN comparisons do not have).

## 4. Outline

  1. **Introduction** (1 p). Claims made for spiking LLMs (SpikeGPT's 32 ×,
     NSLLM's FPGA figures); why they are hard to check (different networks,
     FP32 baselines); what an exact reading allows; the contributions C2–C8.
  2. **Setup** (1 p). RWKV-4-169M, ternary BitLinears, stateless 4-bit LSQ
     LIFs, the conversion ladder in one table (rungs 0–5, 4b, 4c), protocol and
     noise rule. Points to the ladder report for details.
  3. **Exact spike reading** (0.5 p). Signed and unsigned decompositions,
     linearity through a ternary consumer, the check on real counts. The
     consumers where it does not apply (WKV, FP gate).
  4. **Accounting** (1.5 p). Engines D, Z, S, ΔZ, ΔS; `ρ*` and the bands
     fixed before measurement (DEC-24); results per site and pooled; delta
     coding.
  5. **What the SNN formula counts** (0.5 p). The decomposition (C6).
  6. **Hardware** (1 p). Accumulator width, weight fetch, per-event chip
     figures, Loihi 2 graded spikes, the whole-model view (FP head, DRAM).
  7. **The membrane** (0.5 p). 4b and 4c, β distributions, the β = 0
     ablation, the detached-gradient caveat.
  8. **Limits and scope** (0.5 p). See section 7 below.
  9. **Related work** (0.5–1 p), then conclusion.

About 7 pages plus appendix (per-site tables, protocol, reproduction).

## 5. Figures and tables

  * **Fig. 1**: one network, three readings (dense integer, zero-skipping
    integer, spikes), with what each counts.
  * **Fig. 2** (main figure): `ρ*` per site on a log axis, one marker per
    model and seed, against D, Z and ΔZ, with the bands [1/4, 1] shaded and
    the hardware points of the note drawn as horizontal lines. Data:
    `runs/accounting/*.json`; script to write (`scripts/accounting_figure.py`).
  * **Fig. 3**: β histograms, rung 4b against rung 4c.
  * **Table 1**: the ladder (quality).
  * **Table 2**: pooled counts per model (from `scripts/accounting_tables.py`).
  * **Table 3**: the decomposition of the SNN formula.
  * **Table 4**: hardware points and their verdicts.

## 6. What each format needs

| | blog post / report | arXiv note (workshop length) | full paper (TMLR or similar) |
|---|---|---|---|
| Measurements | done | done; tier 3 advisable | tier 3; a second architecture (a ternary transformer) |
| Related work | a paragraph | verified section | full section |
| Figures | Fig. 2 | Fig. 1–3 | Fig. 1–3 + appendix |
| Checks in section 8 | NSLLM authorship | all | all |
| Repository | public or not (Gaétan's call) | public code expected | public code expected |
| Effort | 1–2 days | about a week | several weeks |

Workshop deadlines (negative-result or neuromorphic tracks) are not checked;
look them up before choosing.

## 7. Scope to state

  * One model and size (RWKV-4-169M), one dataset (TinyStories), 32 k tokens
    for the counts. General text is DEC-24 tier 3, not run.
  * The reading covers 4-bit counts read as rate-coded spikes. Natively
    trained SNNs whose neurons integrate spikes over time steps (stateful
    consumers) are not covered: their output depends on spike timing, not only
    on the count.
  * Digital hardware only. Analog or in-memory neuromorphic substrates price
    accumulation differently; the note does not cover them.
  * The LM head stays FP; at batch 1 from DRAM, weight movement dominates any
    arithmetic difference.
  * Counts, not a cost model of a built engine; the hardware figures come from
    different nodes and operating points.

## 8. Checks before any draft

  * NSLLM authorship: DEC-1 and the README say Yao et al.; the PubMed Central
    list read for the note starts with Han Xu and Xuerui Qiu.
  * SpikeGPT's headline factor: 32.2 × in its Table 1, 33.2 × in its text.
  * Spikformer: wording read in arXiv v2, not the ICLR camera-ready.
  * Dampfhoffer et al.: read in part only.
  * Related work not yet read, to find and verify: other spiking LMs
    (SpikeLM, SpikeLLM, SpikeBERT, SpikingBERT), BitNet b1.58 and a4.8,
    delta and sigma-delta networks, zero-skipping accelerators (e.g.
    Cnvlutin), ANN-vs-SNN comparisons on digital hardware.

## 9. Decisions for Gaétan

  1. Format: blog post, arXiv note, or full paper (section 6).
  2. Tier 3 (general text, ≈ 2.5 h of A100 and a 0.2–0.3 GB download): needed
     for a paper, optional for a blog post.
  3. Whether the repository becomes public, and when.
  4. Language of a blog post (EN or FR).
