# Write-up plan (DECISIONS #24, after tiers 1 and 2)

Status 2026-10-31: LaTeX source in `writeup/note.tex` (revision 3, converted from `writeup/note.md`); build and
arXiv tarball by `bash writeup/make_arxiv.sh` (tectonic); this plan is kept for the claims table. Format chosen by Gaétan on
2026-10-31: an arXiv note of workshop length (section 6). Every number below
is in `results/` with its command and commit.

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
| C1 | The ternary 4-bit conversion holds on TinyStories: rung 5, PPL 5.391, 1.081 × rung 1 (92.5 %); on WikiText-103 only 1.39 × (71.8 %) | `results/stage_a_ladder.md`, DEC-22; `results/spike_accounting.md` tier 3 | measured, 2 seeds; TinyStories-only as a quality claim |
| C2 | The spike reading is exact for every BitLinear consumer | unit tests and real-count checks, rung 4a and tier 1 | measured |
| C3 | Against a zero-skipping integer engine, `R = S/Z ≥ 1` by construction; measured 1.7–3.4 per site, 1.9 (rung 3) and 2.8 (rung 5) pooled | `results/spike_accounting.md` | measured, 3 models × 2 seeds |
| C4 | Against a dense integer engine, spikes win only where activations are sparse: `ρ*` ≈ 3.3 and 15 at the two sparse sites, 0.33–0.47 at the input sites, 0.65 pooled at rung 5 | same | measured |
| C5 | Delta coding between tokens saves nothing: `ΔZ/Z` 0.97–1.01 at dense sites, 1.24–1.58 at sparse ones | same | measured |
| C6 | The SNN papers' 3.3 × on rung 5 = 5.11 (multiplications removed) × 1.82 (zeros skipped) × 0.36 (spike count); a zero-skipping integer engine gets 9.3 × with the same constants | same, decomposition table | measured; constants sourced |
| C7 | Under event-driven cost models that read a weight per spike (design (a)), unit spikes cost more than Z at every site (`ρ_eff ≥ 0.82`); with the weight held across a source's spikes (design (b)) the verdict is the cost of a ±1 increment, open | `results/hardware_cost_note.md` (revised 2026-10-31) | cost models from sourced figures; no measurement of ours |
| C8 | A membrane brings no gain at this scale, leak fixed (4b, 5.543) or learned (4c, 5.523); β = 0 at eval does not hurt | `results/stage_a_ladder.md` | measured, 2 seeds each |
| C9 | Design lesson: if spikes at all, only at sparse sites (relu² output, time-mix output), and a zero-skipping integer engine still wins there by `R` (1.7–2.2) | C3, C4 | follows from C3–C4 |

C3 is a short theorem plus a measurement. A reviewer may call it obvious; the
paper's weight is in C4–C7 (the dense-baseline split, the decomposition of
the literature's formula, the hardware placement) and in C2 (the same network
on both sides, which most SNN-vs-ANN comparisons do not have).

## 4. Outline

  1. **Introduction** (1 p). Claims made for spiking LLMs (section 8: six
     papers price spikes at 0.9 pJ against a 4.6 pJ MAC; SDLLM charges an INT4
     baseline that FP32 MAC price; NSLLM's 19.8 × is a power ratio); why they are hard to check (different networks,
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
    the hardware points of the note drawn as a hatched band. Done for rung 5:
    `writeup/figures/rho_star.pdf`, from `uv run --with matplotlib==3.10.7
    python scripts/accounting_figure.py` (rung 3 and rung 5 ×2 stay in the
    appendix tables).
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

## 8. Related work, verified (2026-10-31)

Source notes, with the version read and the location of every figure:
`runs/writeup_sources/related_spiking_lms.md` and `related_background.md`
(git-ignored, with the downloaded PDFs). Paraphrased here; figures marked
"our arithmetic" are computed from the papers' printed tables.

**Energy claims of spiking LMs.** All but one price a spike accumulation at
0.9 pJ and a MAC at 4.6 pJ (SpikeGPT: 4.5), the FP32 pair of Horowitz.

| paper | headline | baseline and how it is priced | what the decomposition says |
|---|---|---|---|
| SpikeGPT (Zhu et al., TMLR 2024) | 32.2 × (abstract, Table 1); 33.2 × in the introduction only | analytical GPT at 4.5 pJ per MAC | constants give 5 ×; the rest is a firing rate of 0.15 |
| SpikeLM (Xing et al., ICML 2024) | 12.9 × vs BERT-base (T = 1) | analytical BERT at 4.6 pJ | ternary spikes; same pricing |
| SpikeBERT (Lv et al., 2023 preprint) | 27.82 % of BERT's energy | analytical BERT at 4.6 pJ | its own table has more operations than BERT (our reading of its Table 2): the saving is the constants |
| SpikingBERT (Bal & Sengupta, AAAI 2024) | "nearly twice" | same network as an ANN, 4.6 pJ | same pricing |
| SpikingMamba (Huang et al., TMLR 2026) | 4.76 × vs Mamba2 | analytical, FP32 constants | multi-level signed spikes, i.e. integer counts |
| SDLLM (Xu et al., arXiv 2604.16475, 2026) | 6 × vs INT4 (QuaRot), "up to 13 ×" | INT4 model charged 4.6 pJ per integer op (our arithmetic from its Table 2) | op ratio 1.15 ×, energy ratio 6 ×: the 5.1 × of the constants applied to an integer baseline; with INT6 it has 2.3 × more ops and still reports 2.2 × less energy |
| NSLLM (Xu et al., NSR 2025) | 19.8 × energy efficiency, FPGA vs A800 | measured powers | 19.8 is the ratio of dynamic powers (our arithmetic), not energy per token; the deployed model is listed with T = 255 binary steps and more FLOPs than FP32 RWKV-4 (its Table S10) |
| SpikeLLM (Xing et al., ICLR 2025) | 10.79 × fewer bit-operations vs LLaMA-7B | bit-operation counts, no pJ | notes that 1-bit event-driven execution costs more operations than GEMM |

SDLLM is the closest work and the clearest case for the note: its claim
against an integer baseline is mostly the FP32 constants. It is a preprint;
cite the version read.

**Background, verified.** BitNet b1.58 (arXiv 2402.17764: matmuls of integer
additions; 71.4 × less arithmetic energy than FP16 LLaMA at 7 nm) and a4.8
(arXiv 2411.04965: INT4 inputs to QKV and FFN up/gate, INT8 sparsified
elsewhere; full INT4 diverged). Delta networks (Neil et al., ICML 2017) and
sigma-delta networks (O'Connor & Welling, arXiv 1611.02024): the change-based
coding our ΔZ/ΔS engines test. Zero-skipping accelerators: Cnvlutin (ISCA
2016, zero activations), EIE (ISCA 2016) and SCNN (ISCA 2017, zero weights and
activations): the Z engine exists in silicon. Davidson & Furber (Front.
Neurosci. 2021): an SNN wins only below about 1.72 spikes per output.
Dampfhoffer et al. (IEEE TETCI 2022): compares against 16-bit and 8-bit
fixed-point accelerators, no 4-bit, ternary or zero-skipping baselines; the
SNN advantage it finds comes from spike sparsity, not from replacing MACs with
accumulates. Neither compares the same network under both readings.

**Still to check.** Spikformer camera-ready wording (arXiv v2 read); the
OpenReview versions of SpikeGPT and SpikingMamba (behind a challenge); the
venues of SpikeBERT and SDLLM; BrainTransformers (arXiv 2410.14687, not read);
DOIs of Cnvlutin, EIE, SCNN.

## 9. Venues (official pages read 2026-10-31)

| venue | format | deadline |
|---|---|---|
| NICE 2027 (Brighton, 6–8 April 2027) | regular paper, 4–8 pages excluding references | **11 Nov 2026 AoE**; late-breaking 1–2 page abstracts 1 Feb 2027 |
| ICLR 2027 workshops | per workshop | list due 29 Nov 2026; suggested paper deadline 1 Feb 2027 |
| ICBINB | – | no 2027 edition announced |
| ICONS 2027 | – | no call published |
| TMLR | any length | rolling |
| arXiv alone | – | none |

NICE is the neuromorphic audience the note addresses, and its page limit fits.
Eleven days is tight but the measurements exist; tier 3 finishes today.

## 10. Decisions for Gaétan

  1. ~~Format~~: arXiv note, chosen 2026-10-31.
  2. ~~Tier 3~~: approved and run 2026-10-31: every site keeps its verdict on
     WikiText-103; rung 5 reaches 71.8 % of rung 1 there.
  3. ~~Venue~~: arXiv first, chosen 2026-10-31 (not NICE 2027); an ICLR 2027
     workshop to consider once the list is out (29 Nov 2026).
  4. ~~Public repository~~: made public 2026-10-31 (https://github.com/Orolol/SpikeBit).
  5. Author: Gaétan Martin (2026-10-31). Affiliation and contact e-mail still open.
  6. arXiv categories, proposed: cs.NE primary, cross-lists cs.LG and cs.AR. A first submission
     to a category may need an arXiv endorsement.

## 11. arXiv submission

  * `bash writeup/make_arxiv.sh` writes `writeup/build/note.pdf` and
    `writeup/build/arxiv.tar.gz` (`note.tex`, `note.bbl`, `figures/rho_star.pdf`). arXiv
    compiles with pdflatex and does not run BibTeX, hence the `.bbl` and an ASCII-only source;
    the local build uses XeTeX (tectonic 0.17.0), so the pdflatex build is first checked by
    arXiv's own preview.
  * Abstract for the submission form (plain text, 1188 characters, limit 1920):
    `writeup/arxiv_abstract.txt`.
