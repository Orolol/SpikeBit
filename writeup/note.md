<!--
Superseded by writeup/note.tex (2026-10-31), the LaTeX source of the arXiv note; this
file is revision 3 as Gaétan reviewed it and is no longer updated.
Draft of the arXiv note (DECISIONS #24; plan in writeup/plan.md). Every number is in
results/ with its command and commit.
Revision 3 (2026-10-31) applies Gaétan's review: cost models instead of a
hardware bound, design (a) / (b), the firing-rate argument corrected, scope
narrowed to unrolling integers into unit spikes, SpikeLLM positioning, three
levels of exactness, membrane moved to an appendix, configurations instead of
rung numbers.
-->

# Unrolling Integer Activations into Unit Spikes Does Not Save Work: An Exact Accounting on a Ternary Language Model

*Author(s) TBD*

## Abstract

Spiking language models report large energy savings, usually by pricing a
spike accumulation at 0.9 pJ against 4.6 pJ for a 32-bit floating-point
multiply-accumulate. Several of them run integer activations as trains of unit
spikes. We measure what that unrolling costs on a model where it is exact:
RWKV-4-169M with ternary weights and 4-bit activations at every projection
input, whose spike engine produces logits bit-identical to its integer engine.
On the projections the usual formula credits spikes with 3.3 × less energy; this
factors into removed multiplications (5.1 ×) and skipped zeros (1.8 ×), both
available to an integer engine with ternary weights, times the cost of the
unrolling (0.36). Counted directly, unit spikes need 1.7–3.4 times the
accumulations of a zero-skipping integer engine at every site, beat a dense
engine only at the two sparse sites, and gain nothing from delta coding; the
verdicts hold on TinyStories and WikiText-103. Under every event-driven cost
model we could price, spikes cost more than the integer engine; where weights
are held across a source's spikes, the answer reduces to the cost of a ±1
increment, which published figures do not settle.

## 1. Introduction

A growing line of work builds language models from spiking neurons and
reports large energy savings: 32.2 × for SpikeGPT [Zhu et al. 2024], 12.9 ×
for SpikeLM [Xing et al. 2024], 4.76 × for SpikingMamba [Huang et al. 2026],
6 × against an INT4 model for SDLLM [Xu et al. 2026]. Most of these figures
come from one formula [Zhou et al. 2023; Yao et al. 2023]: the spiking
network's synaptic operations, firing rate × time steps × the layer's
multiply-accumulates (MACs), are priced at 0.9 pJ, and the reference network's
MACs at 4.6 pJ (4.5 pJ in SpikeGPT). Both are 32-bit floating-point operations
from Horowitz's 45 nm table [Horowitz 2014]: an addition, and a multiplication
plus addition.

These models differ in what they send per activation (Table 1). Some send one
binary or ternary spike per step; then each non-zero activation is one event and
the spike count equals the non-zero count. Some send an integer in one graded
event, which is an integer engine. Others unroll an integer into several unit
spikes over virtual steps; then a count of 3 becomes three events.

**Table 1.** What the cited spiking LMs send per activation.

| model | per activation | in the terms of Section 4 |
|---|---|---|
| SpikeGPT | binary spikes in the projections, one step per token; ReLU² values as graded spikes | one event per non-zero value |
| SpikeLM | ternary spikes, one step (headline) or four | one step: one event per non-zero value; four: unrolled |
| SpikingMamba | signed multi-level spikes, one step (headline row) | a graded event, or unrolled, depending on the hardware |
| SDLLM | integer counts unrolled into bidirectional spikes over 8 steps | unrolled |
| NSLLM [Xu et al. 2025] | integer activations expanded into binary spikes over virtual steps | unrolled |
| SpikeLLM [Xing et al. 2025] | multi-bit neurons, also evaluated expanded into 1-bit spikes | both, compared |

This note is about the unrolling: given the same integer activations, does
sending each as unit spikes save work over an integer engine? Ternary weights
already turn multiplications into additions [Ma et al. 2024], and accelerators
already skip zero activations [Albericio et al. 2016; Han et al. 2016;
Parashar et al. 2017]. SpikeLLM already finds, in bit-operation counts, that
expanding its multi-bit neurons into 1-bit spikes costs more operations. We
make the comparison exact and site by site, on a ternary model, against three
integer engines, and look at which cost models could reverse it.

The question has an exact answer when the unrolling is exact, because then the
spike engine and the integer engine compute the same logits and differ only in
how many accumulations they perform and what each costs. We build such a model
and:

  * count the accumulations of the unit-spike reading and of three integer
    engines (dense, zero-skipping, delta coding between tokens) on the same
    weights and inputs, on two corpora (Section 4);
  * decompose the spiking literature's energy formula on the same model into
    what spikes contribute and what any integer engine already has
    (Section 5);
  * set out which digital cost models make the unrolling pay and which do not
    (Section 6).

The verdict bands of Section 4 and the cost analysis were fixed in a decision
log kept with the code before the dense, delta and hardware analyses were
run; the zero-skipping ratio $R$ was already known from an earlier
measurement on the same checkpoints.

## 2. Setting

**Model.** RWKV-4-169M [Peng et al. 2023]: 12 blocks of width 768, each with
a time mix (the WKV recurrence) and a channel mix (a relu² MLP of width 3072).
Each block first mixes every token with the previous one,
$x_t' = \mu \odot x_t + (1 - \mu) \odot x_{t-1}$ ("token shift"), separately
for each projection input ($x_r$, $x_k$, $x_v$). The 84 projections of the
blocks become ternary BitLinear layers (92.0 M parameters): weights in
$\{-1, 0, +1\}$ with one scale per group of 128 inputs, as in BitNet b1.58
[Ma et al. 2024]. Embeddings, the LM head (38.6 M MACs per token), layer
norms, the WKV recurrence and the sigmoid gates stay in floating point.

**Neurons.** Our integer LIF neuron (FAI-LIF), after NSLLM, quantizes its
input to a 4-bit count $q$ per token on a learned grid (LSQ step size
[Esser et al. 2020], clipped straight-through gradient) and outputs
$s \cdot q$; signed sites use $q \in [-8, 7]$, the channel-mix relu² output
$q \in [0, 15]$. The model has 120 of them: one on each of the 7 projection
inputs of each block, plus the WKV inputs $k$, $v$ and the channel-mix output.
They are stateless; Appendix A tests a membrane.

**Training and evaluation.** Fine-tuning for 10 k steps of 16 × 128 tokens
(20.5 M tokens); quantized models add a KL term to the full-precision
fine-tuned model. Perplexity on the first 1 M tokens of the last 1 % of the
corpus, in 128-token windows each started from a fresh state; two seeds per
configuration, on disjoint text.

| configuration (mean of 2 seeds) | TinyStories | WikiText-103 |
|---|---|---|
| full precision, fine-tuned (reference) | 4.986 | 25.44 |
| ternary weights, full-precision activations | 5.157 | – |
| ternary weights, 4-bit LIF counts at every projection input (studied) | **5.391** (1.081 ×) | **35.41** (1.392 ×) |

The converted RWKV-4-169M before fine-tuning reproduces the original (13.98 on
TinyStories). The studied model holds well on TinyStories [Eldan & Li 2023]
and much less well on WikiText-103 [Merity et al. 2016] at this budget; the
counts below do not depend on that gap.

## 3. An exact unrolling

A signed count $q \in [-8, 7]$ is unrolled into $|q|$ spikes of sign
$\mathrm{sign}(q)$ over $T = 8$ virtual steps; an unsigned count
$q \in [0, 15]$ into $q$ unit spikes over $T = 15$. We call a site
*spike-native* when its consumer is a ternary BitLinear: the 84 projection
inputs. The WKV inputs (whose $e^k$ needs the decoded value) and the
channel-mix output (multiplied by a floating-point gate) are not; they are the
same in every engine and are left out of the counts.

Exactness holds at three levels, which we keep apart:

  1. **Integer accumulations.** Through a ternary consumer, the accumulations
     of the spikes, summed over steps, equal those of $q$ bit for bit
     (checked on real counts at every spike-native site of blocks 0 and 11, and
     on random counts in unit tests).
  2. **The whole model, integer against spike engine.** Running all 84
     consumers as an integer engine (integer accumulation per weight group,
     then the scales applied once) and as a spike engine (the same, with the
     accumulations taken over the spikes) gives bit-identical logits on 4 k
     tokens, for both seeds.
  3. **Against the floating-point training graph.** Both engines match it only
     up to float rounding, which a downstream LIF can turn into a different
     count: cross-entropy within 0.05 %, 98–99 % of argmaxes equal.

The accounting compares levels 1–2: two engines computing the same function.

The change between consecutive tokens, $\Delta q_t = q_t - q_{t-1}$, lies in
$[-15, 15]$ and is unrolled the same way on a signed 5-bit grid ($T = 16$). A
consumer that keeps its previous output computes
$y_t = y_{t-1} + W \Delta q_t = W q_t$ exactly.

## 4. Counting accumulations

For a ternary consumer $W$ (out × in) and the counts $q_i$ of one token, let
$c_i = \mathrm{nnz}(W_{:,i})$. Weight zeros are static and every engine below
skips them (an engine that does not multiplies all counts by 1.47). The
accumulations per token:

| engine | accumulations | operand |
|---|---|---|
| D, dense activations | $\sum_i c_i$ | 4-bit count |
| Z, zero activations skipped | $\sum_i \mathbb{1}[q_i \neq 0]\, c_i$ | 4-bit count |
| S, unit spikes | $\sum_i \lvert q_i \rvert\, c_i$ | one spike |
| ΔZ, ΔS | Z, S on $\Delta q_t$ | 5-bit count, one spike |

**Cost per accumulation.** Let $a$ be the cost of adding a 4-bit operand into
the output accumulator and $\rho a$ that of a ±1 increment of the same
accumulator. Counting additions only, the unit spikes are cheaper than engine X
when $\rho < \rho^* = \mathrm{ops}_X/\mathrm{ops}_S$. Section 6 adds the cost
$f$ of reading the weight. Two pricings bound $\rho$: both operands enter the
same 14–17-bit adder and Horowitz's table prices additions by width, giving
$\rho = 1$; priced by operand width instead, an assumption favourable to spikes,
$\rho = 1/4$. The bands: $\rho^* > 1$, spikes cheaper at any $\rho$ in that
range; $\rho^* < 1/4$, dearer at any; in between, the answer depends on the
hardware.

**The sign is trivial, the size is not.** $|q_i| \geq \mathbb{1}[q_i \neq 0]$,
so $R = S/Z \geq 1$ for any network, with equality when every non-zero count is
±1, as with one-step binary or ternary spikes. What a measurement adds is how
far above 1 unrolling puts it: the mean count of a non-zero activation is 1.7 at
the sparse time-mix output, 2.2 at the relu² output, 2.6 to 3.1 at the dense
projection inputs.

**Results** on TinyStories (first 256 windows of the evaluation tail, 32 k
tokens; the two seeds agree to within 0.06 in $\rho^*$, except 14.3 against
15.3 at the relu² site against D):

| site (pooled over 12 blocks) | non-zero | $E\lvert q\rvert$ | $\rho^*$ vs D | $\rho^*$ vs Z ($=1/R$) | $\rho^*$ ΔS vs ΔZ | ΔZ / Z |
|---|---|---|---|---|---|---|
| channel-mix relu² output → $W_v$ | 3 % | 0.07 | 14.3–15.3 | 0.45–0.46 | 0.50–0.51 | 1.56–1.58 |
| time-mix output → $W_o$ | 17–18 % | 0.30 | 3.3 | 0.57–0.58 | 0.62 | 1.25 |
| time-mix input $x_r$ → $W_r$ | 83 % | 2.16–2.17 | 0.46 | 0.38 | 0.37 | 1.01 |
| time-mix input $x_k$ → $W_k$ | 84 % | 2.23–2.24 | 0.44–0.45 | 0.38 | 0.35 | 1.00 |
| time-mix input $x_v$ → $W_v$ | 87 % | 2.71–2.72 | 0.37 | 0.32 | 0.27 | 1.01 |
| channel-mix input $x_k$ → $W_k$ | 86 % | 2.52–2.55 | 0.39–0.40 | 0.34 | 0.34–0.35 | 0.98 |
| channel-mix input $x_r$ → $W_r$ | 82–84 % | 2.13–2.19 | 0.46–0.47 | 0.38–0.39 | 0.38 | 0.99 |
| **pooled** | | | **0.65** | **0.36** | **0.35** | **1.01** |

Figure 1 shows the same numbers on a log scale.

  * **Against Z, unit spikes always need more accumulations:** $R$ = 1.7 to
    3.1 per site (up to 3.4 at twice the training budget), 2.8 pooled. A ±1
    increment would have to cost between 0.32 and 0.58 of a 4-bit addition.
  * **Against D, the verdict follows sparsity.** At the two sparse sites
    spikes are cheaper at any $\rho \leq 1$. At the five dense input sites,
    which carry 61.5 % of the projection work, and pooled ($\rho^* = 0.65$),
    they are not. Z gets the sparse sites' benefit without the unrolling.
  * **Delta coding saves nothing.** At the dense sites $\Delta q$ is non-zero
    as often as $q$; at the sparse sites more often, since a unit that turns on
    and off makes two changes. The bands are the same as without it, except
    one site of the model trained at twice the budget, which becomes "dearer".

**On WikiText-103**, retrained and counted the same way, no site changes band in
any of the three comparisons: pooled $\rho^*$ 0.68–0.69 against D, 0.37–0.38
against Z ($R$ = 2.66–2.69), 0.36 for ΔS against ΔZ. The model with 4-bit
counts only at the neuron sites (the WKV inputs, the two sparse sites and the
channel-mix output) gives a pooled $\rho^*$ of 8.8–8.9 against D and 0.53
against Z.

**Sparser training.** The quantizers were not trained for sparsity. A
firing-rate penalty would lower the counts: it can turn a 3 into a 1, which
lowers S and leaves Z unchanged, so it can lower $R$ and raise $\rho^*$
against Z, towards 1 but never above it ($S \geq Z$ holds for any counts). It
would also change perplexity; neither effect is measured here.

## 5. What the spiking energy formula counts

Applied to the studied model as Spikformer and Spike-driven Transformer apply
it (synaptic operations = $E|q|$ × MACs at 0.9 pJ, against the same MACs at
4.6 pJ for the full-precision network), the formula gives the spiking model
3.3 × less energy on the projections (3.5 × on WikiText-103), 2.0 × with the
full-precision LM head counted on both sides. On the spike-native layers the
ratio factors exactly, with $M_s$ the MACs and $P_s$ the non-zero fraction of
site $s$:

$$\frac{E_{\mathrm{ANN}}}{E_{\mathrm{SNN}}}
= \frac{4.6}{0.9} \times \frac{\sum_s M_s}{\sum_s P_s M_s}
\times \frac{\sum_s P_s M_s}{\sum_s E|q|_s M_s},$$

which on TinyStories is 5.11 (multiplications removed) × 1.82 (zeros skipped)
× 0.36 (unrolling). The first two factors are what ternary weights and zero
skipping give any engine: an integer engine that skips zeros, priced with the
same constants, gets 5.11 × 1.82 = 9.3 × (13.7 × if it also skips weight
zeros, which the formula does not count). The third is the unrolling's own
contribution, and it is a cost. For one-step binary or ternary spikes the third
factor is 1, and the whole saving is the first two.

The constants are FP32 operations. An integer addition costs 0.03 to 0.1 pJ in
the same table (8 to 32 bits), a further 9–30 × that integer and spike engines
could both claim. When the reference network is itself integer, charging its
operations the FP32 MAC constant hands the spiking side the 5.1 × of the
constants: SDLLM reports 6 × less energy than an INT4 model while performing
1.15 × fewer operations (our arithmetic from its Table 2).

## 6. Cost models

Whether the unrolling pays depends on what an accumulation costs besides the
addition, above all on how often the weight is read. Two designs bracket it.

**(a) One event per spike, weight read per event**, as event-driven chips
process spikes. Each accumulation reads the weight it adds, at cost $f$. From
the smallest SRAM in Horowitz's table (10 pJ per 64-bit read), 32 packed
ternary weights cost $f \approx 0.31$ pJ each, 3 to 10 times an addition
($a$ = 0.03–0.1 pJ). Unit spikes are cheaper than Z only if
$\rho_{\mathrm{eff}} = (\rho a + f)/(a + f) < 1/R$; here
$\rho_{\mathrm{eff}}$ is 0.82 to 0.93 even at $\rho = 1/4$. Davidson and
Furber's per-event model from SpiNNaker 2 post-layout figures [2021] reaches
the same: with $E$ the cost of an addition, a weight read costs about $5E$, a
spike event $7E$, a multiply-accumulate event $12E$; without the
multiplication, which ternary weights remove, an integer event also costs $7E$.
Published per-event energies (23.6 pJ minimum, simulated, on Loihi
[Davies et al. 2018]; 26 pJ with static power on TrueNorth
[Merolla et al. 2014]; 10–20 pJ on SpiNNaker 2 [Höppner et al. 2021]) are
100 to 900 times a 45 nm addition. They come from different chips and do not
give the ratio between the two engines on any one of them, but they show that
an event's cost is dominated by memory, routing and the neuron update rather
than by the width of what it carries. Loihi 2 sends integer payloads in one
"graded" spike at what Intel calls negligible extra energy over a binary spike
[Intel 2021], a qualitative indication that one event per count costs about
the same as one event per unit.

**(b) Weight held across a source's spikes.** If the engine reads the weight
once and applies all $|q_i|$ unit spikes of input $i$ from a register, the read
is paid once per non-zero input in both engines ($Z f + S \rho a$ against
$Z(f + a)$) and drops out. Unit spikes are then cheaper than Z if and only if
$\rho < 1/R$: a ±1 increment of the 14–17-bit accumulator must cost less than
0.29 to 0.58 of adding a 4-bit operand into it. Horowitz prices additions by
width ($\rho = 1$, no); pricing by operand width ($\rho = 1/4$, yes) is an
assumption. We found no published figure that settles it.

| cost model | $\rho$ or $\rho_{\mathrm{eff}}$ | vs Z | vs D, sparse sites | vs D, input sites |
|---|---|---|---|---|
| additions only, priced by operand width | 1/4 | cheaper | cheaper | cheaper |
| additions only, priced by width (Horowitz, 45 nm) | 1 | dearer | cheaper | dearer |
| (a) event per spike, weight read from SRAM | 0.82–1 | dearer | cheaper | dearer |
| (a) per-event model, SpiNNaker 2 figures | ≈ 1 | dearer | cheaper | dearer |
| (b) weight held across a source's spikes | adder $\rho$ | open | cheaper | open |

Under every design (a) model we could price, unit spikes cost more than a
zero-skipping integer engine at every site. Under design (b) the verdict is the
cost of a ±1 increment, which these figures leave open. These are cost models,
not a bound on digital hardware.

**The whole model.** The projections are a small part of a token's energy.
With the LM head in FP16 (58 µJ per token at 1.5 pJ per MAC), their additions
cost 1.0–3.4 µJ in Z and 2.9–9.6 µJ in S at $\rho = 1$, so the reading moves a
token's arithmetic by −2 % to +10 %; under design (a) weight reads add 10.7 µJ
(Z) or 30 µJ (S). If the 23 MB of ternary weights come from DRAM at batch 1,
moving them costs 7 to 125 times the arithmetic. These are orders of magnitude,
not the cost model of a built engine. The spiking LMs cited above also keep a
floating-point head.

## 7. Related work

**Spiking language models.** SpikeGPT, SpikeLM, SpikeBERT [Lv et al. 2023],
SpikingMamba and SDLLM estimate energy with the 0.9 / 4.6 pJ constants (4.5 pJ
in SpikeGPT) against a different, full-precision or INT4 reference network;
SpikingBERT [Bal & Sengupta 2024] uses the same constants against the same
architecture run without spikes. SpikeLLM counts bit operations instead and
finds that expanding its multi-bit neurons into 1-bit spikes costs more
operations than the multi-bit execution; our accounting is that comparison made
exact and site by site, with ternary weights and integer baselines. NSLLM,
whose integer-to-binary inference our neuron follows, reports a 19.8 × ratio of
dynamic power between an FPGA and a GPU.

**Fair accounting of spiking networks.** Davidson and Furber [2021] show that
on digital hardware a rate-coded spiking network wins only below about 1.7
spikes per output, because weight and state memory accesses dominate.
Dampfhoffer et al. [2022] include memory accesses and compare with 16- and
8-bit accelerators, including zero-skipping ones, and attribute any spiking
advantage to spike sparsity rather than to replacing MACs with accumulations;
Lemaire et al. [2022] also find memory dominant. None compares one network
under both executions, or against a ternary integer engine that skips zeros.

**Integer engines.** Ternary weights remove the multiplications
[Ma et al. 2024]. BitNet a4.8 [Wang et al. 2024] runs 4-bit activations on
them for the attention and FFN input projections, keeping the others at 8 bits
with sparsification; our model takes every projection input to 4 bits.
Cnvlutin, EIE and SCNN skip zero activations or weights in hardware. Delta
networks [Neil et al. 2017] and sigma-delta networks
[O'Connor & Welling 2016] propagate only changes; here the changes are as dense
as the values.

## 8. Limits

  * One model (RWKV-4-169M) at one size, two corpora; counts on 32 k tokens
    per checkpoint. The conversion's quality depends on the corpus (1.08 × on
    TinyStories, 1.39 × on WikiText-103 at the same budget); the accounting
    does not.
  * The result concerns unrolling integer counts into unit spikes through
    stateless consumers. One-step binary or ternary spikes have $R = 1$ and
    keep only the first two factors of Section 5; graded spikes are the integer
    engine. A network whose neurons integrate spikes over time steps computes a
    function of spike timing and has no integer twin of this kind.
  * Digital cost models only; analog or in-memory substrates price
    accumulation differently. Design (b) rests on the cost of a ±1 increment,
    not measured here.
  * Quantizers not trained for sparsity (Section 4).

## 9. Conclusion

When a quantized language model's integer activations are unrolled into unit
spikes, the same network runs as an integer engine, and the comparison is about
work rather than about models. Here the unrolling never needs fewer
accumulations than an integer engine that skips zeros, wins against a dense
engine only where activations are sparse, and gains nothing from delta coding,
on two corpora. Under event-driven cost models that read a weight per spike it
costs more; where weights are held across a source's spikes it can pay only if
a ±1 increment costs well under a 4-bit addition, which we could not establish.
The savings reported for such models come from removing multiplications and
skipping zeros, which an integer engine with ternary weights also has, priced
with FP32 constants. If unit spikes are used at all in such a model they
belong at the sparse sites, and there an integer engine that skips zeros needs
$R$ times fewer accumulations.

## Appendix A: the membrane

A LIF neuron differs from a quantizer by its membrane, the residual of one
token carried with leak $\beta$ into the next. On the model with 4-bit counts
at the neuron sites only (PPL 5.328 on TinyStories), we trained the same
configuration with a membrane, its gradient through time detached because it
explodes (×30 per block backward), so $\beta$ learns from its effect within one
token only. Started at $\beta = 0.9$ with a small learning rate, $\beta$ barely
moved (0.87–0.90) and the perplexity was 5.543 (+4.0 %); trained at the
model's learning rate, $\beta$ spread over 0.03–0.95 and the perplexity was
5.523 (+3.7 %); the pre-registered threshold to keep the membrane was 5.062.
Setting $\beta = 0$ at evaluation changes either model's perplexity by −0.02 %
to −0.4 %. No benefit was observed under our detached-state training protocol;
this does not test a membrane trained through time.

## Appendix B: reproduction

Code, decision log (DECISIONS #15–#24) and results: [repository URL TBD].
Repository names of the configurations: rung 1 (full precision), rung 2
(ternary weights), rung 3 (4-bit counts at the neuron sites), rung 5 (studied
model), rungs 4b and 4c (membrane).

  * Training and perplexities: `results/stage_a_ladder.md` (commands, commits,
    checkpoints); WikiText-103 runs: `colab/ladder_general.py`.
  * Counts: `CUDA_VISIBLE_DEVICES= uv run python scripts/spike_accounting.py
    rung5_a4_s0 rung5_a4_s1 ...` (add `CORPUS=wikitext103` for WikiText-103);
    tables by `scripts/accounting_tables.py`; Figure 1 by
    `scripts/accounting_figure.py`.
  * Exactness: unit tests in `src/spiking_ternary_lm/neurons/tests/test_binary_lif.py`
    (spike and delta readings through a ternary BitLinear); whole-model check
    `scripts/spike_logits_check.py`.
  * Cost models and their sources: `results/hardware_cost_note.md`.
  * Membrane: `scripts/_beta_stats.py`, `scripts/_diagnose_state_regime.py`.

## References

Albericio, J. et al. Cnvlutin: Ineffectual-neuron-free deep neural network
computing. ISCA 2016.

Bal, M., Sengupta, A. SpikingBERT: Distilling BERT to train spiking language
models using implicit differentiation. AAAI 2024. arXiv:2308.10873.

Dampfhoffer, M., Mesquida, T., Valentian, A., Anghel, L. Are SNNs really more
energy-efficient than ANNs? An in-depth hardware-aware study. IEEE TETCI,
2022. doi:10.1109/TETCI.2022.3214509.

Davidson, S., Furber, S. B. Comparison of artificial and spiking neural
networks on digital hardware. Frontiers in Neuroscience 15:651141, 2021.
doi:10.3389/fnins.2021.651141.

Davies, M. et al. Loihi: A neuromorphic manycore processor with on-chip
learning. IEEE Micro 38(1), 2018. doi:10.1109/MM.2018.112130359.

Eldan, R., Li, Y. TinyStories: How small can language models be and still
speak coherent English? arXiv:2305.07759, 2023.

Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., Modha, D. S.
Learned step size quantization. ICLR 2020. arXiv:1902.08153.

Han, S. et al. EIE: Efficient inference engine on compressed deep neural
network. ISCA 2016. arXiv:1602.01528.

Höppner, S. et al. The SpiNNaker 2 processing element architecture for hybrid
digital neuromorphic computing. arXiv:2103.08392, 2021.

Horowitz, M. Computing's energy problem (and what we can do about it). ISSCC
2014, pp. 10–14. doi:10.1109/ISSCC.2014.6757323.

Huang, Y. et al. SpikingMamba: Towards energy-efficient large language models
via knowledge distillation from Mamba. TMLR, 2026. arXiv:2510.04595.

Intel Labs. Taking neuromorphic computing to the next level with Loihi 2.
Technology brief, 2021.

Lemaire, E. et al. An analytical estimation of spiking neural networks energy
efficiency. arXiv:2210.13107, 2022.

Lv, C. et al. SpikeBERT: A language Spikformer learned from BERT with
knowledge distillation. arXiv:2308.15122, 2023.

Ma, S. et al. The era of 1-bit LLMs: All large language models are in 1.58
bits. arXiv:2402.17764, 2024.

Merity, S., Xiong, C., Bradbury, J., Socher, R. Pointer sentinel mixture
models. arXiv:1609.07843, 2016.

Merolla, P. A. et al. A million spiking-neuron integrated circuit with a
scalable communication network and interface. Science 345(6197), 2014.
doi:10.1126/science.1254642.

Neil, D., Lee, J. H., Delbruck, T., Liu, S.-C. Delta networks for optimized
recurrent network computation. ICML 2017. arXiv:1612.05571.

O'Connor, P., Welling, M. Sigma delta quantized networks. arXiv:1611.02024,
2016.

Parashar, A. et al. SCNN: An accelerator for compressed-sparse convolutional
neural networks. ISCA 2017. arXiv:1708.04485.

Peng, B. et al. RWKV: Reinventing RNNs for the Transformer era.
arXiv:2305.13048, 2023.

Wang, H., Ma, S., Wei, F. BitNet a4.8: 4-bit activations for 1-bit LLMs.
arXiv:2411.04965, 2024.

Xing, X. et al. SpikeLM: Towards general spike-driven language modeling via
elastic bi-spiking mechanisms. ICML 2024. arXiv:2406.03287.

Xing, X. et al. SpikeLLM: Scaling up spiking neural network to large language
models via saliency-based spiking. ICLR 2025. arXiv:2407.04752.

Xu, H., Qiu, X., et al. (NSLLM) Neuromorphic spike-based large language model.
National Science Review 13(4):nwaf551, 2025. doi:10.1093/nsr/nwaf551.

Xu, H., Qiu, X., et al. (SDLLM) Spike-driven large language model.
arXiv:2604.16475, 2026.

Yao, M. et al. Spike-driven Transformer. NeurIPS 2023. arXiv:2307.01694.

Zhou, Z. et al. Spikformer: When spiking neural network meets Transformer.
ICLR 2023. arXiv:2209.15425.

Zhu, R.-J., Zhao, Q., Li, G., Eshraghian, J. K. SpikeGPT: Generative
pre-trained language model with spiking neural networks. TMLR, 2024.
arXiv:2302.13939.
