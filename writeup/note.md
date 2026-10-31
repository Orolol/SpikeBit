<!--
Draft of the arXiv note (DECISIONS #24; plan in writeup/plan.md). Markdown with
LaTeX math, to convert once a TeX toolchain is chosen. Every number is in
results/ with its command and commit. [TIER 3] marks what waits for the
general-text runs. References marked (to verify) were not read for this draft.
-->

# Reading a Ternary Language Model as Spikes Does Not Save Work: An Exact Accounting Against Integer Engines

*Author(s) TBD*

## Abstract

Spiking language models are often motivated by energy: activations become
binary events, multiplications become additions, and work happens only where
neurons fire. We test that motivation on a network where the comparison can
be made exactly. We convert RWKV-4-169M to ternary weights and 4-bit
activations produced by stateless leaky integrate-and-fire (LIF) neurons, at
1.08 × the perplexity of the full-precision fine-tuned model on TinyStories.
Each 4-bit activation can be read exactly as up to 8 or 15 ±1 spikes, so a
spiking engine and an integer engine compute the same logits; only the work
differs. Counted on the same weights and inputs, the spike reading needs 1.7
to 3.1 times the accumulations of an integer engine that skips zero
activations, at every site of the model. It beats an engine that does not
skip zeros only at the two sparse sites. Coding the change between
consecutive tokens saves nothing. The energy formula common in the spiking
literature, 0.9 pJ per accumulation against 4.6 pJ per 32-bit floating-point
multiply-accumulate, credits the spikes with a 3.3 × saving on this model; it
factors into removed multiplications and skipped zeros, both available to an
integer engine, times a spike-count penalty. Published energies of digital
adders, of weight fetches and of neuromorphic chips put a spike accumulation
at no less than 0.82 of an integer one, where spikes lose. A membrane carried
across tokens, with its leak fixed or learned, did not improve the model.
[TIER 3: one sentence on WikiText-103.]

## 1. Introduction

A growing line of work builds language models from spiking neurons and
reports large energy savings: 32.2 × for SpikeGPT [Zhu et al. 2024], 12.9 ×
for SpikeLM [Xing et al. 2024], 4.76 × for SpikingMamba [Huang et al. 2026],
6 × against an INT4 model for SDLLM [Xu et al. 2026]. Most of these figures
come from one formula: the spiking network's synaptic operations, counted as
firing rate × time steps × the layer's multiply-accumulates (MACs), are priced
at 0.9 pJ, and the reference network's MACs at 4.6 pJ. Both constants come
from Horowitz's 45 nm table [Horowitz 2014] and are 32-bit floating-point
operations: 0.9 pJ is an FP32 addition, 4.6 pJ an FP32 multiplication plus
addition. The reference is usually a different, full-precision network.

For a quantized model the relevant question is narrower. Low-bit integer
networks already replace multiplications with additions when the weights are
ternary [Ma et al. 2024], and accelerators already skip zero activations
[Albericio et al. 2016; Han et al. 2016; Parashar et al. 2017]. What does
reading the same integer activations as spikes add? The question has an
exact answer when the spike reading is exact, because then the spiking and
the integer engine compute the same function and differ only in how many
accumulations they perform and what each one costs.

SpikeBit is such a case. We convert RWKV-4-169M [Peng et al. 2023 (to
verify)] to ternary weights with 4-bit activations produced by stateless LIF
neurons at every input of every ternary projection. A 4-bit count read as ±1
spikes through a ternary layer gives the integer result bit for bit. We then:

  * count, on the same weights and inputs, the accumulations of the spike
    reading and of three integer engines: dense, zero-skipping, and coding
    changes between tokens (Section 4);
  * decompose the spiking literature's energy formula on the same model into
    what spikes contribute and what any integer engine already has
    (Section 5);
  * place published hardware figures against the break-even cost each
    comparison requires (Section 6);
  * test whether a membrane carried across tokens, the part of the LIF
    neuron that a stateless quantizer lacks, improves the model (Section 7).

The verdicts were fixed before the counts were computed, in a decision log
kept with the code. The result is negative for spikes on digital hardware:
against an integer engine that skips zeros they never need fewer
accumulations, and every hardware figure we could source prices a spike
accumulation too high for them to win.

## 2. Setting

**Model.** RWKV-4-169M (12 blocks, width 768, channel-mix width 3072). The 84
projections of the blocks (time-mix $W_r, W_k, W_v, W_o$; channel-mix
$W_k, W_v, W_r$) become ternary BitLinear layers, 92.0 M parameters, with
weights in $\{-1, 0, +1\}$ and one scale per group of 128 inputs, as in
BitNet b1.58 [Ma et al. 2024]. Embeddings, the LM head (38.6 M MACs per
token), layer norms, the WKV recurrence and the sigmoid gates stay in
floating point.

**Neurons.** Following NSLLM [Xu et al. 2025], an integer LIF neuron
(FAI-LIF; <!-- check the name NSLLM uses --> single time step per token)
quantizes its input to a 4-bit integer count $q$ on a learned grid (LSQ step [Esser et al. 2020 (to
verify)], clipped straight-through gradient) and outputs $s \cdot q$. Signed
sites use $q \in [-8, 7]$; the channel-mix relu² output uses
$q \in [0, 15]$. In the final model (rung 5 below) there are 120 LIF sites
per model: one on each of the 84 projection inputs and 36 more on the WKV
inputs and the channel-mix output. They are stateless (leak $\beta = 0$);
Section 7 tests a membrane.

**Training and evaluation.** Fine-tuning on 20.5 M tokens of TinyStories
[Eldan & Li 2023 (to verify)] (10 k steps of 16 × 128 tokens); the
quantized models add a KL term to the full-precision fine-tuned model (rung 1,
seed 0). Perplexity on the first 1 M
tokens of the last 1 % of the corpus, in 128-token windows with a fresh state
each. Two seeds per configuration on disjoint text; a difference counts only
above max(2 × the seed spread, 5 % of perplexity).

**The ladder.** Each change was pre-registered with its threshold:

| rung | model | PPL (2 seeds, mean) | vs rung 1 |
|---|---|---|---|
| 0 | converted teacher, no quantization | 13.98 (= RWKV-4-169M) | – |
| 1 | full precision, fine-tuned (reference) | 4.986 | 1 |
| 2 | + ternary weights | 5.157 | 1.034 |
| 3 | + 4-bit LIF outputs at the 60 neuron sites | 5.328 | 1.069 |
| 4b | rung 3 + membrane, leak fixed near 0.9 | 5.543 | 1.112 |
| 4c | rung 3 + membrane, leak learned | 5.523 | 1.108 |
| 5 | rung 3 + 4-bit LIF on every projection input | **5.391** | 1.081 |

Rung 5 is the model studied below: every input of every ternary projection is
a 4-bit LIF count. At twice the training budget the ratio rung 1 / rung 5
moves from 92.5 % to 94.3 %. [TIER 3: WikiText-103 rows.]

## 3. An exact spike reading

A signed count $q \in [-8, 7]$ is read as $|q|$ spikes of sign
$\mathrm{sign}(q)$ over $T = 8$ virtual steps; an unsigned count
$q \in [0, 15]$ as $q$ unit spikes over $T = 15$. For a consumer that is
linear in its input, the sum over steps of its response to the spikes equals
its response to $q$. For a ternary BitLinear the per-step responses are
integer additions and the equality is exact, scale included. We checked it on
real counts at every spike-native site of blocks 0 and 11, and on random
counts in unit tests.

A site is spike-native when its consumer is a ternary BitLinear: at rung 5,
the 84 projection inputs (the five token-shift mixes of each block, the
time-mix output, the relu² output). The LIF outputs feeding the WKV
(time-mix $k$, $v$, whose $e^k$ needs the decoded value) and the channel-mix
output, multiplied by a floating-point sigmoid gate, are not, and are left out of the counts below; they are
the same in every engine.

The same holds for the change between consecutive tokens,
$\Delta q_t = q_t - q_{t-1} \in [-15, 15]$, read as ±1 spikes over $T = 16$:
a consumer that keeps its previous output computes
$y_t = y_{t-1} + W \Delta q_t = W q_t$ exactly.

## 4. Counting accumulations

For a ternary consumer $W$ (out × in) and the counts $q_i$ of one token, let
$c_i = \mathrm{nnz}(W_{:,i})$. Weight zeros are static, so every engine skips
them. The accumulations per token:

| engine | accumulations | operand |
|---|---|---|
| D, integer, dense activations | $\sum_i c_i$ | 4-bit |
| Z, integer, zero activations skipped | $\sum_i \mathbb{1}[q_i \neq 0]\, c_i$ | 4-bit |
| S, ±1 spikes | $\sum_i \lvert q_i \rvert\, c_i$ | 1 spike |
| ΔZ, ΔS | Z, S on $\Delta q_t$ | 5-bit, 1 spike |

Since $|q_i| \geq \mathbb{1}[q_i \neq 0]$, $R = S/Z \geq 1$: with ternary
weights the integer engine has no multiplication left to remove, and skipping
zeros is open to it as much as to spikes. The question is the cost per
accumulation. If a spike accumulation costs $\rho$ times a 4-bit one into the
same accumulator, spikes are cheaper than engine X when
$\rho < \rho^* = \mathrm{ops}_X / \mathrm{ops}_S$.

**Bands fixed in advance.** Two pricings bound $\rho$. Priced by accumulator
width, as Horowitz prices additions, both are the same 14–17-bit addition and
$\rho = 1$. Priced by operand width, an assumption favourable to spikes,
$\rho = 1/4$. We declared before counting: $\rho^* > 1$ means spikes are
cheaper at any cost in that range, $\rho^* < 1/4$ that they are dearer at any
cost, and anything between depends on the hardware.

**Results** (first 256 windows of the evaluation tail, 32 k tokens; two
seeds, which agree to within 0.06 in $\rho^*$ except at the 3 %-dense relu²
site against D):

| site (rung 5, pooled over 12 blocks) | non-zero | $E\lvert q\rvert$ | $\rho^*$ vs D | $\rho^*$ vs Z ($=1/R$) | $\rho^*$ ΔS vs ΔZ | ΔZ / Z |
|---|---|---|---|---|---|---|
| channel-mix relu² output → $W_v$ | 3 % | 0.07 | 14.3–15.3 | 0.45–0.46 | 0.50–0.51 | 1.56–1.58 |
| time-mix output → $W_o$ | 17–18 % | 0.30 | 3.3 | 0.57–0.58 | 0.62 | 1.25 |
| time-mix input $x_r$ → $W_r$ | 83 % | 2.16–2.17 | 0.46 | 0.38 | 0.37 | 1.01 |
| time-mix input $x_k$ → $W_k$ | 84 % | 2.23–2.24 | 0.44–0.45 | 0.38 | 0.35 | 1.00 |
| time-mix input $x_v$ → $W_v$ | 87 % | 2.71–2.72 | 0.37 | 0.32 | 0.27 | 1.01 |
| channel-mix input $x_k$ → $W_k$ | 86 % | 2.52–2.55 | 0.39–0.40 | 0.34 | 0.34–0.35 | 0.98 |
| channel-mix input $x_r$ → $W_r$ | 82–84 % | 2.13–2.19 | 0.46–0.47 | 0.38–0.39 | 0.38 | 0.99 |
| **pooled** | | | **0.65** | **0.36** | **0.35** | **1.01** |

Figure 2 shows the same numbers on a log scale. Three findings:

  * **Against the zero-skipping engine Z, spikes always need more
    accumulations**: $R$ = 1.7 to 3.1 per site, 2.8 pooled. Every site is in
    the "depends on hardware" band, as $R \geq 1$ dictates; a spike
    accumulation would have to cost under a third to a half of a 4-bit one.
  * **Against the dense engine D, the verdict follows sparsity.** At the two
    sparse sites spikes are cheaper at any $\rho \leq 1$; at the five dense
    input sites, which carry 61.5 % of the projection work, they are not, and
    pooled they are not ($\rho^* = 0.65$). The integer engine that skips zeros
    gets the sparse sites' benefit without the spike count.
  * **Changes between tokens are not sparser than values.** At the dense
    sites $\Delta q$ is non-zero as often as $q$; at the sparse sites more
    often (a unit that turns on and off makes two changes). Delta coding helps
    neither reading, and the bands are the same as without it (one site of
    the ×2-budget model crosses into "dearer").

The same counts on rung 3, whose only spike-native sites are the two sparse
ones, give a pooled $\rho^* = 8.8$–$8.9$ against D and 0.53 against Z; on the
×2-budget rung 5, 0.62 and 0.34. [TIER 3: WikiText-103 counts.]

## 5. What the spiking energy formula counts

Applied to rung 5 as Spikformer and Spike-driven Transformer apply it
[Zhou et al. 2023; Yao et al. 2023] (synaptic operations = $E|q|$ × the
layer's MACs at 0.9 pJ, against the same MACs at 4.6 pJ for the
full-precision network), the formula gives the spiking model 3.3 × less
energy on the projections, 2.0 × with the full-precision LM head counted on
both sides. On the spike-native layers the ratio factors exactly:

$$\frac{E_{\mathrm{ANN}}}{E_{\mathrm{SNN}}}
= \underbrace{\frac{4.6}{0.9}}_{5.11\ \text{multiplications removed}}
\times \underbrace{\frac{\sum_s M_s}{\sum_s P_s M_s}}_{1.82\ \text{zeros skipped}}
\times \underbrace{\frac{\sum_s P_s M_s}{\sum_s E|q|_s M_s}}_{0.36\ \text{spike count}}$$

with $M_s$ the MACs and $P_s$ the non-zero fraction of site $s$. The first two
factors are what ternary weights and zero skipping give any engine: an
integer engine that skips zeros, priced with the same constants, gets
5.11 × 1.82 = 9.3 × (13.7 × if it also skips weight zeros, which the formula
does not count). The third factor is the spike reading's own contribution,
and it is a cost. On rung 3's two sparse sites the formula gives 45 ×, against
84 × for the integer engine.

The constants themselves are FP32 operations. An integer addition costs 0.03
to 0.1 pJ in the same table (8 to 32 bits), so an integer engine has a further
9–30 × on its additions that the formula reserves for no one. When the
reference network is itself integer, pricing its operations at the FP32 MAC
constant hands the spiking side the 5.1 × of the constants: SDLLM reports 6 ×
less energy than an INT4 model while performing 1.15 × fewer operations (our
arithmetic from its Table 2) [Xu et al. 2026].

## 6. Where hardware sits

**Adders.** Both readings add into the same 14–17-bit accumulator; priced by
width, $\rho = 1$ (Section 4).

**Weight fetch.** Each accumulation reads a weight. From the smallest SRAM in
Horowitz's table (10 pJ per 64-bit read), 32 packed ternary weights cost
$f \approx 0.31$ pJ each, 3 to 10 times the addition. A spike engine that
processes each spike as an event reads the weight once per spike; grouping a
source's spikes into one read means holding the count, which is the integer
reading. With the fetch, $\rho_{\mathrm{eff}} = (\rho a + f)/(a + f)$: 0.82 to
0.93 even at $\rho = 1/4$. Davidson and Furber's per-event model from
SpiNNaker 2 post-layout figures [2021] gives the same picture: with $E$ the
cost of an addition, a weight read costs about $5E$, a spike event $7E$ and a
multiply-accumulate event $12E$. Ternary weights remove the multiplication, so
an integer event costs the same $7E$ as a spike event: $\rho \approx 1$.

**Neuromorphic chips** charge per synaptic event, not per bit: 23.6 pJ
(minimum, simulated) on Loihi [Davies et al. 2018], 26 pJ on TrueNorth
[Merolla et al. 2014], about 10 to 20 pJ on SpiNNaker 2 [Höppner et al. 2021],
100 to 900 times a 45 nm integer addition. Loihi 2 carries integer payloads in
a single "graded" spike at, in Intel's words, negligible extra energy over a
binary spike [Intel 2021]: on that chip a 4-bit count can be sent as one
event (engine Z) or as $|q|$ binary events (engine S), and S costs $R$ times
as much.

| hardware point | $\rho$ | vs Z | vs D, sparse sites | vs D, input sites |
|---|---|---|---|---|
| operand-width pricing, no fetch | 1/4 | cheaper | cheaper | cheaper |
| additions priced by width | 1 | dearer | cheaper | dearer |
| additions + weight fetch from SRAM | 0.82–1 | dearer | cheaper | dearer |
| event-driven chips, graded spikes | ≈ 1 | dearer | cheaper | dearer |
| per-event model, SpiNNaker 2 figures | ≈ 1 | dearer | cheaper | dearer |

Only the operand-width assumption without weight fetch lets spikes win
against Z, and no figure we found supports it.

**The whole model.** With the LM head in FP16 (58 µJ per token at 1.5 pJ per
MAC), the projections' additions cost 1.0–3.4 µJ for Z and 2.9–9.6 µJ for S at
$\rho = 1$: the reading moves a token's arithmetic by −2 % to +10 %. Weight
fetch alone adds 10.7 µJ (Z) or 30 µJ (S). If the 23 MB of ternary weights
come from DRAM at batch 1, moving them costs 7 to 125 times the arithmetic.
These are orders of magnitude, not the cost model of a built engine.

## 7. The membrane

A LIF neuron differs from a quantizer by its membrane: the residual of one
token carried, with leak $\beta$, into the next. With the leak started at
0.9 (rung 4b) the model was 4.0 % worse than the stateless rung 3 (5.543 vs
5.328, against a pre-registered retention threshold of 5.062); but $\beta$
barely moved (0.87–0.90), its learning rate being small. With the leak learned at the
model's learning rate (rung 4c), $\beta$ spreads over 0.03–0.95 (mean 0.87),
and the model is still 3.7 % worse (5.523 vs 5.328), within 0.4 % of rung 4b.
Setting $\beta = 0$ at evaluation changes the perplexity of either by
−0.02 % to −0.4 %: the trained models do not use their membrane. The gradient through the carried membrane
explodes (×30 per block backward) and was detached, so $\beta$ is learned from
its effect within a token only; a membrane trained end to end through time is
not tested here.

## 8. Limits

  * One model (RWKV-4-169M) and one training corpus (TinyStories); [TIER 3:
    WikiText-103 replay]. Counts on 32 k tokens.
  * The reading covers rate-coded counts through stateless consumers. A
    natively trained spiking network whose neurons integrate spikes over time
    steps computes a function of spike timing, not only of the count, and has
    no integer twin of this kind.
  * Digital hardware only. Analog or in-memory substrates price accumulation
    differently.
  * The hardware figures come from different nodes and operating points
    (Horowitz is 45 nm; Loihi's figure is a simulated minimum; TrueNorth's
    includes static power). They place orders of magnitude, not a design.
  * The LM head stays in floating point, as in the spiking LMs we cite.

## 9. Related work

**Spiking language models.** SpikeGPT, SpikeLM, SpikeBERT [Lv et al. 2023],
SpikingBERT [Bal & Sengupta 2024], SpikingMamba and SDLLM estimate energy with
the 0.9 / 4.6 pJ constants against a different reference network; SpikeLLM
[Xing et al. 2025] counts bit operations and notes that 1-bit event-driven
execution costs more operations than dense kernels. NSLLM [Xu et al. 2025],
whose integer neuron we follow, converts integer activations to binary spikes over
virtual steps at inference, the reading whose overhead $R$ measures, and
reports 19.8 × against a GPU from measured power on an FPGA.

**Fair accounting of spiking networks.** Davidson and Furber [2021] find that
on digital hardware a rate-coded spiking network wins only below about 1.7
spikes per output, because weight and state memory accesses dominate. Dampfhoffer et al. [2022] include memory accesses and compare with
16- and 8-bit accelerators, attributing any spiking advantage to spike
sparsity rather than to replacing MACs with accumulations; Lemaire et al.
[2022] find memory dominant. None compares one network under both readings,
nor against ternary weights or zero skipping.

**Integer engines.** Ternary weights remove the multiplications
[Ma et al. 2024]; BitNet a4.8 runs 4-bit activations on them
[Wang et al. 2024]. Cnvlutin, EIE and SCNN skip zero activations or weights in
hardware. Delta networks [Neil et al. 2017] and sigma-delta networks
[O'Connor & Welling 2016] propagate only changes; on this model the changes
are as dense as the values.

## 10. Conclusion

When a quantized language model's activations are read as spikes, the same
network can be run as an integer engine, and the comparison is about work,
not about models. Here the spike reading never needs fewer accumulations than
an integer engine that skips zeros, wins against a dense engine only where
activations are sparse, gains nothing from coding changes, and on every
hardware figure we could source costs more. The savings the spiking literature
reports on such models come from ternary weights and sparsity, both available
without spikes, and from pricing them with FP32 constants. If spikes are used
at all in such a model, they belong only at the sparse sites, and even there
an integer engine that skips zeros does better by $R$.

## Appendix: reproduction

Code, decision log (DECISIONS #15–#24) and results: [repository URL TBD].

  * Ladder: `results/stage_a_ladder.md` (commands, commits, checkpoints).
  * Counts: `CUDA_VISIBLE_DEVICES= uv run python scripts/spike_accounting.py
    rung3l_a4_s0 rung3l_a4_s1 rung5_a4_s0 rung5_a4_s1 rung5x2_a4_s0
    rung5x2_a4_s1` (commit `cd409a0`), tables by
    `scripts/accounting_tables.py`, Figure 2 by `scripts/accounting_figure.py`.
  * Hardware figures and their sources: `results/hardware_cost_note.md`.
  * Membrane: `scripts/_beta_stats.py`, `scripts/_diagnose_state_regime.py`.

## References

Bal, M., Sengupta, A. SpikingBERT: Distilling BERT to train spiking language
models using implicit differentiation. AAAI 2024. arXiv:2308.10873.

Albericio, J. et al. Cnvlutin: Ineffectual-neuron-free deep neural network
computing. ISCA 2016.

Dampfhoffer, M., Mesquida, T., Valentian, A., Anghel, L. Are SNNs really more
energy-efficient than ANNs? An in-depth hardware-aware study. IEEE TETCI,
2022. doi:10.1109/TETCI.2022.3214509.

Davidson, S., Furber, S. B. Comparison of artificial and spiking neural
networks on digital hardware. Frontiers in Neuroscience 15:651141, 2021.

Davies, M. et al. Loihi: A neuromorphic manycore processor with on-chip
learning. IEEE Micro 38(1), 2018. doi:10.1109/MM.2018.112130359.

Eldan, R., Li, Y. TinyStories. arXiv:2305.07759, 2023 (to verify).

Esser, S. K. et al. Learned step size quantization. ICLR 2020 (to verify).

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

Merolla, P. A. et al. A million spiking-neuron integrated circuit with a
scalable communication network and interface. Science 345(6197), 2014.
doi:10.1126/science.1254642.

Neil, D., Lee, J. H., Delbruck, T., Liu, S.-C. Delta networks for optimized
recurrent network computation. ICML 2017. arXiv:1612.05571.

O'Connor, P., Welling, M. Sigma delta quantized networks. arXiv:1611.02024,
2016.

Parashar, A. et al. SCNN: An accelerator for compressed-sparse convolutional
neural networks. ISCA 2017. arXiv:1708.04485.

Peng, B. et al. RWKV: Reinventing RNNs for the Transformer era. Findings of
EMNLP 2023 (to verify).

Wang, H., Ma, S., Wei, F. BitNet a4.8: 4-bit activations for 1-bit LLMs.
arXiv:2411.04965, 2024.

Xing, X. et al. SpikeLM: Towards general spike-driven language modeling via
elastic bi-spiking mechanisms. ICML 2024. arXiv:2406.03287.

Xing, X. et al. SpikeLLM: Scaling up spiking neural network to large language
models via saliency-based spiking. ICLR 2025. arXiv:2407.04752.

Xu, H., Qiu, X., et al. Neuromorphic spike-based large language model.
National Science Review 13(4):nwaf551, 2025. doi:10.1093/nsr/nwaf551.

Xu, H., Qiu, X., et al. Spike-driven large language model. arXiv:2604.16475,
2026.

Yao, M. et al. Spike-driven Transformer. NeurIPS 2023. arXiv:2307.01694.

Zhou, Z. et al. Spikformer: When spiking neural network meets Transformer.
ICLR 2023. arXiv:2209.15425.

Zhu, R.-J., Zhao, Q., Li, G., Eshraghian, J. K. SpikeGPT: Generative
pre-trained language model with spiking neural networks. TMLR, 2024.
arXiv:2302.13939.
