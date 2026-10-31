<!--
Draft of the arXiv note (DECISIONS #24; plan in writeup/plan.md). Markdown with
LaTeX math, to convert once a TeX toolchain is chosen. Every number is in
results/ with its command and commit. [TIER 3] marks what waits for the
general-text runs. References marked (to verify) were not read for this draft.
Revision 2 (2026-10-31) applies an internal review: scope of R, the
pre-registration sentence, the encodings of the cited spiking LMs, structure.
-->

# Reading a Ternary Language Model as Spikes Does Not Save Work: An Exact Accounting Against Integer Engines

*Author(s) TBD*

## Abstract

Spiking language models report large energy savings, usually by pricing a
spike accumulation at 0.9 pJ against 4.6 pJ for a 32-bit floating-point
multiply-accumulate. We test where such savings come from on a model where
spiking and integer execution compute exactly the same function: RWKV-4-169M
converted to ternary weights and 4-bit stateless LIF activations at 1.08 × the
full-precision perplexity on TinyStories, whose activations read exactly as ±1
spikes. On the projections, the literature's formula credits spikes with
3.3 × less energy; this factors into removed multiplications (5.1 ×) and
skipped zeros (1.8 ×), which an integer engine with ternary weights also has,
times a spike-count penalty (0.36). Counted directly, spikes need 1.7–3.4 times
the accumulations of a zero-skipping integer engine at every site, beat a
dense engine only at the two sparse sites, and gain nothing from delta coding.
Digital adders, weight fetches and neuromorphic chips all price a spike
accumulation at 0.82 or more of an integer one, where spikes lose. A membrane
carried across tokens did not help. [TIER 3: one clause on WikiText-103.]

## 1. Introduction

A growing line of work builds language models from spiking neurons and
reports large energy savings: 32.2 × for SpikeGPT [Zhu et al. 2024], 12.9 ×
for SpikeLM [Xing et al. 2024], 4.76 × for SpikingMamba [Huang et al. 2026],
6 × against an INT4 model for SDLLM [Xu et al. 2026, "SDLLM"]. Most of these
figures come from one formula [Zhou et al. 2023; Yao et al. 2023]: the
spiking network's synaptic operations, firing rate × time steps × the layer's
multiply-accumulates (MACs), are priced at 0.9 pJ, and the reference network's
MACs at 4.6 pJ (4.5 pJ in SpikeGPT). Both constants are 32-bit floating-point
operations from Horowitz's 45 nm table [Horowitz 2014]: 0.9 pJ is an FP32
addition, 4.6 pJ an FP32 multiplication plus addition.

Look at what these models send as spikes. NSLLM [Xu et al. 2025, "NSLLM"]
trains single-step integer activations and expands each into binary spikes
over virtual time steps at inference. SDLLM unfolds integer counts into
bidirectional spikes over eight steps. SpikingMamba's headline row uses signed
multi-level spikes, SpikeLM ternary spikes at one time step, and SpikeGPT counts
its non-binary ReLU² activations as additions by appeal to graded spikes.
Each is an integer network run either as binary events, one per unit of the
count, or as one graded event per non-zero count. For such a network the
relevant question is not spiking against floating point, but: given the same
integer activations, does sending them as binary spikes save work over an
integer engine? Ternary weights already turn multiplications into additions
[Ma et al. 2024], and accelerators already skip zero activations
[Albericio et al. 2016; Han et al. 2016; Parashar et al. 2017].

The question has an exact answer when the spike reading is exact, because then
the two engines compute the same logits and differ only in how many
accumulations they perform and what each costs. We build such a model and:

  * count, on the same weights and inputs, the accumulations of the spike
    reading and of three integer engines: dense, zero-skipping, and delta
    coding between tokens (Section 4);
  * decompose the spiking literature's energy formula on the same model into
    what spikes contribute and what any integer engine already has
    (Section 5);
  * place published hardware figures against the break-even cost each
    comparison requires (Section 6);
  * test whether a membrane carried across tokens, what a LIF neuron has and a
    quantizer lacks, improves the model (Section 7).

The verdict bands of Section 4 and the hardware analysis were fixed in a
decision log kept with the code before the dense, delta and hardware analyses
were run; the zero-skipping ratio $R$ was already known from an earlier
measurement on the same checkpoints.

## 2. Setting

**Model.** RWKV-4-169M [Peng et al. 2023 (to verify)]: 12 blocks of width 768,
each with a time mix (the WKV recurrence) and a channel mix (a relu² MLP of
width 3072). Each block first mixes every token with the previous one,
$x_t' = \mu \odot x_t + (1 - \mu) \odot x_{t-1}$ ("token shift"), separately
for each projection input ($x_r$, $x_k$, $x_v$). The 84 projections of the
blocks become ternary BitLinear layers (92.0 M parameters): weights in
$\{-1, 0, +1\}$ with one scale per group of 128 inputs, as in BitNet b1.58
[Ma et al. 2024]. Embeddings, the LM head (38.6 M MACs per token), layer
norms, the WKV recurrence and the sigmoid gates stay in floating point.

**Neurons.** Our integer LIF neuron (FAI-LIF), after NSLLM, quantizes its
input to a 4-bit count $q$ per token on a learned grid (LSQ step size
[Esser et al. 2020 (to verify)], clipped straight-through gradient) and
outputs $s \cdot q$; signed sites use $q \in [-8, 7]$, the channel-mix relu²
output $q \in [0, 15]$. The final model has 120 such neuron sites: one on
each of the 7 projection inputs of each block, plus the WKV inputs $k$, $v$ and
the channel-mix output. They are stateless (leak $\beta = 0$); Section 7 tests
a membrane.

**Training and evaluation.** Fine-tuning on 20.5 M tokens of TinyStories
[Eldan & Li 2023 (to verify)], 10 k steps of 16 × 128 tokens; quantized models
add a KL term to the full-precision fine-tuned model. Perplexity on the first
1 M tokens of the last 1 % of the corpus, in 128-token windows each started
from a fresh state. Two seeds per configuration on disjoint text; a difference
counts only above max(2 × seed spread, 5 % of perplexity). Each step was
pre-registered with its threshold:

| rung | model | PPL (mean of 2 seeds) | vs rung 1 |
|---|---|---|---|
| 0 | converted RWKV-4-169M, no quantization | 13.98 (= the original) | – |
| 1 | full precision, fine-tuned (reference) | 4.986 | 1 |
| 2 | + ternary weights | 5.157 | 1.034 |
| 3 | + 4-bit LIF at the WKV inputs, time-mix output and channel-mix hidden and output | 5.328 | 1.069 |
| 5 | rung 3 + 4-bit LIF on every remaining projection input | **5.391** | 1.081 |

Rung 5 is the model studied below: every input of every ternary projection is
a 4-bit count. At twice the training budget, rung 1 / rung 5 moves from
92.5 % to 94.3 %. [TIER 3: WikiText-103 rows.]

## 3. An exact spike reading

A signed count $q \in [-8, 7]$ is read as $|q|$ spikes of sign
$\mathrm{sign}(q)$ over $T = 8$ virtual steps; an unsigned count
$q \in [0, 15]$ as $q$ unit spikes over $T = 15$. Through a layer linear in its
input, the responses to the spikes sum to the response to $q$; through a
ternary BitLinear each response is a set of integer additions, and the
equality is exact, scale included. We checked it on real counts at every site
of blocks 0 and 11, and on random counts in unit tests.

We call a site *spike-native* when its consumer is a ternary BitLinear: at rung
5, the 84 projection inputs. The WKV inputs (whose $e^k$ needs the decoded
value) and the channel-mix output (multiplied by a floating-point gate) are
not; they are the same in every engine and are left out of the counts.

The change between consecutive tokens, $\Delta q_t = q_t - q_{t-1}$, lies in
$[-15, 15]$ and is read the same way on a signed 5-bit grid ($T = 16$). A
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
| S, ±1 spikes | $\sum_i \lvert q_i \rvert\, c_i$ | one spike |
| ΔZ, ΔS | Z, S on $\Delta q_t$ | 5-bit count, one spike |

**Cost model.** Let $a$ be the cost of a 4-bit-operand accumulation, $\rho a$
that of a spike accumulation into the same accumulator, and $f$ the cost of
fetching the weight each accumulation uses (Section 6). Ignoring $f$, spikes
are cheaper than engine X when $\rho < \rho^* = \mathrm{ops}_X/\mathrm{ops}_S$.
Two pricings bound $\rho$: both operands enter the same 14–17-bit adder, and
Horowitz's table prices additions by width, so $\rho = 1$; priced by operand
width instead, an assumption favourable to spikes, $\rho = 1/4$. The bands:
$\rho^* > 1$, spikes cheaper at any cost in that range; $\rho^* < 1/4$,
dearer at any cost; in between, it depends on the hardware.

**The sign is trivial, the size is not.** $|q_i| \geq \mathbb{1}[q_i \neq 0]$,
so $R = S/Z \geq 1$ for any network: with ternary weights an integer engine
has no multiplication left to remove, and skipping zeros is open to it as much
as to spikes. What a measurement adds is how far above 1: the mean count of a
non-zero activation, 1.7 at the sparse time-mix output, 2.2 at the relu²
output, 2.6 to 3.1 at the dense projection inputs.

**Results** (first 256 windows of the evaluation tail, 32 k tokens; the two
seeds agree to within 0.06 in $\rho^*$, except 14.3 against 15.3 at the relu²
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

Figure 1 shows the same numbers on a log scale.

  * **Against Z, spikes always need more accumulations:** $R$ = 1.7 to 3.1 per
    site at rung 5 (up to 3.4 at twice the budget), 2.8 pooled. A spike
    accumulation would have to cost between 0.32 and 0.58 of a 4-bit one.
  * **Against D, the verdict follows sparsity.** At the two sparse sites
    spikes are cheaper at any $\rho \leq 1$. At the five dense input sites,
    which carry 61.5 % of the projection work, and pooled ($\rho^* = 0.65$),
    they are not. Z gets the sparse sites' benefit without the spike count.
  * **Delta coding saves nothing.** At the dense sites $\Delta q$ is non-zero
    as often as $q$; at the sparse sites more often, since a unit that turns on
    and off makes two changes. The bands are the same as without it, except
    one site of the ×2-budget model that becomes "dearer".

The counts on rung 3 (whose only spike-native sites are the two sparse ones)
give a pooled $\rho^*$ of 8.8–8.9 against D and 0.53 against Z; on rung 5 at
twice the budget, 0.62 and 0.34. [TIER 3: WikiText-103 counts.]

The quantizers were not trained for sparsity; a firing-rate penalty would
lower $E|q|$ at some cost in perplexity (not tested here). It would move Z as
much as S, so only the comparison with D could change.

## 5. What the spiking energy formula counts

Applied to rung 5 as Spikformer and Spike-driven Transformer apply it
(synaptic operations = $E|q|$ × MACs at 0.9 pJ, against the same MACs at
4.6 pJ for the full-precision network), the formula gives the spiking model
3.3 × less energy on the projections, 2.0 × with the full-precision LM head
counted on both sides. On the spike-native layers the ratio factors exactly,
with $M_s$ the MACs and $P_s$ the non-zero fraction of site $s$:

$$\frac{E_{\mathrm{ANN}}}{E_{\mathrm{SNN}}}
= \frac{4.6}{0.9} \times \frac{\sum_s M_s}{\sum_s P_s M_s}
\times \frac{\sum_s P_s M_s}{\sum_s E|q|_s M_s},$$

which at rung 5 is 5.11 (multiplications removed) × 1.82 (zeros skipped) ×
0.36 (spike count). The first two factors are what ternary weights and zero
skipping give any engine: an integer engine that skips zeros, priced with the
same constants, gets 5.11 × 1.82 = 9.3 × (13.7 × if it also skips weight
zeros, which the formula does not count). The third is the spike reading's own
contribution, and it is a cost. On rung 3's two sparse sites the formula gives
45 ×, against 84 × for the integer engine.

The constants are FP32 operations. An integer addition costs 0.03 to 0.1 pJ in
the same table (8 to 32 bits), a further 9–30 × that both integer and spike
engines could claim. When the reference network is itself integer, charging
its operations the FP32 MAC constant hands the spiking side the 5.1 × of the
constants: SDLLM reports 6 × less energy than an INT4 model while performing
1.15 × fewer operations (our arithmetic from its Table 2).

## 6. Where hardware sits

**Weight fetch.** Each accumulation reads the weight it adds. From the
smallest SRAM in Horowitz's table (10 pJ per 64-bit read), 32 packed ternary
weights cost $f \approx 0.31$ pJ each, 3 to 10 times an addition
($a$ = 0.03–0.1 pJ). A spike engine that handles each spike as an event reads
the weight once per spike; grouping a source's spikes into one read means
holding the count, which is the integer reading. With the fetch,
$\rho_{\mathrm{eff}} = (\rho a + f)/(a + f)$ is 0.82 to 0.93 even at
$\rho = 1/4$. Davidson and Furber's per-event model from SpiNNaker 2
post-layout figures [2021] gives the same picture: with $E$ the cost of an
addition, a weight read costs about $5E$, a spike event $7E$ and a
multiply-accumulate event $12E$. Without the multiplication, which ternary
weights remove, an integer event costs the same $7E$ as a spike event.

**Neuromorphic chips** charge per synaptic event, not per bit carried: 23.6 pJ
(minimum, simulated, 14 nm) on Loihi [Davies et al. 2018], 26 pJ (28 nm,
static power included) on TrueNorth [Merolla et al. 2014], about 10 to 20 pJ
(22 nm) on SpiNNaker 2 [Höppner et al. 2021]: 100 to 900 times a 45 nm integer
addition, at nodes where an addition is cheaper still. Loihi 2 carries integer
payloads in a single "graded" spike at what Intel calls negligible extra
energy over a binary spike [Intel 2021]; there a 4-bit count can go as one
event (engine Z) or as $|q|$ binary events (engine S), and S costs $R$ times
as much.

| hardware point | $\rho$ | vs Z | vs D, sparse sites | vs D, input sites |
|---|---|---|---|---|
| operand-width pricing, no fetch (bound) | 1/4 | cheaper | cheaper | cheaper |
| additions priced by width (Horowitz, 45 nm) | 1 | dearer | cheaper | dearer |
| additions + weight fetch from SRAM | 0.82–1 | dearer | cheaper | dearer |
| per-event model, SpiNNaker 2 figures (22 nm) | ≈ 1 | dearer | cheaper | dearer |
| event-driven chips, graded spikes | ≈ 1 | dearer | cheaper | dearer |

Only the operand-width bound without weight fetch lets spikes win against Z,
and no published figure we found supports it.

**The whole model.** The projections are a small part of a token's energy.
With the LM head in FP16 (58 µJ per token at 1.5 pJ per MAC), their additions
cost 1.0–3.4 µJ in Z and 2.9–9.6 µJ in S at $\rho = 1$, so the reading moves a
token's arithmetic by −2 % to +10 %; weight fetch adds 10.7 µJ (Z) or 30 µJ
(S). If the 23 MB of ternary weights come from DRAM at batch 1, moving them
costs 7 to 125 times the arithmetic. These are orders of magnitude, not the
cost model of a built engine. The spiking LMs cited above also keep a
floating-point head.

## 7. The membrane

A LIF neuron differs from a quantizer by its membrane, the residual of one
token carried with leak $\beta$ into the next. Started at $\beta = 0.9$ on
rung 3 and trained at a small learning rate, $\beta$ barely moved
(0.87–0.90) and the model was 4.0 % worse than rung 3 (5.543 vs 5.328; the
pre-registered threshold to keep it was 5.062). With $\beta$ trained at the
model's learning rate it spread over 0.03–0.95 and the model was still 3.7 %
worse (5.523). Setting $\beta = 0$ at evaluation changes either model's
perplexity by −0.02 % to −0.4 %: the trained models do not use their membrane.
The gradient through the carried membrane explodes (×30 per block backward)
and was detached, so $\beta$ learns from its effect within one token only.

## 8. Related work

**Spiking language models.** SpikeGPT, SpikeLM, SpikeBERT [Lv et al. 2023],
SpikingMamba and SDLLM estimate energy with the 0.9 / 4.6 pJ constants (4.5 pJ
in SpikeGPT) against a different, full-precision or INT4 reference network;
SpikingBERT [Bal & Sengupta 2024] uses the same constants against the same
architecture run without spikes. SpikeLLM [Xing et al. 2025] counts bit
operations instead and notes that 1-bit event-driven execution costs more
operations than dense kernels. NSLLM, whose integer-to-binary inference our
neuron follows, reports a 19.8 × ratio of dynamic power between an FPGA and a
GPU.

**Fair accounting of spiking networks.** Davidson and Furber [2021] show that
on digital hardware a rate-coded spiking network wins only below about 1.7
spikes per output, because weight and state memory accesses dominate.
Dampfhoffer et al. [2022] include memory accesses and compare with 16- and
8-bit accelerators, including zero-skipping ones, and attribute any spiking
advantage to spike sparsity rather than to replacing MACs with accumulations;
Lemaire et al. [2022] also find memory dominant. None compares one network
under both readings, or against a ternary integer engine that skips zeros.

**Integer engines.** Ternary weights remove the multiplications
[Ma et al. 2024]. BitNet a4.8 [Wang et al. 2024] runs 4-bit activations on
them for the attention and FFN input projections, keeping the others at 8 bits
with sparsification; our rung 5 takes every projection input to 4 bits.
Cnvlutin, EIE and SCNN skip zero activations or weights in hardware. Delta
networks [Neil et al. 2017] and sigma-delta networks
[O'Connor & Welling 2016] propagate only changes; here the changes are as dense
as the values.

## 9. Limits

  * One model (RWKV-4-169M) at one size; one training corpus (TinyStories)
    [TIER 3: plus WikiText-103]; counts on 32 k tokens.
  * The reading covers integer counts sent as rate-coded spikes through
    stateless consumers, the encoding of the spiking LMs above. A network
    whose neurons integrate spikes over time steps computes a function of
    spike timing and has no integer twin of this kind.
  * Digital hardware only; analog or in-memory substrates price accumulation
    differently.
  * The hardware figures come from different nodes and operating points; they
    place orders of magnitude, not a design.

## 10. Conclusion

When a quantized language model's activations are read as spikes, the same
network runs as an integer engine, and the comparison is about work rather
than about models. Here the spike reading never needs fewer accumulations
than an integer engine that skips zeros, wins against a dense engine only
where activations are sparse, gains nothing from delta coding, and costs more
on every hardware figure we could source. The savings reported for such
models come from removing multiplications and skipping zeros, which an integer
engine with ternary weights also has, priced with FP32 constants. If spikes
are used at all in such a model they belong at the sparse sites, and there an
integer engine that skips zeros still does better by $R$.

## Appendix: reproduction

Code, decision log (DECISIONS #15–#24) and results: [repository URL TBD].

  * Ladder: `results/stage_a_ladder.md` (commands, commits, checkpoints).
  * Counts: `CUDA_VISIBLE_DEVICES= uv run python scripts/spike_accounting.py
    rung3l_a4_s0 rung3l_a4_s1 rung5_a4_s0 rung5_a4_s1 rung5x2_a4_s0
    rung5x2_a4_s1` (commit `cd409a0`); tables by `scripts/accounting_tables.py`;
    Figure 1 by `scripts/accounting_figure.py`.
  * Hardware figures and their sources: `results/hardware_cost_note.md`.
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
