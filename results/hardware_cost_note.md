# Hardware cost note (DECISIONS #24, tier 1 point 5)

`results/spike_accounting.md` gives, per site, the cost ratio `ρ*` below which
the ±1 spike reading of a 4-bit count is cheaper than an integer engine running
the same network: spikes win when a spike accumulation costs less than `ρ*`
times a 4-bit one. DEC-24 point 3 fixed the bands with DEC-19's two pricings,
`ρ = 1/4` and `ρ = 1`: reference assumptions, not physical bounds (wording
revised 2026-10-31). This note places published hardware figures inside that range;
it does not move the bands. All figures are from the sources listed at the end,
read in the original documents; none is from memory.

## 1. A digital accumulation is priced by its accumulator

Horowitz (ISSCC 2014, Fig. 1.1.9, 45 nm, 0.9 V, figures the author calls
rough) prices an integer add at 0.03 pJ for 8 bits and 0.1 pJ for 32 bits; there
are no 4-bit or 16-bit rows. An FP32 add is 0.9 pJ and an FP32 multiply 3.7 pJ.

Both readings add into the same accumulator. A BitLinear's integer path
accumulates per weight group of 128 inputs and applies the group's scale once,
in floating point, afterwards (`BitLinear.forward_integer`,
`spiking_ternary_lm.eval.integer_engine`); that last step is the same in every
engine. A group sum of counts of magnitude at most 15 (8 at the signed sites,
15 at the relu² site and for `Δq`) times ternary weights is at most
128 × 15 = 1920 in magnitude: a 12-bit signed accumulator (revised 2026-10-31;
the first version sized it for a sum over the full width, 14–17 bits, which is
not the execution counted). Adding a ±1 spike is an increment of that register,
adding a 4-bit count is an add of a 4-bit operand into it; both are 12-bit
additions, between the 8- and 32-bit rows. Priced by width as Horowitz prices
them, `ρ = 1`. The other reference pricing, `ρ = 1/4`, prices an addition by
its operand: it is an assumption about event-driven hardware, favourable to
spikes, not a figure from this table.

## 2. Each accumulation also fetches a weight

An engine that adds a count or a spike into the outputs of a column first reads
that column's weights. Horowitz prices a 64-bit read of an 8 KB SRAM at 10 pJ
(32 KB: 20 pJ, 1 MB: 100 pJ). With ternary weights packed at 2 bits, 32 weights
per read, that is `f ≈ 0.31 pJ` per weight from the smallest memory in the
table: 3–10 times the add (`a` = 0.03–0.1 pJ, the rows around a 12-bit add;
the 32-bit end, the most favourable to spikes, overstates it). Per non-zero weight it is
0.46 pJ when the zeros are stored (1 in 1.47 weights is non-zero); the table
below uses 0.31 pJ, the value most favourable to spikes.

How often the weight is read depends on the design (revised 2026-10-31; the
first version said grouping a source's spikes amounts to the integer reading,
which is wrong):

  * **(a) One event per spike, weight read per event**, as event-driven chips
    do. S costs `S (ρ a + f)` against `Z (a + f)` for Z: spikes are cheaper
    only if `ρ_eff = (ρ a + f) / (a + f) < 1/R`.
  * **(b) Weight held in a register across a source's `|q|` spikes.** The read
    is paid once per non-zero input in both engines, `Z f + S ρ a` against
    `Z (f + a)`, and drops out: spikes are cheaper only if `ρ < 1/R`, the
    adder-only question of section 1, which no figure here settles.

For design (a):

| `a` (add) | `f` (fetch, 8 KB SRAM) | `ρ_eff` at `ρ = 1/4` | at `ρ = 1` |
|---|---|---|---|
| 0.03 pJ (8-bit row) | 0.31 pJ | 0.93 | 1 |
| 0.045 pJ (8-bit row scaled to 12 bits) | 0.31 pJ | 0.91 | 1 |
| 0.1 pJ (32-bit row) | 0.31 pJ | 0.82 | 1 |

A larger memory raises `f` and pushes `ρ_eff` closer to 1. For design (b) these
figures say nothing.

Davidson and Furber (Frontiers in Neuroscience 15:651141, 2021) build the same
argument from SpiNNaker 2 post-layout figures (22 nm): with `E` the energy of
an addition, a weight read costs about `5E`, a spike event `7E` (weight read,
membrane read and write, addition) and a multiply-accumulate event `12E`. With
ternary weights there is no multiplication, so an integer event that skips
zeros costs the same `7E` as a spike event: `ρ ≈ 1`. Their model is design (a):
every spike event reads its weight and the target state.

## 3. Event-driven chips charge per event, not per bit

Published energies per synaptic event:

| chip | energy per synaptic event | node, conditions | source |
|---|---|---|---|
| Intel Loihi | 23.6 pJ (minimum) | 14 nm, 0.75 V, pre-silicon simulation | Davies et al. 2018, Table 2 |
| IBM TrueNorth | 26 pJ | 28 nm, 0.775 V, total energy at 20 Hz and 128 active synapses per neuron | Merolla et al. 2014 |
| SpiNNaker 2 | ≈ 10 pJ, up to 20 pJ | 22 nm FDSOI, 0.5 V, dynamic power, "equivalent" synaptic operations of a factorised connection | Höppner et al. 2021 |

These are 100 to 900 times a 45 nm integer add, at newer nodes where an add is
cheaper still. Absolute energies from different chips do not give the ratio
between the two readings on any one of them; what they show is that an event's
cost is dominated by memory, routing and the neuron update, not by the width
of what it carries, so on chips that send each spike as an event (design (a))
`ρ` is close to 1. On Loihi a synaptic event is binary; on
TrueNorth one adds a 9-bit weight into a 20-bit membrane (Merolla et al.,
supplement).

Loihi 2 points the same way. Its spikes can carry an integer payload, up to 32
bits and typically 8, which multiplies the downstream weights. Intel says this
costs a "negligible extra energy" over binary spikes, without giving a figure
(Loihi 2 technology brief, 2021). On that chip a 4-bit count can be sent as one
graded spike (engine Z) or as `|q|` binary spikes (engine S); if the per-event
cost is about the same, S costs about `R` times Z (1.9 at rung 3, 2.8 at rung
5). This is a qualitative indication from the vendor, not a measurement of
SpikeBit.

## 4. Where this leaves the verdicts

| cost model | `ρ` (or `ρ_eff`) | S against Z (`ρ*` 0.29–0.58) | S against D, sparse sites (`ρ*` 3.3, 15) | S against D, input sites (`ρ*` 0.33–0.47) |
|---|---|---|---|---|
| adder only, operand-width pricing (reference) | 1/4 | cheaper | cheaper | cheaper |
| adder only, adder-width pricing (Horowitz) | 1 | dearer | cheaper | dearer |
| (a) event per spike, weight read from 8 KB SRAM per event | 0.82–1 | dearer | cheaper | dearer |
| (a) per-event model of Davidson & Furber (SpiNNaker 2 figures) | ≈ 1 | dearer | cheaper | dearer |
| (a) event-driven chips; Loihi 2 graded spikes (qualitative) | ≈ 1 | dearer | cheaper | dearer |
| (b) weight held across a source's spikes | = adder `ρ` | open: as one of the first two rows | cheaper | open |

The bands of DEC-24 stay as measured ("depends on hardware" against Z at every
site). Under every design (a) cost model this note could price, spikes cost
more than a zero-skipping integer engine at every site. Under design (b) the
weight read drops out and the verdict is the adder question alone: spikes win
against Z only if a ±1 increment of the 12-bit group accumulator costs under
`1/R` (0.29 to 0.58) of adding a 4-bit operand into it. Horowitz prices
additions by width (no); pricing them by operand width (yes) is an assumption,
and no measurement here settles it. These are cost models, not a bound on all
digital hardware. Against an engine that does not skip zeros, spikes need fewer
accumulations only at the two sparse sites, and win there under every model,
though a zero-skipping integer engine is cheaper still there, by `R` (1.7–2.2),
whenever `ρ = 1`. At the input sites they win against D only if `ρ` is under
0.33–0.47, as with operand-width pricing.

## 5. The whole model

Per token at rung 5 seed 0, with Horowitz's prices: accumulations as counted
(Z 34.4 M, S 96.2 M), `a` = 0.03–0.1 pJ, the LM head as 38.6 M FP16 MACs at
1.5 pJ (1.1 multiply + 0.4 add), and the 2.5 M other FP operations at 0.4 pJ.

| engine | BitLinear adds | head | other FP | total | vs Z |
|---|---|---|---|---|---|
| Z | 1.0–3.4 µJ | 57.9 µJ | 1.0 µJ | 59.9–62.3 µJ | 1 |
| S, `ρ = 1` | 2.9–9.6 µJ | 57.9 µJ | 1.0 µJ | 61.8–68.5 µJ | 1.03–1.10 |
| S, `ρ = 1/4` | 0.7–2.4 µJ | 57.9 µJ | 1.0 µJ | 59.6–61.3 µJ | 0.98–0.995 |

With an FP head, the reading of the BitLinears moves the arithmetic of a token
by −2 % to +10 %. Weight fetch adds 10.7 µJ (Z) or 30 µJ (S, one read per
spike) at `f` = 0.31 pJ, more than all the BitLinear adds. If the 92 M ternary
weights (≈ 23 MB at 2 bits) come from DRAM at batch 1, Horowitz's 1.3–2.6 nJ
per access (the figure gives no width) puts weight movement at 3.7–7.5 mJ per
token for 64-bit accesses, 0.5–0.9 mJ for 512-bit ones: 7 to 125 times the
arithmetic of a token. These totals illustrate orders of magnitude; they are
not a cost model of a built engine.

## 6. The SNN papers' constants

Spikformer (Zhou et al., ICLR 2023, appendix C.2) and Spike-driven Transformer
(Yao et al., NeurIPS 2023) charge 4.6 pJ per MAC and 0.9 pJ per synaptic
accumulation, citing Horowitz at 45 nm; Spike-driven Transformer states that
these are 32-bit floating-point operations. They are: 4.6 = 3.7 (FP32
multiply) + 0.9 (FP32 add). The comparison is therefore an FP32 MAC against an
FP32 add, and the 5.1 × it credits to spikes is the removal of the multiply,
which ternary weights give any engine (`spike_accounting.md`, decomposition).
SpikeGPT (Zhu et al., TMLR 2024) uses 4.5 and 0.9 pJ and counts integer
activations as accumulations by appeal to graded spikes, which is engine Z.

NSLLM (Xu et al., National Science Review 2025), the project's reference
paper (DEC-1), reports power measured on an FPGA against a GPU, not a pJ
formula. At inference it expands a 4-bit integer activation into 15 binary time steps
(A4T1 to A1T15), the expansion of rung 4a's unsigned reading. `R` measures
the accumulation overhead of such a binary mode over the integer mode of the
same network.

## 7. Related assessments

  * Dampfhoffer et al. (IEEE TETCI 2022) bound the energy advantage of SNNs
    against efficient ANN accelerators, with memory accesses included. With
    8-bit prices from Horowitz an accumulate costs 0.13 of a MAC and a memory
    access 5.4 MACs. SNNs compete only at high spike sparsity, 0.15 to 1.38
    spikes per synapse per inference depending on the ANN implementation.
    SpikeBit's sparse sites (0.07 and 0.30 spikes per input per token) fall in
    or under that range; its input sites (2.1–3.0) are above it.
  * Lemaire et al. (arXiv 2022), with every value on 32 bits, find SNNs 6 to
    8 times cheaper than feed-forward networks at under 0.14 spikes per neuron per
    inference, with memory accesses dominating the total. They compare with no
    quantized network.

Neither compares with a ternary integer engine that skips zeros, the baseline
of this project.

## Not established

  * The cost of a ±1 increment against a 4-bit addition into the same
    accumulator: the whole verdict for design (b) rests on it.

  * No Loihi 2 energy per operation was found in Intel's documents; its graded
    spikes' cost is Intel's qualitative statement.
  * Horowitz has no 4-, 12- or 16-bit integer add; `a` for the 12-bit group
    accumulator is bracketed by the 8- and 32-bit rows. All Horowitz figures
    are 45 nm.
  * DEC-1 cites NSLLM as Yao et al.; PubMed Central and Crossref list 20
    authors, first Han Xu, Man Yao 13th. Cited here as Xu et al.
  * The chip figures are per synaptic event at specific operating points and
    include different overheads (TrueNorth: total energy; SpiNNaker 2: dynamic
    power with a counting convention that flatters it; Loihi: a minimum from
    simulation).

## Sources

  * M. Horowitz, "Computing's energy problem (and what we can do about it)",
    ISSCC 2014, pp. 10–14, doi:10.1109/ISSCC.2014.6757323, Fig. 1.1.9.
  * M. Davies et al., "Loihi: A neuromorphic manycore processor with on-chip
    learning", IEEE Micro 38(1), 2018, doi:10.1109/MM.2018.112130359, Table 2.
  * Intel Labs, "Taking Neuromorphic Computing to the Next Level with Loihi 2",
    technology brief, 2021; G. Orchard et al., "Efficient neuromorphic signal
    processing with Loihi 2", IEEE SiPS 2021, arXiv:2111.03746.
  * P. A. Merolla et al., "A million spiking-neuron integrated circuit with a
    scalable communication network and interface", Science 345(6197), 2014,
    doi:10.1126/science.1254642, and its supplement.
  * S. Höppner et al., "The SpiNNaker 2 processing element architecture for
    hybrid digital neuromorphic computing", arXiv:2103.08392, 2021, Sec. VI.
  * Z. Zhou et al., "Spikformer: When spiking neural network meets
    Transformer", ICLR 2023, arXiv:2209.15425 (v2 read), appendix C.2.
  * M. Yao et al., "Spike-driven Transformer", NeurIPS 2023, arXiv:2307.01694,
    Sec. 4 and supplement S1.
  * R.-J. Zhu et al., "SpikeGPT: Generative pre-trained language model with
    spiking neural networks", TMLR 2024, arXiv:2302.13939 (v5), Sec. 3.7.
  * H. Xu et al., "Neuromorphic spike-based large language model", National
    Science Review 13(4), nwaf551, 2025, doi:10.1093/nsr/nwaf551 (main text;
    supplement not read).
  * M. Dampfhoffer et al., "Are SNNs really more energy-efficient than ANNs? An
    in-depth hardware-aware study", IEEE TETCI, 2022,
    doi:10.1109/TETCI.2022.3214509 (abstract, Table I and parts read).
  * E. Lemaire et al., "An analytical estimation of spiking neural networks
    energy efficiency", arXiv:2210.13107, 2022.
  * S. Davidson, S. B. Furber, "Comparison of artificial and spiking neural
    networks on digital hardware", Frontiers in Neuroscience 15:651141, 2021,
    doi:10.3389/fnins.2021.651141.
