# The spike reading against integer engines (DECISIONS #24)

Tier 1 of DEC-24: counts on the existing Stage A checkpoints, no training. The
question is whether reading SpikeBit's 4-bit LIF outputs as ±1 spikes saves
work against an integer engine running the same network. Quality is not at
stake: the spike reading is exact (rung 4a), so every engine below computes the
same logits as the checkpoint.

## Method

`scripts/spike_accounting.py` (commit `cd409a0`), CPU, on the 4a protocol: the
first 256 windows of 128 tokens of the eval tail (32 768 tokens), fresh state
per window, eval mode.

    CUDA_VISIBLE_DEVICES= uv run python scripts/spike_accounting.py rung3l_a4_s0 rung3l_a4_s1 rung5_a4_s0 rung5_a4_s1 rung5x2_a4_s0 rung5x2_a4_s1

Output `runs/accounting/<run>.json` and `runs/accounting/spike_accounting.txt`.
Checkpoints as in `results/stage_a_summary.md` (sha256 prefixes there).

For a ternary consumer `W` (out × in) of a LIF output, with `c_i = nnz(W[:, i])`
(weight zeros are static; every engine skips them) and the integer counts
`q_i` of one token, accumulations per token:

| engine | accumulations | operand |
|---|---|---|
| D, integer, dense activations | `Σ c_i` | 4-bit |
| Z, integer, zero activations skipped (DEC-19's baseline) | `Σ 1{q_i ≠ 0} c_i` | 4-bit |
| S, ±1 spike reading | `Σ |q_i| c_i` | 1 spike |
| ΔZ, Z on `Δq_t = q_t − q_{t−1}` | `Σ 1{Δq_i ≠ 0} c_i` | 5-bit |
| ΔS, S on `Δq_t` | `Σ |Δq_i| c_i` | 1 spike |

Δ engines keep the consumer's output across tokens, `y_t = y_{t−1} + W Δq_t`,
with `Δq_0 = q_0` at the start of each window. With a spike accumulation priced
`ρ` times a 4-bit one into the same accumulator, spikes are cheaper than engine
X when `ρ < ρ* = ops_X / ops_S`. Verdict bands, fixed in DEC-24 point 3 from
DEC-19's two pricings: `ρ* > 1` spikes cheaper even at equal cost; `ρ* < 1/4`
spikes dearer even priced as a 1-bit operand; otherwise it depends on the
hardware.

Checks passed before any number below was used: the pooled `S / Z` reproduces
the 4a `R` to three decimals for the four checkpoints 4a measured, and on real
counts (first two windows, blocks 0 and 11, every spike-native site) the spike
readings of `q` and of `Δq` give the consumer's integer outputs exactly; the
unit test `test_delta_spike_reading_is_exact_through_a_ternary_bitlinear`
covers random counts including the largest swing.

## Result

  * **Against an integer engine that skips zero activations (Z), the spike
    reading never needs fewer accumulations.** `ρ* = 1/R` lies between 0.29
    and 0.58 at every site, in every model, on both seeds: spikes win only if
    a spike accumulation costs under a third to a half of a 4-bit one. Every
    site falls in the "depends on hardware" band, as `R ≥ 1` dictates.
  * **Against an engine that does not skip zeros (D), the verdict splits by
    site.** At the two sparse sites (time-mix `lif_out`, 17–18 % non-zero;
    channel-mix relu² `lif_k`, 3 %) `ρ* ≈ 3.3` and `≈ 15`: spikes are cheaper
    even at equal add cost. At rung 5's five input sites (82–89 % non-zero,
    `E|q|` 2.1–3.0) `ρ* = 0.33–0.47`: it depends on the hardware. Pooled,
    rung 3 is cheaper in spikes (`ρ* = 8.8–8.9`); rung 5 is not (`ρ* = 0.65`),
    because the five dense input sites carry 61.5 % of the BitLinear MACs.
  * **Coding the change between tokens saves nothing.** `ΔZ / Z` is 0.97–1.01
    at the dense sites and 1.24–1.58 at the sparse ones: consecutive tokens do
    not reuse counts, and a sparse unit that turns on and off produces two
    non-zero changes. ΔS against ΔZ gives the same bands as S against Z,
    except rung 5 ×2's time-mix `lif_xv` (0.246–0.248, "spikes dearer").
  * **The SNN papers' accounting credits the spikes with savings they do not
    make.** On rung 5 it gives 3.3 × less energy than the FP32 model (BitLinears
    only; 2.0 × with the FP head). That factor is 5.11 (the multiplications
    removed by ternary weights) × 1.82 (zero activations skipped) × 0.36 (the
    spike count, a cost). An integer engine that skips zeros gets the first two
    factors without the third: 9.3 × with the same constants, 13.7 × if weight
    zeros are skipped too. On rung 3's sparse sites the same accounting gives
    45 ×, against 84 × for the integer engine.

Where real hardware sits in these bands is the subject of
`results/hardware_cost_note.md`.

## Per site

| model | site | P(q ≠ 0) | E\|q\| | P(Δq ≠ 0) | ρ* S vs D | ρ* S vs Z | ρ* ΔS vs ΔZ | ΔZ / Z |
|---|---|---|---|---|---|---|---|---|
| rung 3 | `time_mix.lif_out` | 0.177 / 0.179 | 0.301 / 0.305 | 0.218 / 0.220 | 3.332 / 3.292 (cheaper) | 0.583 / 0.581 (depends) | 0.624 / 0.622 (depends) | 1.239 / 1.240 |
| rung 3 | `channel_mix.lif_k` | 0.032 / 0.032 | 0.067 / 0.069 | 0.050 / 0.050 | 15.141 / 14.775 (cheaper) | 0.474 / 0.468 (depends) | 0.511 / 0.505 (depends) | 1.565 / 1.570 |
| rung 5 | `time_mix.lif_out` | 0.174 / 0.176 | 0.300 / 0.305 | 0.217 / 0.219 | 3.349 / 3.296 (cheaper) | 0.578 / 0.574 (depends) | 0.621 / 0.619 (depends) | 1.249 / 1.248 |
| rung 5 | `channel_mix.lif_k` | 0.032 / 0.030 | 0.072 / 0.067 | 0.049 / 0.048 | 14.286 / 15.282 (cheaper) | 0.447 / 0.459 (depends) | 0.499 / 0.505 (depends) | 1.556 / 1.584 |
| rung 5 | `time_mix.lif_xr` | 0.825 / 0.825 | 2.156 / 2.168 | 0.832 / 0.832 | 0.462 / 0.459 (depends) | 0.381 / 0.379 (depends) | 0.374 / 0.372 (depends) | 1.006 / 1.008 |
| rung 5 | `time_mix.lif_xk` | 0.842 / 0.843 | 2.225 / 2.242 | 0.844 / 0.845 | 0.448 / 0.444 (depends) | 0.377 / 0.375 (depends) | 0.351 / 0.348 (depends) | 1.002 / 1.002 |
| rung 5 | `time_mix.lif_xv` | 0.872 / 0.872 | 2.713 / 2.716 | 0.878 / 0.878 | 0.371 / 0.371 (depends) | 0.323 / 0.323 (depends) | 0.272 / 0.272 (depends) | 1.010 / 1.010 |
| rung 5 | `channel_mix.lif_xk` | 0.864 / 0.865 | 2.521 / 2.545 | 0.850 / 0.848 | 0.397 / 0.393 (depends) | 0.343 / 0.340 (depends) | 0.347 / 0.342 (depends) | 0.983 / 0.981 |
| rung 5 | `channel_mix.lif_xr` | 0.841 / 0.825 | 2.190 / 2.129 | 0.830 / 0.813 | 0.456 / 0.469 (depends) | 0.384 / 0.387 (depends) | 0.378 / 0.375 (depends) | 0.986 / 0.985 |
| rung 5 ×2 | `time_mix.lif_out` | 0.173 / 0.175 | 0.302 / 0.305 | 0.217 / 0.220 | 3.303 / 3.269 (cheaper) | 0.568 / 0.567 (depends) | 0.614 / 0.612 (depends) | 1.257 / 1.262 |
| rung 5 ×2 | `channel_mix.lif_k` | 0.028 / 0.029 | 0.066 / 0.067 | 0.044 / 0.046 | 15.549 / 15.237 (cheaper) | 0.435 / 0.440 (depends) | 0.494 / 0.492 (depends) | 1.560 / 1.576 |
| rung 5 ×2 | `time_mix.lif_xr` | 0.831 / 0.831 | 2.209 / 2.216 | 0.835 / 0.836 | 0.450 / 0.449 (depends) | 0.374 / 0.373 (depends) | 0.370 / 0.368 (depends) | 1.003 / 1.003 |
| rung 5 ×2 | `time_mix.lif_xk` | 0.849 / 0.848 | 2.313 / 2.305 | 0.848 / 0.848 | 0.431 / 0.432 (depends) | 0.366 / 0.367 (depends) | 0.342 / 0.341 (depends) | 1.000 / 1.000 |
| rung 5 ×2 | `time_mix.lif_xv` | 0.884 / 0.885 | 2.985 / 3.007 | 0.882 / 0.882 | 0.336 / 0.334 (depends) | 0.297 / 0.295 (depends) | 0.248 / 0.246 (dearer) | 1.000 / 0.999 |
| rung 5 ×2 | `channel_mix.lif_xk` | 0.866 / 0.866 | 2.668 / 2.695 | 0.841 / 0.841 | 0.375 / 0.371 (depends) | 0.325 / 0.322 (depends) | 0.331 / 0.326 (depends) | 0.971 / 0.971 |
| rung 5 ×2 | `channel_mix.lif_xr` | 0.838 / 0.836 | 2.252 / 2.230 | 0.821 / 0.818 | 0.443 / 0.447 (depends) | 0.372 / 0.374 (depends) | 0.363 / 0.367 (depends) | 0.979 / 0.978 |

Seed 0 / seed 1. `ρ*` is the cost ratio below which a spike accumulation makes
the spike reading cheaper than the engine it is compared with; the band is in
brackets. Only the spike-native sites (consumer a ternary BitLinear) are
listed; the other LIF sites feed the WKV or the FP gate and are not
accumulations (their `E|q|` and `P(q ≠ 0)` are in the JSON files).

## Pooled over the spike-native sites

| model | D | Z | S | ΔZ | ΔS | ρ* S vs D | ρ* S vs Z (= 1/R) | ρ* ΔS vs ΔZ | ΔZ / Z | ΔS / Z |
|---|---|---|---|---|---|---|---|---|---|---|
| rung 3 | 24.0 / 24.0 | 1.4 / 1.4 | 2.7 / 2.7 | 2.0 / 2.0 | 3.5 / 3.6 | 8.924 / 8.766 (cheaper) | 0.532 / 0.527 (depends) | 0.564 / 0.559 (depends) | 1.377 / 1.380 | 2.441 / 2.467 |
| rung 5 | 62.6 / 62.6 | 34.4 / 34.3 | 96.2 / 96.4 | 34.7 / 34.6 | 99.3 / 99.9 | 0.651 / 0.650 (depends) | 0.358 / 0.356 (depends) | 0.349 / 0.346 (depends) | 1.008 / 1.007 | 2.886 / 2.912 |
| rung 5 ×2 | 62.4 / 62.4 | 34.3 / 34.3 | 100.8 / 101.4 | 34.3 / 34.3 | 102.8 / 103.8 | 0.619 / 0.615 (depends) | 0.340 / 0.339 (depends) | 0.333 / 0.330 (depends) | 0.999 / 0.999 | 2.997 / 3.023 |

D to ΔS in millions of accumulations per token (12 blocks).

Rung 3's spike-native sites carry 35.4 M of the 92.0 M BitLinear MACs per
token; rung 5's carry all 92.0 M. Weight zeros are skipped by every engine:
`D` is the number of non-zero ternary weights (1 in 1.47 MACs).

## The SNN papers' accounting, decomposed

| model | spike-native MACs | synaptic ops | E ANN / E SNN, BitLinears | with the FP head | × multiplications removed | × activation zeros skipped | × spike count | weight zeros (not counted) |
|---|---|---|---|---|---|---|---|---|
| rung 3 | 35.4 M | 4.0 / 4.1 M | 1.60 / 1.60 | 1.36 / 1.36 | 5.11 | 16.43 / 16.27 | 0.533 / 0.528 | 1.47 / 1.47 |
| rung 5 | 92.0 M | 141.3 / 141.6 M | 3.33 / 3.32 | 1.97 / 1.97 | 5.11 | 1.82 / 1.83 | 0.358 / 0.356 | 1.47 / 1.47 |
| rung 5 ×2 | 92.0 M | 148.6 / 149.4 M | 3.16 / 3.15 | 1.93 / 1.93 | 5.11 | 1.82 / 1.82 | 0.341 / 0.339 | 1.48 / 1.47 |

Computed as Spikformer and Spike-driven Transformer compute it (sources in the
hardware note): synaptic operations = firing rate × T × the layer's MACs, which
is `E|q|` × MACs here, at 0.9 pJ, against the same MACs at 4.6 pJ for the FP
model; layers whose input is not spikes (rung 3's FP-fed BitLinears, the LM
head) are charged as MACs on both sides. The three factors multiply to the
ratio on the spike-native layers. Neither 0.9 nor 4.6 pJ is an integer price:
they are an FP32 add and an FP32 multiply-add (Horowitz, 45 nm), so an integer
engine has a further factor of 9–30 on the additions (0.03–0.1 pJ for an 8- to
32-bit integer add) that the spike reading shares.

## Outside the accumulations

Per token, by the convention of `fp_work_per_token` (same in every engine):

| work | rung 3 | rung 5 |
|---|---|---|
| FP LM head, MACs | 38.6 M | 38.6 M |
| FP-fed ternary BitLinears, FP additions | 38.6 M | – |
| group rescale of the BitLinear partial sums | 1.44 M | 1.44 M |
| LIF quantizers | 0.29 M | 0.48 M |
| WKV | 0.17 M | 0.17 M |
| LayerNorms, token-shift mixes, relu², gates, residuals, decoding | 0.42 M | 0.42 M |
| exp, sigmoid, divide, rsqrt | 65 k | 65 k |

At rung 5 the FP work outside the BitLinears and the head is 2.5 M operations
per token, 1.9 % of the matmul MACs, most of it the group rescale. The FP LM
head (29.6 % of the matmul MACs) is the same in every engine and sets a floor
under any whole-model saving; the hardware note prices it.

## Robustness

  * Seeds: every band is the same on both seeds. The largest seed differences
    are at relu² `lif_k` against D (14.3 vs 15.3 at rung 5, 15.1 vs 14.8 at
    rung 3), far inside its band; elsewhere `ρ*` differs by at most 0.06 between seeds, and by 0.005 in the
    pooled S against Z.
  * Budget: rung 5 ×2 gives the same bands as rung 5 except the one ΔS / ΔZ
    case above; its input sites are slightly denser (`R` 2.94–2.95 vs
    2.80–2.81).
  * Rung 3 and rung 5 agree on the sites they share: `lif_out` within 0.06 in
    `ρ*` against D, relu² `lif_k` within its seed spread (14.3–15.3), both
    within 0.03 against Z.

## Limits

  * Counts, not energy: the bands bound the cost ratio a hardware needs, they
    do not say which hardware is built. The hardware note places published
    figures in them.
  * TinyStories only (tier 3 of DEC-24 would check general text), 32 k tokens
    per checkpoint, cold windows of 128 tokens.
  * Weight zeros are skipped by every engine at no cost; an engine that cannot
    skip them multiplies D, Z and S alike by 1.47.
  * The Δ engines are charged nothing for keeping the consumer's previous
    output; they are not worth more even so.
