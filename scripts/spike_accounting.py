"""Tier 1 of DECISIONS #24: the spike reading's accumulations against integer engines.

Extends ``rung4a_spike_reading.py`` on the same protocol: first 256 windows of 128
tokens of the eval tail, fresh state per window, eval mode. For each ternary
BitLinear consumer ``W`` (out x in) of a LIF output, with ``c_i = nnz(W[:, i])``
and the input counts ``q_i`` of one token, the accumulations per token of

  D        integer, dense activations     sum_i c_i
  Z        integer, zeros skipped          sum_i 1{q_i != 0} c_i   (DEC-19's baseline)
  S        the ±1 spike reading            sum_i |q_i| c_i
  dZ, dS   Z and S on dq_t = q_t - q_{t-1} within a window (dq_0 = q_0); the
           consumer keeps its output, y_t = y_{t-1} + W dq_t.

Per site (pooled over the blocks), per block and pooled per model: the counts,
P(q != 0), E|q|, P(dq != 0), E|dq|, ``rho* = ops_X / ops_S`` for S against D and
Z and dS against dZ with the verdict band of DEC-24 point 3, the SNN papers'
accounting and its factors (DEC-24 point 4), and the per-token FP work outside
the integer accumulations (``fp_work_per_token``). Before any of it, the pooled
S / Z must reproduce the 4a ``R`` to three decimals (``R_4A``), and on real
counts the spike readings of q and of dq must give the consumer's integer
outputs exactly.

    CUDA_VISIBLE_DEVICES= uv run python scripts/spike_accounting.py rung3l_a4_s0 rung5_a4_s0

Checkpoints under ``checkpoints/ladder/<run>/final.pt``; results in
``runs/accounting/<run>.json``. ``N_WIN`` (environment) shortens the pass for a
smoke test; the ``R`` check then does not apply. ``CORPUS=wikitext103`` reads the eval
tail of WikiText-103 instead of TinyStories (DEC-24 tier 3).

By default the counts are the float graph's trace (each BitLinear a float matmul on
``s * q``). ``ENGINE=integer`` runs every spike-native consumer as the integer engine of
``spiking_ternary_lm.eval.integer_engine`` (per-group integer sums, then the scales), so the
counts are the trace of the engine being priced; results then go to
``runs/accounting/integer/<run>.json``, with the largest |per-group integer sum| seen.
"""
import json, os, sys, time
import torch
from spiking_ternary_lm.eval import integer_engine
from spiking_ternary_lm.eval.perplexity import protocol_eval_corpus
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.binary_lif import decompose_integer_to_binary, decompose_signed_to_spikes
from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.quantization.bitlinear import BitLinear
from spiking_ternary_lm.training.data import load_corpus

torch.set_grad_enabled(False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ, B = 128, 16
N_WIN = int(os.environ.get("N_WIN", 256))
CORPUS = os.environ.get("CORPUS", "tinystories")  # wikitext103 for DEC-24 tier 3
ENGINE = os.environ.get("ENGINE", "float")  # integer: count the integer engine's own trace
assert ENGINE in ("float", "integer"), ENGINE
OUT_DIR = "runs/accounting" if ENGINE == "float" else "runs/accounting/integer"
SITES = ["time_mix.lif_k", "time_mix.lif_v", "time_mix.lif_out", "channel_mix.lif_k", "channel_mix.lif_v"]
CONSUMER = {"time_mix.lif_out": "time_mix.W_o", "channel_mix.lif_k": "channel_mix.W_v"}
INPUT_CONSUMER = {"time_mix.lif_xr": "time_mix.W_r", "time_mix.lif_xk": "time_mix.W_k",
                  "time_mix.lif_xv": "time_mix.W_v", "channel_mix.lif_xk": "channel_mix.W_k",
                  "channel_mix.lif_xr": "channel_mix.W_r"}
# Pooled R of the 4a measurement (results/stage_a_ladder.md, rungs 4a and 5).
R_4A = {"rung3l_a4_s0": 1.881, "rung3l_a4_s1": 1.897, "rung5_a4_s0": 2.796, "rung5_a4_s1": 2.810}
# Accumulators per (block, site): elements, nz, sum|q|, nz dq, sum|dq|, D, Z, S, dZ, dS.
FIELDS = ["n", "nz", "abs_q", "nz_dq", "abs_dq", "D", "Z", "S", "dZ", "dS"]
PAIRS = [("S", "D"), ("S", "Z"), ("dS", "dZ")]
E_MAC_FP32, E_AC = 4.6, 0.9  # pJ, the SNN papers' constants (Horowitz, 45 nm)


def band(rho_star):
    """DEC-24 point 3, with rho in [1/4, 1] (DEC-19 point 4's two pricings)."""
    if rho_star > 1:
        return "spikes cheaper"
    return "spikes dearer" if rho_star < 0.25 else "depends on hardware"


def spikes_of(q, lif):
    """±1 spikes for signed grids (T = 8 at 4 bits), {0, 1} for unsigned (T = 15)."""
    bits = lif.num_bits
    if lif.observer.qmin < 0:
        return decompose_signed_to_spikes(q, bits=bits)
    return decompose_integer_to_binary(q, bits=bits)


def delta(q):
    """dq_t = q_t - q_{t-1} along the token axis of (B, T, C), dq_0 = q_0."""
    return q - torch.nn.functional.pad(q, (0, 0, 1, 0))[:, :-1]


def fp_work_per_token(m, cfg, consumer_ids):
    """FP work per token outside the integer accumulations, by a stated convention.

    One op per add, sub, mul, max, compare or round; exp, sigmoid, divide and rsqrt
    counted apart as "special". LayerNorm: 7 ops per element (mean, centre,
    square, variance, normalise, weight, bias) and one rsqrt. Token-shift mix:
    3 (two multiplies, one add). LIF quantizer: 4 (scale, round, two clamps);
    decoding a count for a non-BitLinear consumer: 1. WKV (RWKV-4's stable form):
    18 ops and 5 special (4 exp, 1 divide) per channel. Group rescale of a
    BitLinear's partial sums: one multiply and one add per output and group.
    BitLinears fed an FP input (rung 3) accumulate FP values: counted as their
    nnz(W) FP additions.
    """
    H, F, L = cfg["hidden_size"], cfg["ffn_dim"], cfg["n_layers"]
    lif_elems = sum(mod.num_neurons for mod in m.modules() if isinstance(mod, FAILIF))
    bl = [mod for mod in m.modules() if isinstance(mod, BitLinear)]
    work = {
        "layernorm": (2 * L + 2) * 7 * H,
        "residual": 2 * L * H,
        "token_shift_mix": L * 5 * 3 * H,
        "lif_quantize": 4 * lif_elems,
        "decode_for_fp_consumers": L * 3 * H,  # time-mix k, v into the WKV; channel-mix v into the gate
        "wkv": L * 18 * H,
        "relu_sq": L * 2 * F if cfg.get("ffn_relu_sq") else 0,
        "gates": L * 2 * H,  # sigmoid(r) * wkv, sigmoid(W_r xr) * v
        "group_rescale": sum(2 * mod.out_features * mod.num_groups for mod in bl),
        "fp_input_ternary_adds": sum(int((mod.pack_weights()[0] != 0).sum()) for mod in bl
                                     if id(mod) not in consumer_ids),
    }
    special = {"layernorm_rsqrt": 2 * L + 2, "wkv_exp_div": L * 5 * H, "sigmoid": L * 2 * H}
    return {"ops": work, "ops_total": sum(work.values()), "special": special,
            "special_total": sum(special.values()),
            "lm_head_fp_macs": m.lm_head.in_features * m.lm_head.out_features}


ids = torch.from_numpy(protocol_eval_corpus(load_corpus(CORPUS)).ids.astype("int64"))
windows = torch.stack([ids[w * SEQ:(w + 1) * SEQ] for w in range(N_WIN)])
os.makedirs(OUT_DIR, exist_ok=True)

for run in sys.argv[1:]:
    t0 = time.time()
    ck = torch.load(f"checkpoints/ladder/{run}/final.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(**cfg)); m.load_state_dict(ck["model"]); m.to(dev).eval()
    consumer = dict(CONSUMER, **INPUT_CONSUMER) if cfg.get("act_quant_inputs") else CONSUMER
    sites = SITES + list(INPUT_CONSUMER) if cfg.get("act_quant_inputs") else SITES
    L = len(m.blocks)
    acc = {(i, s): torch.zeros(len(FIELDS), dtype=torch.float64, device=dev) for i in range(L) for s in sites}
    nnz_col, lifs, shape, exact = {}, {}, {}, {}
    for i, blk in enumerate(m.blocks):
        for site in sites:
            lifs[(i, site)] = blk.get_submodule(site)
        for site, cons in consumer.items():
            lif, bl = blk.get_submodule(site), blk.get_submodule(cons)
            # The counts below take q itself as the consumer's integer input.
            assert lif.act_quant == "lsq" and float(lif.lsq_zp.abs().max()) == 0.0, f"{run} {i} {site}: zp != 0"
            nnz_col[(i, site)] = (bl.pack_weights()[0] != 0).sum(0).to(torch.float64).to(dev)
            shape[site] = (bl.out_features, bl.in_features)
    engine_stats = None
    if ENGINE == "integer":
        _, engine_stats = integer_engine.install(m, {cons: site for site, cons in consumer.items()})
    first = True
    for b0 in range(0, N_WIN, B):
        m.reset_state()
        m(windows[b0:b0 + B].to(dev))
        for (i, site), lif in lifs.items():
            q = lif.last_spike
            assert q is not None and torch.equal(q, q.round()), f"{run} block {i} {site}: no integer counts"
            q = q.to(torch.float64)
            dq = delta(q)
            a = acc[(i, site)]
            a[0] += q.numel(); a[1] += (q != 0).sum(); a[2] += q.abs().sum()
            a[3] += (dq != 0).sum(); a[4] += dq.abs().sum()
            if site in consumer:
                c = nnz_col[(i, site)]
                a[5] += q.shape[0] * q.shape[1] * c.sum()
                a[6] += ((q != 0).to(torch.float64) * c).sum(); a[7] += (q.abs() * c).sum()
                a[8] += ((dq != 0).to(torch.float64) * c).sum(); a[9] += (dq.abs() * c).sum()
                if first and i in (0, L - 1):
                    # Exactness on real counts (first two windows): the per-step
                    # accumulations of the spikes of q, and the running sum of those
                    # of dq (signed 5-bit, T = 16), give the integer-path outputs.
                    t = m.blocks[i].get_submodule(consumer[site]).pack_weights()[0].to(torch.float64).to(dev)
                    ref = q[:2] @ t.T
                    per_step = torch.stack([s @ t.T for s in spikes_of(q[:2], lif)]).sum(0)
                    d_step = torch.stack([s @ t.T for s in decompose_signed_to_spikes(dq[:2], bits=5)]).sum(0)
                    exact[f"block{i}.{site}"] = {"q": bool(torch.equal(per_step, ref)),
                                                 "dq": bool(torch.equal(d_step.cumsum(1), ref))}
        first = False
        print(f"  {run}: {b0 + B}/{N_WIN} windows, {time.time() - t0:.0f} s", flush=True)

    def summarize(vec, macs=None):
        r = dict(zip(FIELDS, (float(v) for v in vec)))
        n = r["n"]
        rec = {"elements": n, "P_nonzero": r["nz"] / n, "E_abs_q": r["abs_q"] / n,
               "P_dq_nonzero": r["nz_dq"] / n, "E_abs_dq": r["abs_dq"] / n}
        if r["D"] > 0:
            if macs:  # tokens = elements / in_features (summed over the blocks when pooled)
                rec["acc_per_token"] = {k: r[k] / (n / macs[1]) for k in ("D", "Z", "S", "dZ", "dS")}
            rec["counts"] = {k: r[k] for k in ("D", "Z", "S", "dZ", "dS")}
            rec["rho_star"] = {f"{x}_vs_{y}": r[y] / r[x] for x, y in PAIRS}
            rec["band"] = {k: band(v) for k, v in rec["rho_star"].items()}
            rec["cross"] = {"dS/Z": r["dS"] / r["Z"], "dZ/Z": r["dZ"] / r["Z"], "dS/S": r["dS"] / r["S"],
                            "Z/D": r["Z"] / r["D"], "dZ/D": r["dZ"] / r["D"]}
        return rec

    out = {"run": run, "corpus": CORPUS, "engine": ENGINE, "act_quant": cfg.get("act_quant"), "bits": cfg.get("bits"), "windows": N_WIN,
           "seq": SEQ, "device": dev.type, "exact_on_real_counts": exact, "sites": {}, "per_block": {}}
    for site in sites:
        pooled = sum(acc[(i, site)] for i in range(L))
        out["sites"][site] = summarize(pooled, shape.get(site))
        out["sites"][site]["spike_native"] = site in consumer
        if site in consumer:
            out["sites"][site]["consumer_shape"] = shape[site]
            out["per_block"][site] = [summarize(acc[(i, site)], shape[site]) for i in range(L)]
    sn = sum(acc[(i, s)] for i in range(L) for s in consumer)
    out["pooled_spike_native"] = summarize(sn)
    R = out["pooled_spike_native"]["rho_star"]["S_vs_Z"] ** -1
    out["R_pooled"] = R
    if engine_stats is not None:
        out["integer_engine"] = {"max_abs_group_sum": engine_stats["max_acc"],
                                 "max_input_err_vs_s_times_q": engine_stats["max_input_err"]}
    if N_WIN == 256 and CORPUS == "tinystories" and ENGINE == "float" and run in R_4A:
        out["reproduces_4a"] = abs(R - R_4A[run]) <= 5e-4
        if not out["reproduces_4a"]:
            raise SystemExit(f"{run}: R pooled {R:.4f} does not reproduce the 4a value {R_4A[run]}")
    if not all(v["q"] and v["dq"] for v in exact.values()):
        raise SystemExit(f"{run}: a spike reading is not exact on real counts: {exact}")

    # The SNN papers' accounting (DEC-24 point 4): synaptic ops = firing rate x T x MACs
    # = E|q| x MACs per token at E_AC, against the same MACs at E_MAC_FP32; no
    # weight-zero skipping. Factors along F -> Z -> S without weight skipping.
    consumer_ids = {id(blk.get_submodule(c)) for blk in m.blocks for c in consumer.values()}
    fp = fp_work_per_token(m, cfg, consumer_ids)
    macs_sn = sum(shape[s][0] * shape[s][1] for s in consumer) * L
    sop = sum(out["sites"][s]["E_abs_q"] * shape[s][0] * shape[s][1] * L for s in consumer)
    nz_macs = sum(out["sites"][s]["P_nonzero"] * shape[s][0] * shape[s][1] * L for s in consumer)
    macs_bl = sum(mod.in_features * mod.out_features for mod in m.modules() if isinstance(mod, BitLinear))
    macs_fp_in = macs_bl - macs_sn
    head = fp["lm_head_fp_macs"]
    e_ann = {"no_head": E_MAC_FP32 * macs_bl, "with_head": E_MAC_FP32 * (macs_bl + head)}
    e_snn = {"no_head": E_AC * sop + E_MAC_FP32 * macs_fp_in, "with_head": E_AC * sop + E_MAC_FP32 * (macs_fp_in + head)}
    out["snn_papers_accounting"] = {
        "macs_per_token": {"bitlinear_spike_native": macs_sn, "bitlinear_fp_input": macs_fp_in, "lm_head_fp": head},
        "synaptic_ops_per_token": sop,
        "E_pJ_per_token": {"ann_fp32": e_ann, "snn": e_snn},
        "ratio_ann_over_snn": {k: e_ann[k] / e_snn[k] for k in e_ann},
        "factors_spike_native": {"multiplication_removed": E_MAC_FP32 / E_AC,
                                 "activation_zeros_skipped": macs_sn / nz_macs,
                                 "spike_count": nz_macs / sop,
                                 "product": (E_MAC_FP32 / E_AC) * macs_sn / sop},
        "weight_zeros_skipped_not_in_their_count": macs_sn / sum(float(c.sum()) for c in nnz_col.values()),
    }
    out["fp_work_per_token"] = fp
    out["seconds"] = time.time() - t0

    print(f"=== {run} ({out['act_quant']}, {out['bits']}-bit, {N_WIN} windows, {dev.type}) "
          f"exact on real counts: {all(v['q'] and v['dq'] for v in exact.values())} ({len(exact)} checks)")
    print(f"  {'site':20s} {'P(q!=0)':>8s} {'E|q|':>6s} {'P(dq!=0)':>8s} {'E|dq|':>6s}"
          f"  rho* S/D  rho* S/Z  rho* dS/dZ   dZ/Z   dS/Z")
    for site, r in out["sites"].items():
        line = f"  {site:20s} {r['P_nonzero']:8.3f} {r['E_abs_q']:6.3f} {r['P_dq_nonzero']:8.3f} {r['E_abs_dq']:6.3f}"
        if "rho_star" in r:
            rs, cr = r["rho_star"], r["cross"]
            line += (f"  {rs['S_vs_D']:8.3f}  {rs['S_vs_Z']:8.3f}  {rs['dS_vs_dZ']:10.3f}"
                     f"  {cr['dZ/Z']:5.3f}  {cr['dS/Z']:5.3f}")
        print(line)
    p = out["pooled_spike_native"]
    print(f"  pooled spike-native: R {R:.4f}; rho* S/D {p['rho_star']['S_vs_D']:.3f} ({p['band']['S_vs_D']}), "
          f"S/Z {p['rho_star']['S_vs_Z']:.3f} ({p['band']['S_vs_Z']}), dS/dZ {p['rho_star']['dS_vs_dZ']:.3f} "
          f"({p['band']['dS_vs_dZ']}); dZ/Z {p['cross']['dZ/Z']:.3f}, dS/Z {p['cross']['dS/Z']:.3f}")
    sa = out["snn_papers_accounting"]
    print(f"  SNN papers' accounting: ANN/SNN {sa['ratio_ann_over_snn']['no_head']:.2f} (BitLinears), "
          f"{sa['ratio_ann_over_snn']['with_head']:.2f} (with the FP head); spike-native factors "
          + ", ".join(f"{k} {v:.3f}" for k, v in sa["factors_spike_native"].items()))
    print(f"  FP work/token: {fp['ops_total'] / 1e6:.2f} M ops + {fp['special_total'] / 1e6:.3f} M special; "
          f"head {head / 1e6:.1f} M FP MACs; {out['seconds']:.0f} s")
    if engine_stats is not None:
        print(f"  integer engine: largest |group sum| {engine_stats['max_acc']:.0f}")
    with open(f"{OUT_DIR}/{run}.json", "w") as f:
        json.dump(out, f, indent=1)
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
