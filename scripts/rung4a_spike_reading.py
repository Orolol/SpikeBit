"""Rung 4a (DECISIONS #19): spike reading of rung-3 checkpoints.

First 256 windows of 128 tokens of the eval tail, fresh state per window,
eval mode. For every FAI-LIF site (time-mix k, v, out; channel-mix k, v,
pooled over the blocks): E|q|, P(q != 0). For the spike-native sites (the
consumer is a ternary BitLinear: time-mix ``lif_out`` -> ``W_o``, channel-mix
``lif_k`` -> ``W_v``): accumulations of the ±1 spike reading,
``sum |q_i| * nnz(W[:, i])``, over those of the INT4 x ternary baseline,
``sum 1{q_i != 0} * nnz(W[:, i])``, i.e. ``R``. Also checks on real counts
that the spike reading gives the consumer's integer accumulations exactly,
and reports which share of the matmul work per token takes LIF outputs.
With ``act_quant_inputs`` (rung 5, DECISIONS #22) the five LIFs on the
token-shift mixes are spike-native too (consumers ``W_r``, ``W_k``, ``W_v`` of
the time-mix, ``W_k``, ``W_r`` of the channel-mix).

    uv run python scripts/rung4a_spike_reading.py rung3l_a4_s0 rung3l_a4_s1

Checkpoints under ``checkpoints/ladder/<run>/final.pt``; results also written
to ``runs/ladder/rung4a_<run>.json``.
"""
import json, sys
import torch
from spiking_ternary_lm.eval.perplexity import protocol_eval_corpus
from spiking_ternary_lm.model.config import SpikingTernaryRWKVConfig
from spiking_ternary_lm.model.model import SpikingTernaryRWKV
from spiking_ternary_lm.neurons.binary_lif import decompose_integer_to_binary, decompose_signed_to_spikes
from spiking_ternary_lm.neurons.fai_lif import FAILIF
from spiking_ternary_lm.quantization.bitlinear import BitLinear
from spiking_ternary_lm.training.data import tokenize_tinystories

torch.set_grad_enabled(False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ, N_WIN, B = 128, 256, 16
SITES = ["time_mix.lif_k", "time_mix.lif_v", "time_mix.lif_out", "channel_mix.lif_k", "channel_mix.lif_v"]
CONSUMER = {"time_mix.lif_out": "time_mix.W_o", "channel_mix.lif_k": "channel_mix.W_v"}
INPUT_CONSUMER = {"time_mix.lif_xr": "time_mix.W_r", "time_mix.lif_xk": "time_mix.W_k",
                  "time_mix.lif_xv": "time_mix.W_v", "channel_mix.lif_xk": "channel_mix.W_k",
                  "channel_mix.lif_xr": "channel_mix.W_r"}
ids = torch.from_numpy(protocol_eval_corpus(tokenize_tinystories(tokenizer_id="RWKV/rwkv-4-169m-pile")).ids.astype("int64"))
windows = torch.stack([ids[w * SEQ:(w + 1) * SEQ] for w in range(N_WIN)])


def spikes_of(q, lif):
    """±1 spikes for signed grids (T = 8 at 4 bits), {0, 1} for unsigned (T = 15)."""
    bits = lif.num_bits
    if lif.observer.qmin < 0:
        return decompose_signed_to_spikes(q, bits=bits)
    return decompose_integer_to_binary(q, bits=bits)


for run in sys.argv[1:]:
    ck = torch.load(f"checkpoints/ladder/{run}/final.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = SpikingTernaryRWKV(SpikingTernaryRWKVConfig(**cfg)); m.load_state_dict(ck["model"]); m.to(dev).eval()
    consumer = dict(CONSUMER, **INPUT_CONSUMER) if cfg.get("act_quant_inputs") else CONSUMER
    sites = SITES + list(INPUT_CONSUMER) if cfg.get("act_quant_inputs") else SITES
    stats = {s: torch.zeros(5, dtype=torch.float64, device=dev) for s in sites}  # n, nz, sum|q|, acc spike, acc base
    nnz_col, lifs, exact = {}, {}, {}
    for i, blk in enumerate(m.blocks):
        for site in sites:
            lifs[(i, site)] = blk.get_submodule(site)
        for site, cons in consumer.items():
            ternary, _ = blk.get_submodule(cons).pack_weights()
            nnz_col[(i, site)] = (ternary != 0).sum(0).to(torch.float64)
    first = True
    for b0 in range(0, N_WIN, B):
        m.reset_state()
        m(windows[b0:b0 + B].to(dev))
        for (i, site), lif in lifs.items():
            q = lif.last_spike
            assert q is not None and torch.equal(q, q.round()), f"{run} block {i} {site}: no integer counts"
            q = q.to(torch.float64)
            st = stats[site]
            st[0] += q.numel(); st[1] += (q != 0).sum(); st[2] += q.abs().sum()
            if site in consumer:
                st[3] += (q.abs() * nnz_col[(i, site)]).sum()
                st[4] += ((q != 0).to(torch.float64) * nnz_col[(i, site)]).sum()
                if first and i in (0, len(m.blocks) - 1):
                    # Exactness on real counts (first two windows): per-step accumulations
                    # through the ternary consumer sum to the integer-path ones.
                    t = m.blocks[i].get_submodule(consumer[site]).pack_weights()[0].to(torch.float64)
                    sp = spikes_of(q[:2], lif)
                    per_step = torch.stack([s @ t.T for s in sp]).sum(0)
                    exact[f"block{i}.{site}"] = bool(torch.equal(per_step, q[:2] @ t.T))
        first = False

    out = {"run": run, "act_quant": cfg.get("act_quant"), "bits": cfg.get("bits"),
           "windows": N_WIN, "seq": SEQ, "exact_on_real_counts": exact, "sites": {}}
    for site, st in stats.items():
        n, nz, sabs, acc_s, acc_b = (float(v) for v in st)
        rec = {"elements": n, "E_abs_q": sabs / n, "P_nonzero": nz / n, "E_abs_q_given_nonzero": sabs / nz,
               "spike_native": site in consumer}
        if site in consumer:
            rec.update(acc_spike=acc_s, acc_int4=acc_b, R=acc_s / acc_b)
        out["sites"][site] = rec
    acc_s = sum(out["sites"][s]["acc_spike"] for s in consumer)
    acc_b = sum(out["sites"][s]["acc_int4"] for s in consumer)
    out["R_pooled"] = acc_s / acc_b
    n_all = sum(r["elements"] for r in out["sites"].values())
    out["spike_native_share_of_lif_outputs"] = sum(out["sites"][s]["elements"] for s in consumer) / n_all
    # Matmul work per token: ternary BitLinears (LIF-fed or FP-fed) and the FP LM head.
    macs = {"bitlinear_lif_input": 0, "bitlinear_fp_input": 0}
    consumers = {id(blk.get_submodule(c)) for blk in m.blocks for c in consumer.values()}
    for mod in m.modules():
        if isinstance(mod, BitLinear):
            key = "bitlinear_lif_input" if id(mod) in consumers else "bitlinear_fp_input"
            macs[key] += mod.in_features * mod.out_features
    macs["lm_head_fp"] = m.lm_head.in_features * m.lm_head.out_features
    out["macs_per_token"] = macs
    print(f"=== {run} ({out['act_quant']}, {out['bits']}-bit) exact on real counts: {exact}")
    for site, r in out["sites"].items():
        extra = f"  R {r['R']:.3f}" if "R" in r else ""
        print(f"  {site:18s} E|q| {r['E_abs_q']:.3f}  P(q!=0) {r['P_nonzero']:.3f}  "
              f"E|q| | q!=0 {r['E_abs_q_given_nonzero']:.3f}{extra}")
    tot = sum(macs.values())
    print(f"  R pooled {out['R_pooled']:.3f} (R/4 {out['R_pooled'] / 4:.3f}); spike-native share of LIF outputs "
          f"{out['spike_native_share_of_lif_outputs']:.3f}; MACs/token: LIF-fed ternary "
          f"{macs['bitlinear_lif_input'] / tot:.3f}, FP-fed ternary {macs['bitlinear_fp_input'] / tot:.3f}, "
          f"FP head {macs['lm_head_fp'] / tot:.3f}")
    with open(f"runs/ladder/rung4a_{run}.json", "w") as f:
        json.dump(out, f, indent=1)
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
