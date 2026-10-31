"""Markdown tables of DEC-24 tier 1 from ``runs/accounting/<run>.json`` (spike_accounting.py).

    uv run python scripts/accounting_tables.py

Values are given as seed 0 / seed 1 for each model (rung 3, rung 5, rung 5 x2, and
rung 5 trained and counted on WikiText-103, DEC-24 tier 3).
"""
import json

MODELS = {"rung 3": "rung3l_a4", "rung 5": "rung5_a4", "rung 5 ×2": "rung5x2_a4",
          "rung 5, WikiText-103": "wt103_rung5_a4"}
BAND = {"spikes cheaper": "cheaper", "spikes dearer": "dearer", "depends on hardware": "depends"}
runs = {m: [json.load(open(f"runs/accounting/{p}_s{s}.json")) for s in (0, 1)] for m, p in MODELS.items()}


def pair(vals, fmt="{:.3f}"):
    return " / ".join(fmt.format(v) for v in vals)


def bands(recs, key):
    b = {BAND[r["band"][key]] for r in recs}
    return "/".join(sorted(b))


print("### Per site\n")
print("| model | site | P(q ≠ 0) | E\\|q\\| | P(Δq ≠ 0) | ρ* S vs D | ρ* S vs Z | ρ* ΔS vs ΔZ | ΔZ / Z |")
print("|---|---|---|---|---|---|---|---|---|")
for m, rs in runs.items():
    for site, rec in rs[0]["sites"].items():
        if not rec["spike_native"]:
            continue
        recs = [r["sites"][site] for r in rs]
        cells = [pair(x[k] for x in recs) for k in ("P_nonzero", "E_abs_q", "P_dq_nonzero")]
        for key in ("S_vs_D", "S_vs_Z", "dS_vs_dZ"):
            cells.append(f"{pair(x['rho_star'][key] for x in recs)} ({bands(recs, key)})")
        cells.append(pair(x["cross"]["dZ/Z"] for x in recs))
        print(f"| {m} | `{site}` | " + " | ".join(cells) + " |")

print("\n### Pooled over the spike-native sites\n")
print("| model | D | Z | S | ΔZ | ΔS | ρ* S vs D | ρ* S vs Z (= 1/R) | ρ* ΔS vs ΔZ | ΔZ / Z | ΔS / Z |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m, rs in runs.items():
    recs = [r["pooled_spike_native"] for r in rs]
    tokens = rs[0]["windows"] * rs[0]["seq"]
    per_tok = [pair((x["counts"][k] / tokens / 1e6 for x in recs), "{:.1f}") for k in ("D", "Z", "S", "dZ", "dS")]
    rho = [f"{pair(x['rho_star'][k] for x in recs)} ({bands(recs, k)})" for k in ("S_vs_D", "S_vs_Z", "dS_vs_dZ")]
    cross = [pair(x["cross"][k] for x in recs) for k in ("dZ/Z", "dS/Z")]
    print(f"| {m} | " + " | ".join(per_tok + rho + cross) + " |")
print("\nD to ΔS in millions of accumulations per token (12 blocks).")

print("\n### The SNN papers' accounting\n")
print("| model | spike-native MACs | synaptic ops | E ANN / E SNN, BitLinears | with the FP head | "
      "× multiplications removed | × activation zeros skipped | × spike count | weight zeros (not counted) |")
print("|---|---|---|---|---|---|---|---|---|")
for m, rs in runs.items():
    sa = [r["snn_papers_accounting"] for r in rs]
    f = [x["factors_spike_native"] for x in sa]
    print(f"| {m} | {sa[0]['macs_per_token']['bitlinear_spike_native'] / 1e6:.1f} M | "
          f"{pair((x['synaptic_ops_per_token'] / 1e6 for x in sa), '{:.1f}')} M | "
          f"{pair((x['ratio_ann_over_snn']['no_head'] for x in sa), '{:.2f}')} | "
          f"{pair((x['ratio_ann_over_snn']['with_head'] for x in sa), '{:.2f}')} | "
          f"{f[0]['multiplication_removed']:.2f} | {pair((x['activation_zeros_skipped'] for x in f), '{:.2f}')} | "
          f"{pair((x['spike_count'] for x in f), '{:.3f}')} | "
          f"{pair((x['weight_zeros_skipped_not_in_their_count'] for x in sa), '{:.2f}')} |")

print("\n### FP work per token outside the integer accumulations\n")
for m, rs in runs.items():
    fp = rs[0]["fp_work_per_token"]
    ops = ", ".join(f"{k} {v / 1e6:.2f} M" for k, v in fp["ops"].items() if v)
    print(f"  * {m}: LM head {fp['lm_head_fp_macs'] / 1e6:.1f} M FP MACs; {fp['ops_total'] / 1e6:.2f} M ops ({ops}); "
          f"{fp['special_total'] / 1e3:.0f} k exp/sigmoid/divide/rsqrt")
