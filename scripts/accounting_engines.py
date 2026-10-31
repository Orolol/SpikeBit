"""Counts on the float graph's trace against counts on the integer engine's own trace (DEC-24).

Reads ``runs/accounting/<run>.json`` (spike_accounting.py, default) and
``runs/accounting/integer/<run>.json`` (``ENGINE=integer``) and prints, per run, the largest
change of every ratio the write-up quotes (rho* for S/D, S/Z, dS/dZ per spike-native site and
pooled; R; dZ/Z; P(q != 0); E|q|; the SNN papers' factors), whether any verdict band changes,
and the largest per-group integer sum.

    uv run python scripts/accounting_engines.py rung5_a4_s0 rung5_a4_s1 ...
"""
import json, sys

PAIRS = ["S_vs_D", "S_vs_Z", "dS_vs_dZ"]


def ratios(d):
    """Flat {name: value} of the quoted ratios, and {name: band}."""
    vals, bands = {}, {}
    entries = {s: r for s, r in d["sites"].items() if r.get("spike_native")}
    entries["pooled"] = d["pooled_spike_native"]
    for site, r in entries.items():
        for k in PAIRS:
            vals[f"{site} rho* {k}"] = r["rho_star"][k]
            bands[f"{site} {k}"] = r["band"][k]
        vals[f"{site} dZ/Z"] = r["cross"]["dZ/Z"]
        vals[f"{site} P(q!=0)"] = r["P_nonzero"]
        vals[f"{site} E|q|"] = r["E_abs_q"]
    vals["R pooled"] = d["R_pooled"]
    for k, v in d["snn_papers_accounting"]["factors_spike_native"].items():
        vals[f"factor {k}"] = v
    return vals, bands


worst = 0.0
for run in sys.argv[1:]:
    fl = json.load(open(f"runs/accounting/{run}.json"))
    it = json.load(open(f"runs/accounting/integer/{run}.json"))
    # Tier-1 files predate the corpus field.
    assert it["engine"] == "integer" and fl["windows"] == it["windows"]
    assert fl.get("corpus", "tinystories") == it["corpus"]
    (vf, bf), (vi, bi) = ratios(fl), ratios(it)
    diffs = sorted(((abs(vi[k] - vf[k]), k) for k in vf), reverse=True)
    rel = max(abs(vi[k] / vf[k] - 1) for k in vf if vf[k])
    changed = [k for k in bf if bf[k] != bi[k]]
    worst = max(worst, diffs[0][0])
    print(f"{run} ({it['corpus']}, {it['windows']} windows): largest |change| {diffs[0][0]:.4f} "
          f"({diffs[0][1]}), largest relative change {100 * rel:.2f} %, bands changed: "
          f"{changed or 'none'}; largest |group sum| {it['integer_engine']['max_abs_group_sum']:.0f}")
    for d, k in diffs[1:3]:
        print(f"    next: {d:.4f} ({k})")
print(f"largest |change| over all runs: {worst:.4f}")
