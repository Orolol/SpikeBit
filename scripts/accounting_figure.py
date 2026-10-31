"""Figure 1 of the write-up (DECISIONS #24): break-even cost ratio rho* per site, rung 5.

From ``runs/accounting/rung5_a4_s{0,1}.json`` (spike_accounting.py). One row per
spike-native site plus the pooled row; for each baseline (S vs D, S vs Z, dS vs dZ)
one marker per seed. Spikes are cheaper than the baseline when a spike accumulation
costs less than rho* times a 4-bit one. Shaded: between the two reference pricings
rho = 1/4 and rho = 1 (DEC-24's "depends on hardware" band); hatched: design (a) of results/hardware_cost_note.md, one event per
spike with a weight read per event (rho_eff >= 0.82).

    uv run --with matplotlib==3.10.7 python scripts/accounting_figure.py

Writes writeup/figures/rho_star.pdf and .png.
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "rung5_a4"
# Rows top to bottom: the two sparse sites, the five input sites, then pooled.
SITES = [("channel_mix.lif_k", "channel-mix relu² out → $W_v$"),
         ("time_mix.lif_out", "time-mix out → $W_o$"),
         ("time_mix.lif_xr", "time-mix input $x_r$ → $W_r$"),
         ("time_mix.lif_xk", "time-mix input $x_k$ → $W_k$"),
         ("time_mix.lif_xv", "time-mix input $x_v$ → $W_v$"),
         ("channel_mix.lif_xk", "channel-mix input $x_k$ → $W_k$"),
         ("channel_mix.lif_xr", "channel-mix input $x_r$ → $W_r$")]
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light);
# marker shapes carry the identity too.
BASELINES = [("S_vs_D", "vs dense integer (D)", "#2a78d6", "o"),
             ("S_vs_Z", "vs zero-skipping integer (Z)", "#eb6834", "s"),
             ("dS_vs_dZ", "Δ spikes vs Δ integer (ΔS vs ΔZ)", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

recs = [json.load(open(f"runs/accounting/{RUN}_s{s}.json")) for s in (0, 1)]
rows = [(label + f"  ({100 * recs[0]['sites'][key]['P_nonzero']:.0f}% non-zero)",
         [r["sites"][key] for r in recs]) for key, label in SITES]
rows.append(("pooled", [r["pooled_spike_native"] for r in recs]))

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.set_xscale("log")
ax.set_xlim(0.2, 25)
ax.axvspan(0.25, 1.0, color="#efeeea", zorder=0, lw=0)
ax.axvspan(0.82, 1.0, facecolor="none", edgecolor="#a3a29c", hatch="////", lw=0, zorder=0.5)
for x in (0.25, 1.0):
    ax.axvline(x, color="#a3a29c", lw=0.8, zorder=1)
n = len(rows)
for i, (label, rs) in enumerate(rows):
    y = n - 1 - i
    if i == n - 1:
        ax.axhline(y + 0.5, color=GRID, lw=0.8)
    for j, (key, _, color, marker) in enumerate(BASELINES):
        yy = y + (1 - j) * 0.22
        xs = [r["rho_star"][key] for r in rs]
        ax.plot(xs, [yy] * len(xs), ls="none", marker=marker, ms=6, mfc=color, mec="white", mew=0.8, zorder=3)
ax.set_yticks(range(n))
ax.set_yticklabels([label for label, _ in rows][::-1], fontsize=8, color=INK)
ax.set_ylim(-0.6, n - 0.4)
ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20])
ax.set_xticklabels(["¼", "½", "1", "2", "5", "10", "20"], fontsize=8, color=INK)
ax.minorticks_off()
ax.set_xlabel("break-even cost ratio ρ* (spike accumulation / 4-bit accumulation)", fontsize=8, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#a3a29c")
ax.tick_params(color="#a3a29c", length=3)
top = n - 0.45
ax.text(0.5, top, "reference pricings", fontsize=7, color=MUTED, va="bottom", ha="center")
ax.text(0.905, (n - 1) / 2 + 0.3, "design (a): event per spike, $\\rho_{\\mathrm{eff}} \\geq 0.82$", fontsize=7, color=MUTED, va="center",
        ha="center", rotation=90, bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=2)
ax.text(4.5, top, "spikes need fewer accumulations", fontsize=7, color=MUTED, va="bottom", ha="center")
handles = [plt.Line2D([], [], ls="none", marker=m, ms=6, mfc=c, mec="white", mew=0.8, label=lab)
           for _, lab, c, m in BASELINES]
ax.legend(handles=handles, loc="center right", fontsize=7, frameon=False, labelcolor=INK,
          title="unit spikes (one marker per seed)", title_fontsize=7)
fig.tight_layout()
os.makedirs("writeup/figures", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"writeup/figures/rho_star.{ext}", dpi=200)
print("wrote writeup/figures/rho_star.pdf and .png")
