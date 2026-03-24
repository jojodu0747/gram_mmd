"""
replot_sdvae_rho_gamma.py
=========================
Regénère uniquement la figure 08_sdvae_rho_vs_gamma.png en ajoutant
le marqueur de la moyenne heuristique (carré ■) en plus de la médiane (▲).

Lit les données depuis :
  - results/sweep_coco_degraded/plots/sdvae_gamma_sweep.csv
  - results/sweep_coco_degraded/cache/sd_vae_l??_Drr_ref200_std.npy

Usage :
    .venv/bin/python experiments/replot_sdvae_rho_gamma.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR  = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "results" / "sweep_coco_degraded" / "cache"
CSV_PATH  = BASE_DIR / "results" / "sweep_coco_degraded" / "plots" / "sdvae_gamma_sweep.csv"
OUT_DIR   = BASE_DIR / "results" / "sweep_coco_degraded" / "plots"

LAYER_NAMES = {
    0:  "conv_in",
    1:  "down.0.res.0",
    2:  "down.0.res.1",
    3:  "down.0.ds",
    4:  "down.1.res.0",
    5:  "down.1.res.1",
    6:  "down.1.ds",
    7:  "down.2.res.0",
    8:  "down.2.res.1",
    9:  "down.2.ds",
    10: "down.3.res.0",
    11: "down.3.res.1",
    12: "mid.res.0",
    13: "mid.attn",
    14: "mid.res.1",
    15: "conv_norm_out",
    16: "conv_out",
}

# ── Chargement CSV ────────────────────────────────────────────────────

print(f"Chargement : {CSV_PATH.name}")
results = defaultdict(lambda: {"gamma": [], "rho": []})

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        l   = int(row["layer"])
        g   = float(row["gamma"])
        rho = row["spearman_rho"]
        results[l]["gamma"].append(g)
        results[l]["rho"].append(float(rho) if rho != "" else np.nan)

# Convertir en arrays triés par gamma
for l in results:
    order = np.argsort(results[l]["gamma"])
    results[l]["gamma"] = np.array(results[l]["gamma"])[order]
    results[l]["rho"]   = np.array(results[l]["rho"])[order]

layers = sorted(results.keys())
gammas = results[layers[0]]["gamma"]   # même grille pour tous
print(f"  {len(layers)} layers  ·  {len(gammas)} gammas")

# ── Calcul des gammas heuristiques (médiane + moyenne) ───────────────

median_gammas    = {}
median10x_gammas = {}

for l in layers:
    drr_path = CACHE_DIR / f"sd_vae_l{l:02d}_Drr_ref200_std.npy"
    if not drr_path.exists():
        continue
    D_rr    = np.load(drr_path).astype(np.float64)
    nonzero = D_rr[D_rr > 0]
    if len(nonzero) == 0:
        continue
    g_med = float(1.0 / (2.0 * np.median(nonzero)))
    median_gammas[l]    = g_med
    median10x_gammas[l] = 50.0 * g_med

print(f"  Médiane heuristique calculée pour {len(median_gammas)} layers")
print(f"  50×médiane calculée pour {len(median10x_gammas)} layers")

# ── Figure ────────────────────────────────────────────────────────────

n_layers = len(layers)
cmap     = plt.get_cmap("plasma", n_layers)

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_facecolor("#f9f9f9")

for idx, l in enumerate(layers):
    rhos  = results[l]["rho"]
    valid = np.isfinite(rhos)
    if not valid.any():
        continue

    ax.plot(gammas[valid], rhos[valid],
            color=cmap(idx), linewidth=1.4, alpha=0.85,
            label=f"L{l:02d} {LAYER_NAMES.get(l, '')}")

    # ○ — optimal γ (meilleur ρ)
    best_idx = int(np.nanargmax(rhos))
    ax.scatter([gammas[best_idx]], [rhos[best_idx]],
               color=cmap(idx), s=40, zorder=5,
               marker="o", edgecolors="black", linewidths=0.4)

    # ▲ — médiane heuristique γ
    if l in median_gammas:
        gi = int(np.argmin(np.abs(gammas - median_gammas[l])))
        rho_med = float(rhos[gi]) if np.isfinite(rhos[gi]) else np.nan
        if np.isfinite(rho_med):
            ax.scatter([gammas[gi]], [rho_med],
                       color=cmap(idx), s=70, zorder=6,
                       marker="^", edgecolors="black", linewidths=0.6)

    # ■ — 10× médiane heuristique
    if l in median10x_gammas:
        gi = int(np.argmin(np.abs(gammas - median10x_gammas[l])))
        rho_50x = float(rhos[gi]) if np.isfinite(rhos[gi]) else np.nan
        if np.isfinite(rho_50x):
            ax.scatter([gammas[gi]], [rho_50x],
                       color=cmap(idx), s=70, zorder=6,
                       marker="s", edgecolors="black", linewidths=0.6)

ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("γ (log scale)", fontsize=12)
ax.set_ylabel("Spearman ρ", fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.set_title(
    "SD-VAE — Spearman ρ as a function of γ  (1 000 values, log grid)\n"
    "COCO-degraded  ·  anchor = 200 clean refs  ·  standardisation ON",
    fontsize=13, fontweight="bold",
)

# Légende layers
layer_legend = ax.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.85)
ax.add_artist(layer_legend)

# Légende marqueurs
extra_handles = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None",
           markersize=6, markeredgecolor="black", markeredgewidth=0.4,
           label="Optimal γ (best ρ)"),
    Line2D([0], [0], marker="^", color="gray", linestyle="None",
           markersize=7, markeredgecolor="black", markeredgewidth=0.6,
           label="γ = 1/(2·median(D_rr))"),
    Line2D([0], [0], marker="s", color="gray", linestyle="None",
           markersize=7, markeredgecolor="black", markeredgewidth=0.6,
           label="γ = 50 × 1/(2·median(D_rr))"),
]
ax.legend(handles=extra_handles, fontsize=8, loc="upper left",
          framealpha=0.9, bbox_to_anchor=(0.0, 1.0))

ax.grid(True, which="both", alpha=0.25)
plt.tight_layout()

out_path = OUT_DIR / "08_sdvae_rho_vs_gamma.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Sauvegardé : {out_path}")
