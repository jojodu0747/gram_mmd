#!/usr/bin/env python3
"""Regenerate 08_gamma_curve_sd_vae_v15.png from rho_curves_sd_vae.npz."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from gram_mmd.config import BACKBONE_CONFIGS

NPZ_PATH = BASE_DIR / "results" / "synthetic_kadid_sweep" / "plots_new" / "rho_curves_sd_vae.npz"
OUT_PATH = BASE_DIR / "results" / "synthetic_kadid_sweep" / "plots_new" / "08_gamma_curve_sd_vae_v15.png"

LAYER_NAMES = BACKBONE_CONFIGS.get("sd_vae", {}).get("layer_names", {})


def main():
    d = np.load(NPZ_PATH)
    layers = list(d["layers"])
    n = len(layers)
    cmap = plt.get_cmap("plasma", n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#f9f9f9")

    for idx, l in enumerate(layers):
        gammas = d[f"{l}_gammas"]
        rho    = d[f"{l}_rho"]
        gmed   = float(d[f"{l}_gmed"])
        name   = LAYER_NAMES.get(int(l), "?")
        color  = cmap(idx)

        valid = np.isfinite(rho)
        ax.plot(gammas[valid], rho[valid], color=color, linewidth=1.4,
                alpha=0.85, label=f"L{int(l):02d} {name}")

        best_i = int(np.nanargmax(rho))
        ax.scatter([gammas[best_i]], [rho[best_i]], color=color, s=40,
                   marker="o", edgecolors="black", linewidths=0.4, zorder=5)

        g_med_rbf = 1.0 / (2.0 * gmed)
        med_i = int(np.argmin(np.abs(gammas - g_med_rbf)))
        if np.isfinite(rho[med_i]):
            ax.scatter([gammas[med_i]], [rho[med_i]], color=color, s=70,
                       marker="^", edgecolors="black", linewidths=0.6, zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel("γ (log scale)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("sd_vae — Spearman ρ as a function of γ", fontsize=17, fontweight="bold", pad=15)
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax.axhline(0.8, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    ncol = 2 if n > 10 else 1
    layer_leg = ax.legend(fontsize=7.5, ncol=ncol, loc="lower left", framealpha=0.85)
    ax.add_artist(layer_leg)

    extra = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=6, markeredgecolor="black", markeredgewidth=0.4,
               label="Optimal γ (best ρ)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=7, markeredgecolor="black", markeredgewidth=0.6,
               label="γ = 1/(2·median(D²))"),
    ]
    ax.legend(handles=extra, fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
