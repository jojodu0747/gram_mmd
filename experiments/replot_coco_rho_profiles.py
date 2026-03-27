#!/usr/bin/env python3
"""Replot 01_rho_profiles.png from existing CSV — no sweep needed."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from gram_mmd.config import BACKBONE_CONFIGS

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path,
    default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "summary_full_sweep.csv")
parser.add_argument("--outdir", type=Path,
    default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "plots")
args = parser.parse_args()

# Order: sd_vae, dc_ae, lpips_vgg, vgg19, dinov2 — NO resnet50
# dinov2 moved to position 5 (bottom-middle), legend bottom-right
BACKBONES = ["sd_vae", "dc_ae", "lpips_vgg", "vgg19", "dinov2_vitb14"]
BB_COLORS = {
    "sd_vae": "#e41a1c", "dc_ae": "#377eb8", "lpips_vgg": "#4daf4a",
    "vgg19": "#984ea3", "dinov2_vitb14": "#ff7f00",
}


def layer_label(backbone, layer):
    names = BACKBONE_CONFIGS.get(backbone, {}).get("layer_names", {})
    return f"L{layer} {names.get(layer, '?')}"


def main():
    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv.name}: {len(df)} rows")

    gammas = sorted(df["gamma_mult"].unique())[3:]  # skip 3 lowest
    cmap = cm.get_cmap("plasma", len(gammas))
    gamma_colors = {g: cmap(i) for i, g in enumerate(gammas)}

    n_cols = 2
    n_bb = len(BACKBONES)  # 5
    n_rows = 3  # 3×2 grid: 5 backbones + 1 legend cell
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6.5 * n_rows),
                                  sharey=True, constrained_layout=False)
    fig.subplots_adjust(hspace=0.45)
    axes = [axes_grid[r][c] for r in range(n_rows) for c in range(n_cols)]

    # Hide the legend cell for now
    axes[5].set_visible(False)

    fig.suptitle("GMMD: Spearman ρ vs Layer for each Backbone", fontsize=30, y=1.01, fontweight="bold")

    for ax, bb in zip(axes[:n_bb], BACKBONES):
        sub = df[df["backbone"] == bb]
        layers = sorted(sub["layer"].unique())
        names = [layer_label(bb, l) for l in layers]

        for g in gammas:
            sg = sub[sub["gamma_mult"] == g].sort_values("layer")
            rhos = [sg[sg["layer"] == l]["spearman"].values[0]
                    if len(sg[sg["layer"] == l]) else np.nan for l in layers]
            ax.plot(range(len(layers)), rhos, color=gamma_colors[g],
                    linewidth=1.2, alpha=0.8, label=f"γ×{g}")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(names, rotation=75, ha="right", fontsize=6.5)
        ax.set_title(bb, fontsize=24)
        ax.set_xlabel("Layer", fontsize=15)
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylim(0.2, 1.02)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Spearman ρ")

    # Legend in bottom-right cell (position 5)
    handles = [plt.Line2D([0], [0], color=gamma_colors[g], linewidth=1.5, label=f"γ×{g}")
               for g in gammas]
    ax_leg = axes[5]
    ax_leg.set_visible(True)
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, fontsize=20, title="γ median mult.",
                  title_fontsize=22, loc="center", ncol=2, frameon=True)

    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "01_rho_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
