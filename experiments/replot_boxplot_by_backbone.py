#!/usr/bin/env python3
"""Replot 07_boxplot_by_backbone.png from existing CSV — no sweep needed."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=Path,
    default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "summary_full_sweep.csv")
parser.add_argument("--outdir", type=Path,
    default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "plots_new")
args = parser.parse_args()

BACKBONES = ["sd_vae", "dc_ae", "lpips_vgg", "vgg19", "dinov2_vitb14"]
BB_COLORS = {
    "sd_vae": "#e41a1c", "dc_ae": "#377eb8", "lpips_vgg": "#4daf4a",
    "vgg19": "#984ea3", "dinov2_vitb14": "#ff7f00",
}
BB_LABELS = {bb: ("dinov2" if bb == "dinov2_vitb14" else bb) for bb in BACKBONES}


def main():
    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv.name}: {len(df)} rows")

    top_gammas = sorted(df["gamma_mult"].unique())[3:]
    df_filt = df[df["gamma_mult"].isin(top_gammas)]

    sorted_bbs = sorted(BACKBONES,
                        key=lambda bb: np.median(df_filt[df_filt["backbone"] == bb]["spearman"].values),
                        reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    data   = [df_filt[df_filt["backbone"] == bb]["spearman"].values for bb in sorted_bbs]
    colors = [BB_COLORS[bb] for bb in sorted_bbs]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(sorted_bbs) + 1))
    ax.set_xticklabels([BB_LABELS[bb] for bb in sorted_bbs],
                       rotation=20, ha="right", fontsize=13)
    ax.set_ylabel("Spearman ρ", fontsize=16)
    ax.set_ylim(0.0, 1.02)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Spearman ρ distribution per backbone", fontsize=17, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "07_boxplot_by_backbone.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
