#!/usr/bin/env python3
"""Generate 09_top20_spearman_kendall.png — Spearman and Kendall top-20 stacked."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from gram_mmd.config import BACKBONE_CONFIGS

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


def layer_label(backbone, layer):
    names = BACKBONE_CONFIGS.get(backbone, {}).get("layer_names", {})
    return f"L{layer} {names.get(layer, '?')}"


def plot_top20(ax, df, metric, title, top_n=20):
    best = df.loc[df.groupby(["backbone", "layer"])[metric].idxmax()].copy()
    best = best.sort_values(metric, ascending=False).head(top_n)
    best["label"] = best.apply(
        lambda r: f"{r['backbone']}\n{layer_label(r['backbone'], r['layer'])}", axis=1)
    colors = [BB_COLORS.get(bb, "gray") for bb in best["backbone"]]

    bars = ax.bar(range(len(best)), best[metric].values, color=colors)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best["label"].values, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(metric, fontsize=13)
    ax.set_title(title, fontsize=24, fontweight="bold")
    ymin = min(0.5, float(best[metric].min()) - 0.05)
    ax.set_ylim(ymin, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, best[metric].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    BB_LABELS = {bb: ("dinov2" if bb == "dinov2_vitb14" else bb) for bb in BACKBONES}
    handles = [Patch(color=BB_COLORS[bb], label=BB_LABELS[bb])
               for bb in BACKBONES if bb in best["backbone"].values]
    ax.legend(handles=handles, fontsize=14, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)


def main():
    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv.name}: {len(df)} rows")

    top_n = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, top_n * 0.7), 10))

    plot_top20(ax1, df, "spearman", "Top-20 Spearman ρ (best gamma per layer)", top_n)
    plot_top20(ax2, df, "kendall",  "Top-20 Kendall τ (best gamma per layer)",  top_n)

    fig.tight_layout(pad=3.0)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "09_top20_spearman_kendall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
