#!/usr/bin/env python3
"""
For each backbone: one plot showing |Spearman ρ| vs layer index,
with one line per gamma multiplier.

Input : results/synthetic_kadid_sweep/summary_all.csv
Output: results/synthetic_kadid_sweep/plots/per_backbone/spearman_{backbone}.png
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

CSV_PATH = Path("/home/jnapolitano/Documents/metrique/results/synthetic_kadid_sweep/summary_all.csv")
OUT_DIR  = Path("/home/jnapolitano/Documents/metrique/results/synthetic_kadid_sweep/plots/per_backbone")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_LABELS = {
    "sd_vae":         "SD-VAE",
    "dc_ae":          "DC-AE",
    "lpips_vgg":      "LPIPS-VGG",
    "vgg19":          "VGG-19",
    "dinov2_vitb14":  "DINOv2-B",
}

GAMMA_MULTS = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

def load():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "backbone":    r["backbone"],
                "layer":       int(r["layer"]),
                "gamma_mult":  float(r["gamma_mult"]),
                "spearman":    float(r["spearman"]),
                "kendall":     float(r["kendall"]),
                "mono_strict": float(r["mono_strict"]),
            })
    return rows

def plot_backbone(rows, backbone):
    bb_rows = [r for r in rows if r["backbone"] == backbone]
    if not bb_rows:
        print(f"  No data for {backbone}, skipping.")
        return

    layers = sorted({r["layer"] for r in bb_rows})
    gammas = sorted({r["gamma_mult"] for r in bb_rows})

    # Color map: one color per gamma (cool→warm = small→large)
    cmap = cm.get_cmap("plasma", len(gammas))
    gamma_color = {g: cmap(i) for i, g in enumerate(gammas)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    metrics = [("spearman", "|Spearman ρ|"), ("kendall", "|Kendall τ|"), ("mono_strict", "Strict mono")]

    for ax, (met, mlabel) in zip(axes, metrics):
        for gm in gammas:
            vals = []
            for l in layers:
                hits = [r for r in bb_rows if r["layer"] == l and r["gamma_mult"] == gm]
                if hits:
                    vals.append((l, abs(hits[0][met])))
            if vals:
                xs, ys = zip(*vals)
                lw = 2.0 if gm == 1.0 else 1.0
                alpha = 1.0 if gm == 1.0 else 0.75
                ax.plot(xs, ys, marker="o", ms=3, lw=lw, alpha=alpha,
                        color=gamma_color[gm], label=f"×{gm:g}")
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.25)

    # Single legend outside on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="γ multiplier", fontsize=7,
               loc="center right", bbox_to_anchor=(1.0, 0.5), ncol=1)
    fig.suptitle(
        f"{BACKBONE_LABELS.get(backbone, backbone)} — |Spearman ρ|, |Kendall τ|, Strict mono vs layer\n"
        "20 COCO refs · 1000 anchor · 125 synthetic KADID distortions",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    out = OUT_DIR / f"spearman_{backbone}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

rows = load()
print(f"Loaded {len(rows)} rows")
backbones = list({r["backbone"] for r in rows})
for bb in backbones:
    plot_backbone(rows, bb)
print("All done.")
