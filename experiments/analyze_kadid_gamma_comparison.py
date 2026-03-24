"""
analyze_kadid_gamma_comparison.py
==================================
Figures pour evaluate_kadid_gamma_comparison.py.
10 points : 9 configs (3 backbones × 3 gammas) + CMMD.

Figures :
  01_scatter_spearman_vs_mono.png — scatter Spearman vs Mono.stricte, 1 point/config
  02_barplot_spearman.png         — barplot Spearman trié
  03_barplot_mono_strict.png      — barplot monotonie stricte trié

Usage :
    .venv/bin/python experiments/analyze_kadid_gamma_comparison.py
"""

import csv
import glob as glob_module
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "kadid_gamma_comparison"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Chargement ────────────────────────────────────────────────────────

def latest_csv(pattern):
    files = sorted(glob_module.glob(str(RESULTS_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"Aucun fichier : {RESULTS_DIR / pattern}")
    print(f"  {os.path.basename(files[-1])}")
    return Path(files[-1])


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


print("Chargement...")
rows = load_csv(latest_csv("summary_*.csv"))

for r in rows:
    r["spearman"]     = float(r["spearman"])
    r["kendall"]      = float(r["kendall"])
    r["mono_strict"]  = float(r["mono_strict"])
    r["mono_partial"] = float(r["mono_partial"])

print(f"  {len(rows)} configs chargées")


# ── Couleurs et styles ────────────────────────────────────────────────

BACKBONE_COLORS = {
    "sd_vae":         "#2196F3",   # bleu
    "dc_ae":          "#9C27B0",   # violet
    "dinov2_vitb14":  "#FF9800",   # orange
    "clip_vit_large": "#4CAF50",   # vert (CMMD)
}

GAMMA_MARKERS = {
    "median":     "o",
    "10x_median": "s",
    "50x_median": "^",
    "fixed":      "D",   # CMMD
}

GAMMA_SHORT = {
    "median":     "γ_med",
    "10x_median": "10×γ",
    "50x_median": "50×γ",
    "fixed":      "γ=0.005",
}

BACKBONE_SHORT = {
    "sd_vae":         "SD-VAE L11",
    "dc_ae":          "DC-AE L8",
    "dinov2_vitb14":  "DINOv2 L3",
    "clip_vit_large": "CMMD",
}


def get_color(r):
    return BACKBONE_COLORS.get(r["backbone"], "#888888")


def get_marker(r):
    return GAMMA_MARKERS.get(r["gamma_tag"], "o")


def short_label(r):
    b = BACKBONE_SHORT.get(r["backbone"], r["backbone"])
    g = GAMMA_SHORT.get(r["gamma_tag"], r["gamma_tag"])
    if r["backbone"] == "clip_vit_large":
        return b
    return f"{b}  {g}"


# ═══════════════════════════════════════════════════════════════════════
#  Figure 1 — Scatter Spearman vs Monotonie stricte
# ═══════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(9, 7))
ax1.set_title(
    "KADID-10k — Spearman ρ vs Monotonie stricte\n"
    "9 configs (3 backbones × 3 gammas heuristiques) + CMMD",
    fontsize=12, fontweight="bold"
)

for r in rows:
    ax1.scatter(
        r["spearman"], r["mono_strict"],
        color=get_color(r), marker=get_marker(r),
        s=120, zorder=5,
        edgecolors="black", linewidths=0.7,
    )
    ax1.annotate(
        short_label(r),
        (r["spearman"], r["mono_strict"]),
        fontsize=7.5,
        xytext=(6, 4), textcoords="offset points",
        color=get_color(r),
    )

ax1.set_xlabel("Spearman ρ(score, 5−dmos)  [↑ meilleur]", fontsize=11)
ax1.set_ylabel("Taux de monotonie stricte  [↑ meilleur]", fontsize=11)
ax1.yaxis.grid(True, alpha=0.3)
ax1.xaxis.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# Légende backbone
backbone_legend = [
    mpatches.Patch(color=BACKBONE_COLORS["sd_vae"],         label="SD-VAE L11"),
    mpatches.Patch(color=BACKBONE_COLORS["dc_ae"],          label="DC-AE L8"),
    mpatches.Patch(color=BACKBONE_COLORS["dinov2_vitb14"],  label="DINOv2 L3"),
    mpatches.Patch(color=BACKBONE_COLORS["clip_vit_large"], label="CMMD"),
]
# Légende gamma
from matplotlib.lines import Line2D
gamma_legend = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None",
           markersize=8, markeredgecolor="black", label="γ_median"),
    Line2D([0], [0], marker="s", color="gray", linestyle="None",
           markersize=8, markeredgecolor="black", label="10 × γ_median"),
    Line2D([0], [0], marker="^", color="gray", linestyle="None",
           markersize=8, markeredgecolor="black", label="50 × γ_median"),
    Line2D([0], [0], marker="D", color=BACKBONE_COLORS["clip_vit_large"],
           linestyle="None", markersize=8, markeredgecolor="black", label="CMMD (γ fixe)"),
]
leg1 = ax1.legend(handles=backbone_legend, fontsize=9, loc="lower left",
                  title="Backbone", title_fontsize=9)
ax1.add_artist(leg1)
ax1.legend(handles=gamma_legend, fontsize=9, loc="upper left",
           title="Gamma", title_fontsize=9)

plt.tight_layout()
p1 = PLOTS_DIR / "01_scatter_spearman_vs_mono.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure 1 : {p1}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 2 — Barplot Spearman trié
# ═══════════════════════════════════════════════════════════════════════

sorted_rows = sorted(rows, key=lambda r: -r["spearman"])

labels  = [short_label(r) for r in sorted_rows]
vals    = [r["spearman"] for r in sorted_rows]
colors  = [get_color(r) for r in sorted_rows]

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.set_title(
    "KADID-10k — Spearman ρ(score, 5−dmos)  |  81 refs × 10125 images  |  N=1\n"
    "3 backbones × 3 gammas heuristiques + CMMD",
    fontsize=11, fontweight="bold"
)

x = np.arange(len(labels))
bars = ax2.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, zorder=3, width=0.7)

ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

for bar, v in zip(bars, vals):
    mid = bar.get_y() + bar.get_height() / 2
    txt_c = "white" if abs(v) > 0.35 else "black"
    ax2.text(bar.get_x() + bar.get_width() / 2, mid, f"{v:+.3f}",
             ha="center", va="center", fontsize=9, color=txt_c, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("Spearman ρ  [↑ meilleur]", fontsize=11)
ax2.set_ylim(-0.1, 1.0)
ax2.yaxis.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

backbone_legend2 = [
    mpatches.Patch(color=BACKBONE_COLORS["sd_vae"],         label="SD-VAE L11"),
    mpatches.Patch(color=BACKBONE_COLORS["dc_ae"],          label="DC-AE L8"),
    mpatches.Patch(color=BACKBONE_COLORS["dinov2_vitb14"],  label="DINOv2 L3"),
    mpatches.Patch(color=BACKBONE_COLORS["clip_vit_large"], label="CMMD"),
]
ax2.legend(handles=backbone_legend2, fontsize=9, loc="upper right")

plt.tight_layout()
p2 = PLOTS_DIR / "02_barplot_spearman.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure 2 : {p2}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 3 — Barplot monotonie stricte trié
# ═══════════════════════════════════════════════════════════════════════

sorted_rows3 = sorted(rows, key=lambda r: -r["mono_strict"])

labels3 = [short_label(r) for r in sorted_rows3]
vals3   = [r["mono_strict"] for r in sorted_rows3]
colors3 = [get_color(r) for r in sorted_rows3]

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.set_title(
    "KADID-10k — Taux de monotonie stricte  |  81 refs × 10125 images\n"
    "3 backbones × 3 gammas heuristiques + CMMD",
    fontsize=11, fontweight="bold"
)

x3 = np.arange(len(labels3))
bars3 = ax3.bar(x3, vals3, color=colors3, edgecolor="white", linewidth=0.5, zorder=3, width=0.7)

for bar, v in zip(bars3, vals3):
    mid = bar.get_y() + bar.get_height() / 2
    txt_c = "white" if v > 0.4 else "black"
    ax3.text(bar.get_x() + bar.get_width() / 2, mid, f"{v:.1%}",
             ha="center", va="center", fontsize=9, color=txt_c, fontweight="bold")

ax3.set_xticks(x3)
ax3.set_xticklabels(labels3, rotation=30, ha="right", fontsize=9)
ax3.set_ylabel("Taux de monotonie stricte  [↑ meilleur]", fontsize=11)
ax3.set_ylim(0, 1.05)
ax3.yaxis.grid(True, alpha=0.3)
ax3.set_axisbelow(True)
ax3.legend(handles=backbone_legend2, fontsize=9, loc="upper right")

plt.tight_layout()
p3 = PLOTS_DIR / "03_barplot_mono_strict.png"
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure 3 : {p3}")

print(f"\nPlots dans : {PLOTS_DIR}")
