#!/usr/bin/env python3
"""
Plots pour le sweep cumulatif (200 groupes × 50 imgs, 8 niveaux × 25 types).

Génère (style sweep_coco_degraded) :
  01_rho_profiles.png       — Spearman ρ vs layer, une ligne par gamma, un subplot par backbone
  02_top20_spearman.png     — Barplot Top-20 |Spearman ρ|  (label = "Lk layer_name")
  03_top20_kendall.png      — Barplot Top-20 |Kendall τ|
  04_top20_mono.png         — Barplot Top-20 strict monotonicity
  05_scatter_rho_mono.png   — Scatter ρ vs mono, coloré par backbone
  06_heatmap_{bb}.png       — Heatmap layer×gamma pour chaque backbone
"""

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

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Plot sweep results")
_parser.add_argument("--csv", type=Path,
    default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "summary_all.csv")
_parser.add_argument("--outdir", type=Path,
    default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "plots")
_args, _ = _parser.parse_known_args()

CSV_PATH = _args.csv
OUT_DIR  = _args.outdir
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = ["sd_vae", "dc_ae", "lpips_vgg", "vgg19", "resnet50", "dinov2_vitb14"]
BB_COLORS = {
    "sd_vae":        "#e41a1c",
    "dc_ae":         "#377eb8",
    "lpips_vgg":     "#4daf4a",
    "vgg19":         "#984ea3",
    "resnet50":      "#a65628",
    "dinov2_vitb14": "#ff7f00",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def layer_label(backbone: str, layer: int) -> str:
    names = BACKBONE_CONFIGS.get(backbone, {}).get("layer_names", {})
    name = names.get(layer, "?")
    return f"L{layer} {name}"


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["layer_label"] = df.apply(lambda r: layer_label(r["backbone"], r["layer"]), axis=1)
    df["abs_spearman"] = df["spearman"].abs()
    if "kendall" in df.columns:
        df["abs_kendall"] = df["kendall"].abs()
    return df


# ── Plot 01 : Spearman ρ vs layer ────────────────────────────────────────────

def plot_rho_profiles(df: pd.DataFrame):
    gammas = sorted(df["gamma_mult"].unique())[3:]  # skip 3 lowest
    cmap   = cm.get_cmap("plasma", len(gammas))
    gamma_colors = {g: cmap(i) for i, g in enumerate(gammas)}

    n_cols = 3
    n_bb   = len(BACKBONES)
    # ensure at least one spare cell for legend
    total_cells = n_cols * ((n_bb + n_cols - 1) // n_cols)
    if total_cells == n_bb:
        total_cells += n_cols
    n_rows = total_cells // n_cols
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
    axes = [axes_grid[r][c] for r in range(n_rows) for c in range(n_cols)]
    # hide unused axes
    for ax in axes[n_bb:]:
        ax.set_visible(False)
    fig.suptitle("Spearman ρ vs Layer — cumulative sweep (200 groups)", fontsize=20, y=1.01)

    for ax, bb in zip(axes, BACKBONES):
        sub = df[df["backbone"] == bb]
        layers = sorted(sub["layer"].unique())
        names  = [layer_label(bb, l) for l in layers]

        for g in gammas:
            sg = sub[sub["gamma_mult"] == g].sort_values("layer")
            rhos = [sg[sg["layer"] == l]["spearman"].values[0]
                    if len(sg[sg["layer"] == l]) else np.nan for l in layers]
            ax.plot(range(len(layers)), rhos, color=gamma_colors[g],
                    linewidth=1.2, alpha=0.8, label=f"γ×{g}")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(names, rotation=75, ha="right", fontsize=6.5)
        ax.set_title(bb, fontsize=15)
        ax.set_xlabel("Layer", fontsize=12)
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylim(0.2, 1.02)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Spearman ρ")

    # légende γ dans la cellule vide (après les backbones)
    handles = [plt.Line2D([0], [0], color=gamma_colors[g], linewidth=1.5, label=f"γ×{g}")
               for g in gammas]
    ax_leg = axes[n_bb]
    ax_leg.set_visible(True)
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, fontsize=8, title="γ median mult.",
                  loc="center", ncol=2, frameon=True)

    fig.tight_layout()
    path = OUT_DIR / "01_rho_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 02/03/04 : Top-N barplots ───────────────────────────────────────────

def plot_top_barplot(df: pd.DataFrame, metric: str, title: str, fname: str, top_n: int = 20):
    best = df.loc[df.groupby(["backbone", "layer"])[metric].idxmax()].copy()
    best = best.sort_values(metric, ascending=False).head(top_n)
    best["label"] = best.apply(lambda r: f"{r['backbone']}\n{layer_label(r['backbone'], r['layer'])}", axis=1)

    colors = [BB_COLORS.get(bb, "gray") for bb in best["backbone"]]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.7), 5))
    bars = ax.bar(range(len(best)), best[metric].values, color=colors)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best["label"].values, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ymin = min(0.5, float(best[metric].min()) - 0.05)
    ax.set_ylim(ymin, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # annotate values
    for bar, val in zip(bars, best[metric].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=BB_COLORS[bb], label=bb) for bb in BACKBONES if bb in best["backbone"].values]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 05 : Scatter ρ vs mono ───────────────────────────────────────────────

def plot_scatter_rho_mono(df: pd.DataFrame):
    best = df.loc[df.groupby(["backbone", "layer"])["spearman"].idxmax()]
    metric = "kendall" if "kendall" in df.columns else "mono_strict"
    xlabel = "Kendall τ" if metric == "kendall" else "Strict monotonicity"

    fig, ax = plt.subplots(figsize=(7, 5))
    for bb in BACKBONES:
        sub = best[best["backbone"] == bb]
        ax.scatter(sub[metric], sub["spearman"],
                   color=BB_COLORS.get(bb, "gray"), label=bb, alpha=0.75, s=50)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Spearman ρ vs {xlabel} — best gamma per (backbone, layer)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "05_scatter_rho_kendall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 06 : Heatmap layer × gamma ──────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, backbone: str):
    sub = df[df["backbone"] == backbone]
    layers = sorted(sub["layer"].unique())
    gammas = sorted(sub["gamma_mult"].unique())

    mat = np.zeros((len(layers), len(gammas)))
    for i, l in enumerate(layers):
        for j, g in enumerate(gammas):
            sel = sub[(sub["layer"] == l) & (sub["gamma_mult"] == g)]["spearman"]
            mat[i, j] = sel.values[0] if len(sel) else np.nan

    fig, ax = plt.subplots(figsize=(max(6, len(gammas) * 0.8), max(4, len(layers) * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f"{g}" for g in gammas], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([layer_label(backbone, l) for l in layers], fontsize=7)
    ax.set_xlabel("gamma multiplier")
    ax.set_title(f"Spearman ρ — {backbone}")
    fig.colorbar(im, ax=ax, label="Spearman ρ")

    for i in range(len(layers)):
        for j in range(len(gammas)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if 0.2 < v < 0.8 else "white")

    fig.tight_layout()
    path = OUT_DIR / f"06_heatmap_{backbone}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 07 : Boxplot ρ par backbone ─────────────────────────────────────────

def plot_boxplot_by_backbone(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Skip 3 lowest gamma multipliers
    top_gammas = sorted(df["gamma_mult"].unique())[3:]
    df_filt = df[df["gamma_mult"].isin(top_gammas)]

    sorted_bbs = sorted(BACKBONES,
                        key=lambda bb: np.median(df_filt[df_filt["backbone"] == bb]["spearman"].values),
                        reverse=True)
    data   = [df_filt[df_filt["backbone"] == bb]["spearman"].values for bb in sorted_bbs]
    colors = [BB_COLORS[bb] for bb in sorted_bbs]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(sorted_bbs) + 1))
    ax.set_xticklabels(sorted_bbs, rotation=20, ha="right", fontsize=13, fontweight="bold")
    ax.set_ylabel("Spearman ρ", fontsize=16, fontweight="bold")
    ax.set_ylim(0.0, 1.02)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Spearman ρ distribution per backbone\n(all layers × 7 best gammas)", fontsize=20, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "07_boxplot_by_backbone.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {CSV_PATH}")
    df = load_df()
    print(f"  {len(df)} rows — backbones: {df['backbone'].unique().tolist()}")

    print("01 Rho profiles…")
    plot_rho_profiles(df)

    print("02 Top-20 Spearman…")
    plot_top_barplot(df, "spearman", "Top-20 Spearman ρ (best gamma per layer)", "02_top20_spearman.png")

    print("03 Top-20 rho_min…")
    plot_top_barplot(df, "rho_min", "Top-20 ρ_min — worst type per (backbone, layer)", "03_top20_rhomin.png")

    if "kendall" in df.columns:
        print("04 Top-20 Kendall τ…")
        plot_top_barplot(df, "kendall", "Top-20 Kendall τ (best gamma per layer)", "04_top20_kendall.png")
    else:
        print("04 Top-20 Strict Mono…")
        plot_top_barplot(df, "mono_strict", "Top-20 Strict Monotonicity (best gamma per layer)", "04_top20_mono.png")

    print("05 Scatter ρ vs mono…")
    plot_scatter_rho_mono(df)

    print("06 Heatmaps…")
    for bb in BACKBONES:
        plot_heatmap(df, bb)

    print("07 Boxplot by backbone…")
    plot_boxplot_by_backbone(df)

    print(f"\nDone. Plots dans {OUT_DIR}")


if __name__ == "__main__":
    main()
