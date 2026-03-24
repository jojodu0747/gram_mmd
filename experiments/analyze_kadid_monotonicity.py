"""
analyze_kadid_monotonicity.py
=============================
Graphiques pour evaluate_kadid_monotonicity.py.

Figures :
  1. Bar chart comparatif : Spearman / Kendall / Mono. stricte / Mono. partielle
  2. Courbes score moyen par niveau (1→5) par métrique
  3. Bar chart taux de monotonie stricte par backbone

Usage :
    python experiments/analyze_kadid_monotonicity.py
    python experiments/analyze_kadid_monotonicity.py --results-dir results/kadid_monotonicity
"""

import argparse
import csv
import glob as glob_module
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG graphiques
# ============================================================

METRIC_COLORS = {
    "sd_vae":   "#2196F3",
    "dc_ae":    "#9C27B0",
    "dinov2":   "#FF9800",
    "CMMD":     "#4CAF50",
    "vgg":      "#F44336",
    "resnet":   "#795548",
    "lpips":    "#607D8B",
    "clip":     "#009688",
}

def get_color(metric_key: str) -> str:
    for key, color in METRIC_COLORS.items():
        if key.lower() in metric_key.lower():
            return color
    return "#888888"


def short_label(label: str) -> str:
    return (label
            .replace("gamma=", "γ=")
            .replace("CLIP ViT-L/14", "CLIP-L14"))


# ============================================================
# Utilitaires
# ============================================================

def latest_csv(results_dir: str, pattern: str) -> str | None:
    files = sorted(glob_module.glob(os.path.join(results_dir, pattern)))
    if not files:
        return None
    print(f"  Chargement : {os.path.basename(files[-1])}")
    return files[-1]


def load_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================
# Figure 1 — Bar chart comparatif multi-métrique
# ============================================================

def plot_summary_bars(sum_rows: list, out_dir: str):
    """4 sous-graphiques : Spearman, Kendall, Mono stricte, Mono partielle."""
    metrics = [r["metric"] for r in sum_rows]
    labels  = [short_label(r["label"]) for r in sum_rows]
    colors  = [get_color(m) for m in metrics]

    spearman = [float(r["spearman_vs_degradation"]) for r in sum_rows]
    kendall  = [float(r["kendall_vs_degradation"]) for r in sum_rows]
    mono_s   = [float(r["mono_strict_rate"]) for r in sum_rows]
    mono_p   = [float(r["mono_partial_rate"]) for r in sum_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("KADID-10k — Comparaison des métriques (81 refs, 10125 dégradées, N=1)",
                 fontsize=13, fontweight="bold")

    datasets = [
        (axes[0, 0], spearman, "Spearman ρ(score, 5−dmos)", (-0.2, 1.0)),
        (axes[0, 1], kendall,  "Kendall τ(score, 5−dmos)",  (-0.2, 0.7)),
        (axes[1, 0], mono_s,   "Monotonie stricte (5/5 ↑)", (0, 1.0)),
        (axes[1, 1], mono_p,   "Monotonie partielle (≥3/4 ↑)", (0, 1.0)),
    ]

    x = np.arange(len(metrics))

    for ax, vals, title, ylim in datasets:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(ylim)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        for bar, v in zip(bars, vals):
            mid = bar.get_y() + bar.get_height() / 2
            txt_c = "white" if abs(v) > 0.4 else "black"
            fmt = f"{v:.3f}" if abs(v) < 1 else f"{v:.1%}"
            ax.text(bar.get_x() + bar.get_width() / 2, mid, fmt,
                    ha="center", va="center", fontsize=9, color=txt_c, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_summary_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 : {path}")


# ============================================================
# Figure 2 — Courbes score moyen par niveau
# ============================================================

def plot_level_curves(sum_rows: list, out_dir: str):
    """Score moyen par niveau de dégradation (1→5) pour chaque métrique."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("KADID-10k — Score MMD² moyen par niveau de dégradation\n"
                 "(1 = léger, 5 = sévère)  |  Score ↑ avec dégradation = bon",
                 fontsize=12, fontweight="bold")

    levels = [1, 2, 3, 4, 5]

    for r in sum_rows:
        metric = r["metric"]
        label  = short_label(r["label"])
        color  = get_color(metric)

        scores = []
        for lv in levels:
            val = r.get(f"score_lv{lv}", "")
            scores.append(float(val) if val != "" else np.nan)

        # Normaliser pour visualiser sur même échelle (min-max par métrique)
        s = np.array(scores)
        if not np.all(np.isnan(s)):
            s_min, s_max = np.nanmin(s), np.nanmax(s)
            if s_max > s_min:
                s_norm = (s - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros_like(s)
            ax.plot(levels, s_norm, marker="o", color=color, linewidth=2,
                    markersize=8, label=label, zorder=3)
            for lv, raw, norm in zip(levels, scores, s_norm):
                ax.annotate(f"{raw:.5f}", (lv, norm), fontsize=6.5,
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", color=color, alpha=0.8)

    ax.set_xlabel("Niveau de dégradation", fontsize=11)
    ax.set_ylabel("Score normalisé (min-max)", fontsize=11)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{lv}\n(léger)" if lv == 1 else f"{lv}\n(sévère)" if lv == 5
                        else str(lv) for lv in levels])
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_level_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 : {path}")


# ============================================================
# Figure 3 — Bar chart Spearman trié
# ============================================================

def plot_spearman_ranking(sum_rows: list, out_dir: str):
    """Bar chart Spearman trié (meilleur en premier)."""
    sorted_rows = sorted(sum_rows, key=lambda r: -float(r["spearman_vs_degradation"]))

    labels   = [short_label(r["label"]) for r in sorted_rows]
    vals     = [float(r["spearman_vs_degradation"]) for r in sorted_rows]
    colors   = [get_color(r["metric"]) for r in sorted_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.set_title("KADID-10k — Classement par Spearman ρ(score, 5−dmos)\n"
                 "81 refs × 10125 dégradées  |  scoring N=1",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    for bar, v in zip(bars, vals):
        mid = bar.get_y() + bar.get_height() / 2
        txt_c = "white" if abs(v) > 0.35 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2, mid, f"{v:+.3f}",
                ha="center", va="center", fontsize=10, color=txt_c, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_ylim(-0.3, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_spearman_ranking.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 : {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse graphique KADID-10k monotonie")
    parser.add_argument("--results-dir", default="results/kadid_monotonicity",
                        help="Dossier contenant les CSV")
    parser.add_argument("--out-dir", default=None,
                        help="Dossier de sortie (défaut: {results-dir}/analysis)")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print("Chargement CSV summary...")
    sum_path = latest_csv(results_dir, "summary_*.csv")
    if not sum_path:
        raise FileNotFoundError(f"Aucun summary_*.csv dans {results_dir}")

    sum_rows = load_csv(sum_path)
    print(f"  {len(sum_rows)} métriques trouvées")
    for r in sum_rows:
        print(f"    {r['metric']:25s}  Spearman={float(r['spearman_vs_degradation']):+.4f}  "
              f"Mono.strict={float(r['mono_strict_rate']):.1%}")

    plot_summary_bars(sum_rows, out_dir)
    plot_level_curves(sum_rows, out_dir)
    plot_spearman_ranking(sum_rows, out_dir)

    print(f"\nTous les graphiques dans : {out_dir}")


if __name__ == "__main__":
    main()
