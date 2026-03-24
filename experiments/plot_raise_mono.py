"""
plot_raise_mono.py
==================
Visualisation des résultats de raise_mono_dual.py.

Figures (dans results/raise_mono_dual/plots/) :
  fig1_neg_rho_barplot.png    — barplot −ρ(Spearman)  (48 groupes × 10 images)
  fig2_scatter_mmd_vs_mos.png — scatter MMD vs MOS avec régression linéaire
                                (12 groupes × 40 images, 1 panneau par famille)
  fig3_heatmap.png            — heatmap −ρ : métrique × granularité

Usage :
    .venv/bin/python experiments/plot_raise_mono.py
    .venv/bin/python experiments/plot_raise_mono.py --res-dir results/raise_mono_dual
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist

BASE_DIR = Path(__file__).resolve().parent.parent
RES_DIR  = BASE_DIR / "results" / "raise_mono_dual"

# ─── couleurs par backbone ────────────────────────────────────────────────────

MODEL_COLORS = {
    "sd_vae":        "#2196F3",   # bleu
    "dc_ae":         "#E91E63",   # rose
    "dinov2_vitb14": "#FF9800",   # orange
    "CMMD":          "#4CAF50",   # vert
}

def _model_of(label: str) -> str:
    for k in ("sd_vae", "dc_ae", "dinov2_vitb14", "CMMD"):
        if k in label:
            return k
    return "other"

def _color(label: str) -> str:
    return MODEL_COLORS.get(_model_of(label), "#888888")

def _short(label: str) -> str:
    return (label
            .replace("MMD_sd_vae_", "SD-VAE ")
            .replace("MMD_dinov2_vitb14_", "DINOv2 ")
            .replace("MMD_dc_ae_", "DC-AE ")
            .replace("CMMD_CLIP-ViT-L14", "CMMD")
            .replace("_L", " L")
            .replace("_g", " γ="))


# ─── chargement ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("spearman_rho", "p_spearman", "pearson_r", "p_pearson"):
                if key in row:
                    row[key] = float(row[key])
            if "n_groups" in row:
                row["n_groups"] = int(row["n_groups"])
            rows.append(row)
    return rows


def load_detail_csv(path: Path) -> dict:
    """Retourne {label: [(mos_mean, mmd_score, group_idx), ...]}."""
    series: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lbl   = row["label"]
            mos   = float(row["mos_mean"])
            score = float(row["mmd_score"])
            idx   = int(row["group_idx"])
            series.setdefault(lbl, []).append((mos, score, idx))
    for lbl in series:
        series[lbl].sort(key=lambda x: x[2])
    return series


def find_latest(pattern: str, res_dir: Path) -> Path:
    files = sorted(res_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier {pattern} dans {res_dir}")
    return files[-1]


# ─── Figure 1 : barplot −ρ ───────────────────────────────────────────────────

def plot_neg_rho_barplot(rows: list, out_path: Path):
    n_groups = rows[0]["n_groups"] if rows else 48
    t_crit   = t_dist.ppf(0.975, df=n_groups - 2)
    rho_crit = t_crit / math.sqrt(t_crit ** 2 + n_groups - 2)

    sorted_rows = sorted(rows, key=lambda r: -r["spearman_rho"])  # -rho desc = best first
    labels   = [_short(r["label"]) for r in sorted_rows]
    neg_rhos = [-r["spearman_rho"] for r in sorted_rows]
    ps       = [r["p_spearman"] for r in sorted_rows]
    colors   = [_color(r["label"]) for r in sorted_rows]
    n        = len(sorted_rows)

    fig, ax = plt.subplots(figsize=(max(12, n * 1.1 + 2), 6))
    bars = ax.bar(range(n), neg_rhos, color=colors,
                  edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline( rho_crit, color="red", linewidth=1.8, linestyle="--", alpha=0.85,
                label=f"Seuil sig. p<0.05 (|ρ|={rho_crit:.3f})")
    ax.axhline(-rho_crit, color="red", linewidth=1.8, linestyle="--", alpha=0.3)

    for bar, p, nr in zip(bars, ps, neg_rhos):
        if p < 0.05:
            yt = bar.get_height() + 0.03 if nr >= 0 else bar.get_y() - 0.07
            ax.text(bar.get_x() + bar.get_width() / 2, yt, "✱",
                    ha="center", fontsize=11, color="black")
        mid = bar.get_y() + bar.get_height() / 2
        txt_color = "white" if abs(nr) > 0.45 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2, mid,
                f"{nr:+.2f}", ha="center", va="center",
                fontsize=8.5, color=txt_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("−ρ (Spearman)  [>0 : MMD décroît avec MOS → cohérent]", fontsize=11)
    ax.set_ylim(-1.1, 1.15)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Monotonicité RAISE — −ρ(Spearman) entre MMD et MOS  |  ancrage 1000 COCO\n"
        f"{n_groups} groupes × {480 // n_groups} images IA RAISE  |  "
        f"seuil sig. p<0.05 : |ρ| > {rho_crit:.3f}",
        fontsize=12, fontweight="bold"
    )

    patches = [mpatches.Patch(color=v, label=k) for k, v in MODEL_COLORS.items()]
    patches.append(plt.Line2D([0], [0], color="red", linewidth=1.8, linestyle="--",
                               label=f"Seuil sig. |ρ|>{rho_crit:.3f}"))
    ax.legend(handles=patches, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 2 : scatter avec régression linéaire ─────────────────────────────

def plot_scatter_panels(detail: dict, summary: list, out_path: Path):
    stats_map = {r["label"]: r for r in summary}

    # Meilleure métrique par famille (rho le plus négatif)
    def best_for(family):
        cands = [r for r in summary if _model_of(r["label"]) == family]
        if not cands:
            return None
        return min(cands, key=lambda r: r["spearman_rho"])

    panels = []
    for family in ("sd_vae", "dinov2_vitb14", "dc_ae", "CMMD"):
        b = best_for(family)
        if b and b["label"] in detail:
            panels.append(b["label"])

    n_panels = len(panels)
    if n_panels == 0:
        print("  Aucun panneau à tracer.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        "MMD vs MOS — 12 groupes × 40 images IA RAISE  |  ancrage 1000 COCO\n"
        "(groupe 1 = qualité la plus faible, groupe 12 = qualité la plus élevée)",
        fontsize=12, fontweight="bold"
    )

    for ax, lbl in zip(axes, panels):
        pts  = detail[lbl]
        mos  = np.array([p[0] for p in pts])
        mmd  = np.array([p[1] for p in pts])
        idxs = [p[2] for p in pts]

        color   = _color(lbl)
        sr      = stats_map[lbl]
        neg_rho = -sr["spearman_rho"]
        p_val   = sr["p_spearman"]
        sig_str = " ✱" if p_val < 0.05 else ""

        ax.scatter(mos, mmd, color=color, s=70, zorder=4,
                   edgecolors="white", linewidth=0.6)

        # régression linéaire
        m_fit, b_fit = np.polyfit(mos, mmd, 1)
        x_line = np.linspace(mos.min(), mos.max(), 100)
        ax.plot(x_line, m_fit * x_line + b_fit, color=color,
                linewidth=1.5, alpha=0.7, linestyle="--")

        # numéros de groupes
        for i_pt, (x, y) in enumerate(zip(mos, mmd)):
            ax.annotate(str(idxs[i_pt]), (x, y), fontsize=7.5,
                        xytext=(4, 3), textcoords="offset points", color="#333")

        ax.set_title(
            f"{_short(lbl)}\n"
            f"−ρ = {neg_rho:+.3f}{sig_str}   p = {p_val:.3e}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("MOS moyen du groupe", fontsize=10)
        ax.set_ylabel("MMD²", fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure 3 : heatmap −ρ métrique × granularité ────────────────────────────

def plot_heatmap(rows_48: list, rows_12: list, out_path: Path):
    map_48   = {r["label"]: -r["spearman_rho"] for r in rows_48}
    map_48_p = {r["label"]: r["p_spearman"]    for r in rows_48}
    map_12   = {r["label"]: -r["spearman_rho"] for r in rows_12}
    map_12_p = {r["label"]: r["p_spearman"]    for r in rows_12}

    # Trier par -rho sur 12×40 (desc), puis 48×10
    labels = sorted(set(map_48) | set(map_12),
                    key=lambda l: map_12.get(l, map_48.get(l, 0)), reverse=True)

    n_rows     = len(labels)
    col_labels = ["48×10", "12×40"]
    mat        = np.full((n_rows, 2), np.nan)
    sig        = np.zeros((n_rows, 2), dtype=bool)

    for i, lbl in enumerate(labels):
        if lbl in map_48:
            mat[i, 0] = map_48[lbl]
            sig[i, 0] = map_48_p[lbl] < 0.05
        if lbl in map_12:
            mat[i, 1] = map_12[lbl]
            sig[i, 1] = map_12_p[lbl] < 0.05

    row_labels = [_short(l) for l in labels]

    fig, ax = plt.subplots(figsize=(6, max(4, n_rows * 0.65 + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="−ρ (Spearman)", fraction=0.06, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(
        "Heatmap −ρ(Spearman)  |  ancrage 1000 COCO\n"
        "✱ = p<0.05",
        fontsize=11, fontweight="bold"
    )

    for i in range(n_rows):
        for j in range(2):
            v = mat[i, j]
            if not np.isnan(v):
                star  = " ✱" if sig[i, j] else ""
                color = "white" if abs(v) > 0.65 else "black"
                ax.text(j, i, f"{v:+.2f}{star}",
                        ha="center", va="center", fontsize=10,
                        color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--res-dir", type=str, default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    res_dir = Path(args.res_dir) if args.res_dir else RES_DIR
    out_dir = res_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Résultats dans : {res_dir}")
    print(f"Plots dans     : {out_dir}\n")

    sum_48_path = find_latest("raise_mono_g48x10_summary_*.csv", res_dir)
    det_12_path = find_latest("raise_mono_g12x40_detailed_*.csv", res_dir)
    sum_12_path = find_latest("raise_mono_g12x40_summary_*.csv", res_dir)

    print(f"Summary 48×10 : {sum_48_path.name}")
    print(f"Detail  12×40 : {det_12_path.name}")
    print(f"Summary 12×40 : {sum_12_path.name}\n")

    rows_48 = load_csv(sum_48_path)
    rows_12 = load_csv(sum_12_path)
    detail  = load_detail_csv(det_12_path)

    # Figure 1 — barplot −ρ (48×10)
    plot_neg_rho_barplot(rows_48, out_dir / "fig1_neg_rho_barplot.png")

    # Figure 2 — scatter avec régression linéaire (12×40)
    plot_scatter_panels(detail, rows_12, out_dir / "fig2_scatter_mmd_vs_mos.png")

    # Figure 3 — heatmap −ρ par granularité
    plot_heatmap(rows_48, rows_12, out_dir / "fig3_heatmap.png")

    print(f"\nFigures dans : {out_dir}")


if __name__ == "__main__":
    main()
