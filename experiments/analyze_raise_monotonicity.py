"""
analyze_raise_monotonicity.py
=============================
Graphiques descriptifs pour evaluate_raise_monotonicity.py.

Lit automatiquement les CSV les plus récents dans results/raise_monotonicity/.
Gère les deux modes : distribution (NxM) et point (N=1).

Figures (par mode) :
  1. Bar chart −ρ(Spearman) pour toutes les métriques (triées)
  2. Scatter MMD vs MOS — meilleure métrique par backbone
  3. Heatmap −ρ : gamma × backbone

Si les deux modes existent :
  4. Comparaison côte-à-côte distribution vs point (meilleur gamma par backbone)

Usage :
    python experiments/analyze_raise_monotonicity.py
    python experiments/analyze_raise_monotonicity.py --results-dir results/raise_monotonicity
    python experiments/analyze_raise_monotonicity.py --pattern "coco1000_g240x2"
"""

import argparse
import csv
import glob as glob_module
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist


# ============================================================
# CONFIG — Modifier ici pour adapter les graphiques
# ============================================================

BACKBONE_COLORS = {
    "dc_ae":    ("#9C27B0", "DC-AE — gram spatial l10"),
    "sd_vae":   ("#2196F3", "SD-VAE — gram spatial l7"),
    "dinov2":   ("#FF9800", "DinoV2 ViT-B/14 — gram spatial l5"),
    "CMMD":     ("#4CAF50", "CMMD — CLIP ViT-L/14 (γ fixe)"),
}

GAMMAS_ORDER = ["1.00e-06", "1.00e-05", "2.80e-05", "1.00e-04", "1.00e-03", "median"]

MODELS_ORDER = None  # None = auto-détection


# ============================================================
# Utilitaires
# ============================================================

def detect_model(label: str) -> str:
    if "CMMD" in label:
        return "CMMD"
    for key in BACKBONE_COLORS:
        if key in label:
            return key
    return "unknown"


def get_color(model: str) -> str:
    return BACKBONE_COLORS.get(model, ("#888888", model))[0]


def get_legend_label(model: str) -> str:
    return BACKBONE_COLORS.get(model, ("#888888", model))[1]


def short_metric_label(label: str) -> str:
    return (label
            .replace("MMDpt_", "pt:")
            .replace("CMMDpt_", "pt:CMMD_")
            .replace("MMD_", "")
            .replace("_vitb14", "")
            .replace("dinov2", "DINOv2")
            .replace("sd_vae", "SD-VAE")
            .replace("dc_ae", "DC-AE")
            .replace("CLIP_ViT-L14", "CLIP-L14"))


def latest_csv(results_dir: str, pattern: str) -> str | None:
    files = sorted(glob_module.glob(os.path.join(results_dir, pattern)))
    if not files:
        return None
    print(f"  Chargement : {os.path.basename(files[-1])}")
    return files[-1]


def load_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_rows(sum_rows, det_rows):
    for r in sum_rows:
        r["neg_rho"] = float(r["neg_rho_mos_rating"])
        r["p"]       = float(r["p_mos_rating"])
        r["model"]   = detect_model(r["metric_label"])
    for r in det_rows:
        r["mos_r_mean"] = float(r["mos_r_mean"])
        r["mmd_score"]  = float(r["mmd_score"])
        r["group_idx"]  = int(r["group_idx"])
        r["model"]      = detect_model(r["metric_label"])


def compute_rho_crit(n_groups: int) -> float:
    t_crit = t_dist.ppf(0.975, df=n_groups - 2)
    return t_crit / math.sqrt(t_crit ** 2 + n_groups - 2)


# ============================================================
# Figure 1 — Bar chart −ρ pour toutes les métriques
# ============================================================

def plot_bar_chart(sum_rows: list, n_groups: int, n_anchor: int,
                   group_size: int, rho_crit: float, out_dir: str,
                   mode_tag: str = ""):
    sorted_sum = sorted(sum_rows, key=lambda x: -x["neg_rho"])

    labels   = [short_metric_label(r["metric_label"]) for r in sorted_sum]
    neg_rhos = [r["neg_rho"] for r in sorted_sum]
    ps       = [r["p"] for r in sorted_sum]
    colors   = [get_color(r["model"]) for r in sorted_sum]

    mode_str = "point (N=1)" if group_size == 1 else f"distribution (N={group_size})"

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    ax.set_title(
        f"Monotonicité RAISE [{mode_str}] — −ρ(Spearman) entre MMD et MOS\n"
        f"ancrage {n_anchor} COCO  |  {n_groups} groupes × {group_size} images  |  "
        f"seuil sig. p<0.05 : |ρ| > {rho_crit:.3f}",
        fontsize=11, fontweight="bold"
    )
    bars = ax.bar(range(len(labels)), neg_rhos,
                  color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(rho_crit, color="red", linewidth=1.8, linestyle="--", alpha=0.85,
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
                fontsize=7.5, color=txt_color, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel("−ρ (Spearman)  [>0 : MMD décroît avec MOS → bon]", fontsize=11)
    ax.set_ylim(-1.1, 1.15)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    seen, patches = [], []
    for r in sorted_sum:
        m = r["model"]
        if m not in seen:
            seen.append(m)
            patches.append(mpatches.Patch(color=get_color(m), label=get_legend_label(m)))
    patches.append(plt.Line2D([0], [0], color="red", linewidth=1.8, linestyle="--",
                              label=f"Seuil sig. |ρ| > {rho_crit:.3f}"))
    ax.legend(handles=patches, fontsize=9, loc="lower right")

    plt.tight_layout()
    suffix = f"_{mode_tag}" if mode_tag else ""
    path = os.path.join(out_dir, f"fig1_neg_rho_barplot{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1{suffix} : {path}")


# ============================================================
# Figure 2 — Scatter MMD vs MOS (meilleure métrique par backbone)
# ============================================================

def plot_scatter(sum_rows: list, det_rows: list, n_groups: int,
                 n_anchor: int, group_size: int, out_dir: str,
                 mode_tag: str = ""):
    models_seen = []
    for r in sum_rows:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    panels = []
    for model in models_seen:
        cands = [r for r in sum_rows if r["model"] == model]
        best  = max(cands, key=lambda x: x["neg_rho"])
        panels.append((model, best["metric_label"]))

    n_panels = len(panels)
    if n_panels == 0:
        return

    mode_str = "point (N=1)" if group_size == 1 else f"distribution (N={group_size})"

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"MMD vs MOS [{mode_str}] — {n_groups} groupes  |  ancrage {n_anchor} COCO\n"
        f"(groupe 1 = pire qualité, groupe {n_groups} = meilleure qualité)",
        fontsize=12, fontweight="bold"
    )

    for ax, (model_name, mlabel) in zip(axes, panels):
        pts = sorted([r for r in det_rows if r["metric_label"] == mlabel],
                     key=lambda x: x["group_idx"])
        mos = np.array([r["mos_r_mean"] for r in pts])
        mmd = np.array([r["mmd_score"] for r in pts])

        color   = get_color(model_name)
        sr      = next(r for r in sum_rows if r["metric_label"] == mlabel)
        neg_rho = sr["neg_rho"]
        p_val   = sr["p"]
        gstr    = sr.get("gamma_str", "?")
        sig_str = " ✱" if p_val < 0.05 else ""

        # Pour N=1 (480 points), petits marqueurs sans numéros
        if group_size == 1:
            ax.scatter(mos, mmd, color=color, s=15, zorder=4, alpha=0.6,
                       edgecolors="none")
        else:
            ax.scatter(mos, mmd, color=color, s=70, zorder=4,
                       edgecolors="white", linewidth=0.6)
            for i, (x, y) in enumerate(zip(mos, mmd)):
                ax.annotate(str(i + 1), (x, y), fontsize=7.5,
                            xytext=(4, 3), textcoords="offset points", color="#333")

        if len(mos) > 1:
            m_fit, b_fit = np.polyfit(mos, mmd, 1)
            x_line = np.linspace(mos.min(), mos.max(), 100)
            ax.plot(x_line, m_fit * x_line + b_fit, color=color,
                    linewidth=1.5, alpha=0.7, linestyle="--")

        display_name = get_legend_label(model_name).split(" — ")[0]
        ax.set_title(
            f"{display_name}  γ={gstr}\n"
            f"−ρ = {neg_rho:+.3f}{sig_str}   p = {p_val:.3e}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("MOS" + ("" if group_size == 1 else " moyen du groupe"), fontsize=10)
        ax.set_ylabel("MMD²", fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    suffix = f"_{mode_tag}" if mode_tag else ""
    path = os.path.join(out_dir, f"fig2_scatter_mmd_vs_mos{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2{suffix} : {path}")


# ============================================================
# Figure 3 — Heatmap −ρ : gamma × backbone
# ============================================================

def plot_heatmap(sum_rows: list, n_groups: int, n_anchor: int,
                 group_size: int, out_dir: str, mode_tag: str = ""):
    if MODELS_ORDER is not None:
        models = MODELS_ORDER
    else:
        seen = []
        for r in sum_rows:
            if r["model"] not in seen:
                seen.append(r["model"])
        models = seen

    gammas_in_data = []
    for r in sum_rows:
        if r["model"] != "CMMD" and r.get("gamma_str", "") not in gammas_in_data:
            gammas_in_data.append(r.get("gamma_str", ""))
    gammas = [g for g in GAMMAS_ORDER if g in gammas_in_data]
    if not gammas:
        gammas = gammas_in_data

    has_cmmd = any(r["model"] == "CMMD" for r in sum_rows)
    n_rows = len(gammas) + (1 if has_cmmd else 0)

    mat     = np.full((n_rows, len(models)), np.nan)
    sig_mat = np.zeros((n_rows, len(models)), dtype=bool)

    for r in sum_rows:
        model = r["model"]
        if model not in models:
            continue
        ci   = models.index(model)
        gstr = r.get("gamma_str", "")

        if model == "CMMD":
            ri = len(gammas)
        elif gstr in gammas:
            ri = gammas.index(gstr)
        else:
            continue

        mat[ri, ci]     = r["neg_rho"]
        sig_mat[ri, ci] = r["p"] < 0.05

    row_labels = [f"γ = {g}" for g in gammas]
    if has_cmmd:
        cmmd_gamma = next((r.get("gamma_str", "?") for r in sum_rows if r["model"] == "CMMD"), "?")
        row_labels.append(f"CMMD\nγ={cmmd_gamma} (fixe)")

    col_labels = [get_legend_label(m).split(" — ")[0] for m in models]

    mode_str = "point (N=1)" if group_size == 1 else f"distribution (N={group_size})"

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.5), max(6, n_rows * 0.8)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, label="−ρ (Spearman)", fraction=0.04, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(
        f"Heatmap −ρ(Spearman) [{mode_str}]  |  ancrage {n_anchor} COCO\n"
        f"{n_groups} groupes × {group_size} RAISE IA  |  ✱ = p<0.05",
        fontsize=11, fontweight="bold"
    )

    for ri in range(n_rows):
        for ci in range(len(models)):
            val = mat[ri, ci]
            if not np.isnan(val):
                star  = " ✱" if sig_mat[ri, ci] else ""
                color = "white" if abs(val) > 0.65 else "black"
                ax.text(ci, ri, f"{val:+.2f}{star}",
                        ha="center", va="center", fontsize=11,
                        color=color, fontweight="bold")

    if has_cmmd:
        ax.axhline(len(gammas) - 0.5, color="white", linewidth=2.5)

    plt.tight_layout()
    suffix = f"_{mode_tag}" if mode_tag else ""
    path = os.path.join(out_dir, f"fig3_heatmap_gamma_backbone{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 3{suffix} : {path}")


# ============================================================
# Figure 4 — Comparaison distribution vs point (côte-à-côte)
# ============================================================

def plot_comparison(sum_dist: list, sum_point: list, n_anchor: int, out_dir: str):
    """Bar chart comparant −ρ distribution vs point pour chaque backbone (meilleur gamma)."""

    def best_per_model(rows):
        models = []
        for r in rows:
            if r["model"] not in models:
                models.append(r["model"])
        result = {}
        for m in models:
            cands = [r for r in rows if r["model"] == m]
            result[m] = max(cands, key=lambda x: x["neg_rho"])
        return result

    best_dist  = best_per_model(sum_dist)
    best_point = best_per_model(sum_point)

    # Fusionner les modèles présents dans les deux
    all_models = list(dict.fromkeys(list(best_dist.keys()) + list(best_point.keys())))

    n = len(all_models)
    x = np.arange(n)
    width = 0.35

    dist_vals  = [best_dist[m]["neg_rho"]  if m in best_dist  else 0 for m in all_models]
    point_vals = [best_point[m]["neg_rho"] if m in best_point else 0 for m in all_models]
    dist_p     = [best_dist[m]["p"]  if m in best_dist  else 1 for m in all_models]
    point_p    = [best_point[m]["p"] if m in best_point else 1 for m in all_models]
    colors     = [get_color(m) for m in all_models]

    # Group sizes pour annotation
    gs_dist  = int(sum_dist[0].get("group_size", "?"))  if sum_dist  else "?"
    gs_point = int(sum_point[0].get("group_size", "?")) if sum_point else "?"
    ng_dist  = int(sum_dist[0].get("n_groups", "?"))  if sum_dist  else "?"
    ng_point = int(sum_point[0].get("n_groups", "?")) if sum_point else "?"

    fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 6))
    ax.set_title(
        f"Comparaison distribution vs point — −ρ(Spearman)  |  ancrage {n_anchor} COCO\n"
        f"Distribution : {ng_dist}×{gs_dist}  |  Point : {ng_point}×{gs_point}",
        fontsize=12, fontweight="bold"
    )

    bars_d = ax.bar(x - width / 2, dist_vals,  width, label=f"Distribution (N={gs_dist})",
                    color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    bars_p = ax.bar(x + width / 2, point_vals, width, label=f"Point (N=1)",
                    color=colors, edgecolor="white", linewidth=0.8, alpha=0.55,
                    hatch="//")

    # Annotations
    for bar, p_val, nr in zip(bars_d, dist_p, dist_vals):
        star = "✱" if p_val < 0.05 else ""
        mid = bar.get_y() + bar.get_height() / 2
        txt_c = "white" if abs(nr) > 0.4 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2, mid,
                f"{nr:+.2f}{star}", ha="center", va="center",
                fontsize=9, color=txt_c, fontweight="bold")

    for bar, p_val, nr in zip(bars_p, point_p, point_vals):
        star = "✱" if p_val < 0.05 else ""
        mid = bar.get_y() + bar.get_height() / 2
        txt_c = "white" if abs(nr) > 0.4 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2, mid,
                f"{nr:+.2f}{star}", ha="center", va="center",
                fontsize=9, color=txt_c, fontweight="bold")

    # Gamma annotations sous chaque barre
    for i, m in enumerate(all_models):
        if m in best_dist:
            g = best_dist[m].get("gamma_str", "")
            ax.text(x[i] - width / 2, -0.08, f"γ={g}", ha="center",
                    fontsize=7, color="#666", rotation=30)
        if m in best_point:
            g = best_point[m].get("gamma_str", "")
            ax.text(x[i] + width / 2, -0.08, f"γ={g}", ha="center",
                    fontsize=7, color="#666", rotation=30)

    ax.set_xticks(x)
    ax.set_xticklabels([get_legend_label(m).split(" — ")[0] for m in all_models],
                       fontsize=11, fontweight="bold")
    ax.set_ylabel("−ρ (Spearman)", fontsize=11)
    ax.set_ylim(-1.1, 1.15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Légende
    legend_patches = [
        mpatches.Patch(facecolor="#999", alpha=0.9, label=f"Distribution (N={gs_dist})"),
        mpatches.Patch(facecolor="#999", alpha=0.55, hatch="//", label="Point (N=1)"),
    ]
    ax.legend(handles=legend_patches, fontsize=10, loc="lower right")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_comparison_dist_vs_point.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 4 : {path}")


# ============================================================
# Main
# ============================================================

def generate_figures(sum_rows, det_rows, n_groups, group_size, n_anchor,
                     out_dir, mode_tag=""):
    """Génère les 3 figures standard pour un jeu de résultats."""
    rho_crit = compute_rho_crit(n_groups)
    print(f"\n{'='*60}")
    print(f"Mode : {'point (N=1)' if group_size == 1 else f'distribution (N={group_size})'}")
    print(f"  {n_groups} groupes × {group_size} images  |  seuil |ρ| > {rho_crit:.3f}")
    print(f"  Métriques : {len(sum_rows)}")
    print(f"  Backbones : {list(dict.fromkeys(r['model'] for r in sum_rows))}")
    print(f"{'='*60}")

    plot_bar_chart(sum_rows, n_groups, n_anchor, group_size, rho_crit, out_dir, mode_tag)
    plot_scatter(sum_rows, det_rows, n_groups, n_anchor, group_size, out_dir, mode_tag)


def main():
    parser = argparse.ArgumentParser(description="Analyse graphique des résultats de monotonicité RAISE")
    parser.add_argument("--results-dir", default="results/raise_monotonicity",
                        help="Dossier contenant les CSV")
    parser.add_argument("--pattern", default=None,
                        help="Pattern distribution (ex: 'coco1000_g240x2'). Auto-détecté si absent.")
    parser.add_argument("--out-dir", default=None,
                        help="Dossier de sortie (défaut: {results-dir}/analysis)")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir     = args.out_dir or os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Auto-détection du pattern distribution ----
    if args.pattern:
        pat_dist = args.pattern
    else:
        # Chercher les summary CSV, exclure ceux qui finissent par _point
        summary_files = sorted(glob_module.glob(
            os.path.join(results_dir, "raise_monotonicity_*_summary_*.csv")
        ))
        # Filtrer les fichiers _point
        dist_files = [f for f in summary_files if "_point" not in os.path.basename(f)]
        if not dist_files:
            raise FileNotFoundError(f"Aucun CSV summary dans {results_dir}")
        last = os.path.basename(dist_files[-1])
        parts = last.replace("raise_monotonicity_", "").split("_summary_")
        pat_dist = parts[0]
        print(f"Pattern distribution auto-détecté : {pat_dist}")

    # ---- Chargement CSV distribution ----
    print("\nChargement CSV distribution...")
    sum_dist_path = latest_csv(results_dir, f"raise_monotonicity_{pat_dist}_summary_*.csv")
    det_dist_path = latest_csv(results_dir, f"raise_monotonicity_{pat_dist}_detailed_*.csv")

    sum_dist = load_csv(sum_dist_path) if sum_dist_path else []
    det_dist = load_csv(det_dist_path) if det_dist_path else []

    if sum_dist:
        parse_rows(sum_dist, det_dist)

    # ---- Chargement CSV point (auto-détection) ----
    print("\nRecherche CSV point (N=1)...")
    point_files = sorted(glob_module.glob(
        os.path.join(results_dir, "raise_monotonicity_*_summary_*_point.csv")
    ))
    sum_point, det_point = [], []
    if point_files:
        sum_pt_path = point_files[-1]
        print(f"  Chargement : {os.path.basename(sum_pt_path)}")
        # Déduire le pattern point depuis le nom du summary
        pt_base = os.path.basename(sum_pt_path)
        pt_parts = pt_base.replace("raise_monotonicity_", "").split("_summary_")
        pat_point = pt_parts[0]

        det_pt_path = latest_csv(results_dir, f"raise_monotonicity_{pat_point}_detailed_*_point.csv")
        sum_point = load_csv(sum_pt_path)
        det_point = load_csv(det_pt_path) if det_pt_path else []
        if sum_point:
            parse_rows(sum_point, det_point)
    else:
        print("  Aucun CSV point trouvé")

    # Extraire n_anchor
    n_anchor = 1000
    if "coco" in pat_dist:
        try:
            n_anchor = int(pat_dist.split("coco")[1].split("_")[0])
        except (ValueError, IndexError):
            pass

    # ---- Figures distribution ----
    if sum_dist:
        n_groups_d   = int(sum_dist[0].get("n_groups", 40))
        group_size_d = int(sum_dist[0].get("group_size", 12))
        generate_figures(sum_dist, det_dist, n_groups_d, group_size_d, n_anchor,
                         out_dir, mode_tag="dist")

    # ---- Figures point ----
    if sum_point:
        n_groups_p   = int(sum_point[0].get("n_groups", 480))
        group_size_p = int(sum_point[0].get("group_size", 1))
        generate_figures(sum_point, det_point, n_groups_p, group_size_p, n_anchor,
                         out_dir, mode_tag="point")

    # ---- Figure 4 : comparaison ----
    if sum_dist and sum_point:
        print(f"\n{'='*60}")
        print("Comparaison distribution vs point")
        print(f"{'='*60}")
        plot_comparison(sum_dist, sum_point, n_anchor, out_dir)

    print(f"\nTous les graphiques dans : {out_dir}")


if __name__ == "__main__":
    main()
