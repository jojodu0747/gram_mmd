"""
plot_gamma_sensitivity_dinov2.py
=================================
Figures pour raise_gamma_sensitivity_dinov2.py (DINOv2 L5, 24×20, 3 gammas).

Figures produites :
  01_barplot_neg_rho.png  — barplot −ρ(Spearman) pour les 3 gammas
  02_scatters_mmd_mos.png — grille 3 scatters (MMD vs MOS), 1 par gamma,
                            avec droite de régression linéaire et annotation
                            −ρ + p-value

Usage :
    .venv/bin/python experiments/plot_gamma_sensitivity_dinov2.py
"""

import csv
import glob as glob_module
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "raise_gamma_sensitivity_dinov2"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Chargement des CSV les plus récents ──────────────────────────────

def latest_csv(pattern: str) -> Path:
    files = sorted(glob_module.glob(str(RESULTS_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"Aucun fichier : {RESULTS_DIR / pattern}")
    print(f"  {os.path.basename(files[-1])}")
    return Path(files[-1])


def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


print("Chargement...")
sum_rows = load_csv(latest_csv("gamma_sensitivity_*_summary_*.csv"))
det_rows = load_csv(latest_csv("gamma_sensitivity_*_detailed_*.csv"))

for r in sum_rows:
    r["neg_rho"]   = float(r["neg_rho"])
    r["p_spearman"] = float(r["p_spearman"])
    r["pearson_r"]  = float(r["pearson_r"])
    r["p_pearson"]  = float(r["p_pearson"])
    r["n_groups"]   = int(r["n_groups"])

for r in det_rows:
    r["mos_mean"]  = float(r["mos_mean"])
    r["mmd_score"] = float(r["mmd_score"])
    r["group_idx"] = int(r["group_idx"])

# Ordre des gammas
GAMMA_ORDER = ["median", "10x_median", "50x_median"]
GAMMA_LABELS = {
    "median":     "γ_median",
    "10x_median": "10 × γ_median",
    "50x_median": "50 × γ_median",
}

# Seuil de significativité p<0.05 pour n=24
n_groups = sum_rows[0]["n_groups"]
t_crit   = t_dist.ppf(0.975, df=n_groups - 2)
rho_crit = t_crit / math.sqrt(t_crit ** 2 + n_groups - 2)
print(f"Seuil sig. p<0.05 (n={n_groups}) : |ρ| > {rho_crit:.3f}")

GAMMA_COLORS = {
    "median":     "#1565C0",
    "10x_median": "#FF9800",
    "50x_median": "#F44336",
}


# ── Helper ────────────────────────────────────────────────────────────

def get_sum(gamma_tag: str) -> dict:
    for r in sum_rows:
        if r["gamma_tag"] == gamma_tag:
            return r
    return None


def get_det(gamma_tag: str) -> list:
    return sorted(
        [r for r in det_rows if r["gamma_tag"] == gamma_tag],
        key=lambda x: x["group_idx"]
    )


# ═══════════════════════════════════════════════════════════════════════
#  Figure 1 — Barplot −ρ(Spearman) pour les 3 gammas
# ═══════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_title(
    f"DINOv2 ViT-B/14 L5 — Sensibilité au gamma  |  RAISE {n_groups} groupes × 20 images  |  ancrage 1000 COCO\n"
    f"−ρ(Spearman) entre MMD² et MOS  |  seuil sig. p<0.05 : |ρ| > {rho_crit:.3f}",
    fontsize=11, fontweight="bold"
)

x_pos     = np.arange(len(GAMMA_ORDER))
neg_rhos  = []
ps        = []
colors    = []
gamma_vals = []

for gt in GAMMA_ORDER:
    r = get_sum(gt)
    neg_rhos.append(r["neg_rho"])
    ps.append(r["p_spearman"])
    colors.append(GAMMA_COLORS[gt])
    gamma_vals.append(r["gamma_val"])

bars = ax1.bar(x_pos, neg_rhos, color=colors, edgecolor="white",
               linewidth=0.5, zorder=3, width=0.6)

ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.axhline( rho_crit, color="red", linewidth=1.5, linestyle="--", alpha=0.8,
             label=f"Seuil sig. p<0.05 (|ρ|={rho_crit:.3f})")
ax1.axhline(-rho_crit, color="red", linewidth=1.5, linestyle="--", alpha=0.3)

for bar, p, nr, gv in zip(bars, ps, neg_rhos, gamma_vals):
    if p < 0.05:
        yt = bar.get_height() + 0.03 if nr >= 0 else bar.get_y() - 0.07
        ax1.text(bar.get_x() + bar.get_width() / 2, yt, "✱",
                 ha="center", fontsize=13, color="black")
    mid = bar.get_y() + bar.get_height() / 2
    txt_col = "white" if abs(nr) > 0.4 else "black"
    ax1.text(bar.get_x() + bar.get_width() / 2, mid,
             f"{nr:+.3f}", ha="center", va="center",
             fontsize=9, color=txt_col, fontweight="bold")
    ax1.text(bar.get_x() + bar.get_width() / 2, -0.98,
             f"γ={gv}", ha="center", va="bottom",
             fontsize=7, color="#555", rotation=0)

ax1.set_xticks(x_pos)
ax1.set_xticklabels([GAMMA_LABELS[gt] for gt in GAMMA_ORDER], fontsize=10)
ax1.set_ylabel("−ρ (Spearman)  [>0 = bon : MMD↓ quand MOS↑]", fontsize=10)
ax1.set_ylim(-1.1, 1.15)
ax1.yaxis.grid(True, alpha=0.3)
ax1.set_axisbelow(True)
ax1.legend(fontsize=9, loc="lower right")

plt.tight_layout()
p1 = PLOTS_DIR / "01_barplot_neg_rho.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure 1 : {p1}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 2 — Grille 3 scatters (MMD vs MOS)
# ═══════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(
    f"DINOv2 ViT-B/14 L5 — MMD² vs MOS  |  RAISE {n_groups} groupes × 20 images  |  ancrage 1000 COCO\n"
    "(groupe 1 = pire qualité, groupe 24 = meilleure qualité)",
    fontsize=12, fontweight="bold"
)

for ax, gamma_tag in zip(axes, GAMMA_ORDER):
    pts = get_det(gamma_tag)
    sr  = get_sum(gamma_tag)

    mos = np.array([r["mos_mean"]  for r in pts])
    mmd = np.array([r["mmd_score"] for r in pts])

    color   = GAMMA_COLORS[gamma_tag]
    neg_rho = sr["neg_rho"]
    rho_val = sr["spearman_rho"]
    p_val   = sr["p_spearman"]
    r_val   = sr["pearson_r"]
    gval    = sr["gamma_val"]
    sig_str = " ✱" if float(p_val) < 0.05 else ""

    ax.scatter(mos, mmd, color=color, s=60, zorder=4,
               edgecolors="white", linewidth=0.6)

    m_fit, b_fit = np.polyfit(mos, mmd, 1)
    x_line = np.linspace(mos.min(), mos.max(), 100)
    ax.plot(x_line, m_fit * x_line + b_fit, color=color,
            linewidth=1.8, alpha=0.75, linestyle="--")

    for i, (x, y) in enumerate(zip(mos, mmd)):
        ax.annotate(str(i + 1), (x, y), fontsize=6.5,
                    xytext=(3, 2), textcoords="offset points", color="#333")

    ax.set_title(
        f"{GAMMA_LABELS[gamma_tag]}\nγ = {gval}\n"
        f"−ρ = {neg_rho:+.3f}{sig_str}   p = {float(p_val):.3e}\n"
        f"r = {r_val:+.3f}",
        fontsize=9, fontweight="bold"
    )
    ax.set_xlabel("MOS moyen du groupe", fontsize=9)
    ax.set_ylabel("MMD²", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)

plt.tight_layout()
p2 = PLOTS_DIR / "02_scatters_mmd_vs_mos.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure 2 : {p2}")

print(f"\nPlots dans : {PLOTS_DIR}")
