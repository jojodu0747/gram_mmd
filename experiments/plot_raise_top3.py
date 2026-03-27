"""
plot_raise_top3.py
===================
Tracé des résultats raise_top3_coco_distdist.py depuis le CSV + caches .npz.

Usage :
    .venv/bin/python experiments/plot_raise_top3.py
"""

import csv
import glob as glob_module
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.stats import linregress

BASE_DIR  = Path(__file__).resolve().parent.parent
RESULT    = BASE_DIR / "results" / "raise_top3_distdist"
CSV_FILE  = sorted(RESULT.glob("summary_*.csv"))[-1]
CACHE     = RESULT / "cache"
OUT       = RESULT / "plots"
RAISE_ROOT = BASE_DIR / "dataset" / "RAISE" / "dataset"

N_GROUPS   = 24
GROUP_SIZE = 20

# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "DINOv2 L5":  "#F57C00",
    "DC-AE L12":  "#2E7D32",
    "SD-VAE M13": "#1565C0",
    "CMMD":       "#C62828",
}
MARKERS = {
    "median": "o",
    "fixed":  "D",
}
HATCH = {
    "median": "",
    "fixed":  "///",
}


# ── chargement CSV ────────────────────────────────────────────────────────────
def load_csv():
    rows = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "label":       r["label"],
                "backbone":    r["backbone"],
                "layer":       r["layer"],
                "gamma_tag":   r["gamma_tag"],
                "gamma_val":   float(r["gamma_val"]) if r["gamma_val"] else None,
                "spearman":    float(r["spearman"]),
                "spearman_p":  float(r["spearman_p"]),
                "kendall":     float(r["kendall"]),
                "kendall_p":   float(r["kendall_p"]),
                "mono_strict": float(r["mono_strict"]),
            })
    return rows


# ── chargement scores par groupe ──────────────────────────────────────────────
def load_scores(rows, device="cpu"):
    """Recharge les scores MMD²/CMMD depuis les caches .npz."""
    import torch

    scores = {}
    for r in rows:
        bb  = r["backbone"].replace(" ", "_").replace("-", "_")
        lay = r["layer"]
        tag = r["gamma_tag"]

        if r["backbone"] == "CMMD":
            # Nouveau nommage (avec groupage) puis legacy
            p_cmmd_new = CACHE / f"cmmd_scores_{N_GROUPS}x{GROUP_SIZE}.npz"
            p_cmmd_leg = CACHE / "cmmd_scores.npz"
            cmmd_path  = p_cmmd_new if p_cmmd_new.exists() else p_cmmd_leg
            d = np.load(cmmd_path)
            scores[r["label"]] = d["median" if tag == "median" else "fixed"]
        else:
            # backbone clé  ex: "DC_AE_L12" → "dc_ae_L12"
            bb_map = {
                "DINOv2_L5":  "dinov2_vitb14_L05",
                "DC_AE_L12":  "dc_ae_L12",
                "SD_VAE_M13": "sd_vae_L13",
            }
            key = f"{r['backbone'].replace(' ', '_').replace('-', '_')}"
            # reconstruit la clé cache depuis backbone + layer
            backbone_id = {
                "DINOv2 L5":  "dinov2_vitb14",
                "DC-AE L12":  "dc_ae",
                "SD-VAE M13": "sd_vae",
            }[r["backbone"]]
            layer_i = int(r["layer"])
            npz_key = f"{backbone_id}_L{layer_i:02d}"
            # Nouveau nommage (avec groupage) puis legacy
            p_new    = CACHE / f"{npz_key}_dist_{N_GROUPS}x{GROUP_SIZE}.npz"
            p_legacy = CACHE / f"{npz_key}_distances.npz"
            npz_path = p_new if p_new.exists() else p_legacy
            npz = np.load(npz_path)
            D_rr = npz["D_rr_triu"]
            D_qq = npz["D_qq_triu"]   # (N_GROUPS, n_pairs)
            D_rq = npz["D_rq"]         # (N_GROUPS, N_ANCHOR, GROUP_SIZE)

            g = r["gamma_val"]
            K_rr = float(np.exp(-g * D_rr).mean())
            K_qq = np.exp(-g * D_qq).mean(axis=1)   # (N_GROUPS,)
            K_rq = np.exp(-g * D_rq).mean(axis=(1, 2))
            scores[r["label"]] = K_qq + K_rr - 2 * K_rq

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — scatter ρ vs τ  (signé)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_scatter(rows, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.set_facecolor("#f5f5f5")

    # Quadrant de référence : correct = ρ < 0, τ < 0 (MMD² ↑ quand qualité ↓)
    ax.axhline(0, color="#888", lw=0.8, ls="--", zorder=1)
    ax.axvline(0, color="#888", lw=0.8, ls="--", zorder=1)
    ax.fill_betweenx([-1, 0], -1, 0, color="#e8f5e9", alpha=0.55, zorder=0)
    ax.text(-0.85, -0.08, "correct\n(MMD²↑ quand qualité↓)",
            fontsize=7.5, color="#2e7d32", va="top", style="italic")

    for r in rows:
        sp = r["spearman"]
        kt = r["kendall"]
        bb = r["backbone"]
        c  = COLORS.get(bb, "#9C27B0")
        m  = MARKERS.get(r["gamma_tag"], "o")
        ax.scatter(kt, sp, color=c, marker=m, s=120,
                   edgecolors="black", linewidths=0.8, zorder=5)

    # Annotations manuelles sans chevauchement
    offsets = {
        "GMMD DINOv2 L5 γ_med":  (-10, 8),
        "GMMD DC-AE L12 γ_med":  (8,  8),
        "GMMD SD-VAE M13 γ_med": (8, -14),
        "CMMD γ_med":             (-60, 8),
        "CMMD γ=0.005":           (-68, -14),
    }
    for r in rows:
        xy = (r["kendall"], r["spearman"])
        dx, dy = offsets.get(r["label"], (8, 5))
        ax.annotate(
            r["label"], xy,
            textcoords="offset points", xytext=(dx, dy),
            fontsize=7, zorder=6,
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.7,
                            shrinkA=0, shrinkB=3) if abs(dx) > 20 else None,
        )

    handles_c = [
        Line2D([0],[0], marker="o", color=c, ls="None", ms=9,
               mec="black", mew=0.7, label=n)
        for n, c in COLORS.items()
        if any(r["backbone"] == n for r in rows)
    ]
    handles_m = [
        Line2D([0],[0], marker=m, color="gray", ls="None", ms=8,
               mec="black", mew=0.5, label=f"γ_med" if t == "median" else f"γ fixe")
        for t, m in MARKERS.items()
        if any(r["gamma_tag"] == t for r in rows)
    ]
    leg1 = ax.legend(handles=handles_c, fontsize=8, loc="upper left",
                     title="Backbone", framealpha=0.92)
    ax.add_artist(leg1)
    ax.legend(handles=handles_m, fontsize=8, loc="lower right",
              title="γ", framealpha=0.92)

    ax.set_xlabel("Spearman ρ", fontsize=11)
    ax.set_ylabel("Kendall τ", fontsize=11)
    ax.set_xlim(-0.95, 0.7); ax.set_ylim(-0.75, 0.55)
    ax.set_title(
        f"RAISE IA  ·  1 000 COCO anchor  ·  {N_GROUPS}×{GROUP_SIZE}\n"
        "Spearman ρ vs Kendall τ  (valeurs signées)",
        fontsize=10.5, fontweight="bold"
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = out / "01_scatter_signed.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — barplot ρ signé + mono
# ═══════════════════════════════════════════════════════════════════════════════
def fig_barplot(rows, out: Path):
    # Tri par ρ croissant (plus négatif = meilleur pour nous)
    sorted_r = sorted(rows, key=lambda x: x["spearman"])

    labels = [r["label"] for r in sorted_r]
    sp_v   = [r["spearman"]    for r in sorted_r]
    kt_v   = [r["kendall"]     for r in sorted_r]
    mono_v = [r["mono_strict"] for r in sorted_r]
    colors = [COLORS.get(r["backbone"], "#9C27B0") for r in sorted_r]
    hatches= [HATCH.get(r["gamma_tag"], "") for r in sorted_r]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    fig.suptitle(
        f"RAISE IA  ·  1 000 COCO anchor  ·  {N_GROUPS}×{GROUP_SIZE}\n"
        "Valeurs signées (négatif = corrélation correcte)",
        fontsize=11, fontweight="bold", y=1.01
    )

    def bar_panel(ax, vals, ylabel, ref=None):
        bars = ax.bar(x, vals, color=colors, edgecolor="black",
                      linewidth=0.6, hatch=hatches)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(0, color="gray", lw=0.8)
        if ref is not None:
            ax.axhline(ref, color="#888", lw=0.8, ls=":", alpha=0.7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_facecolor("#f5f5f5")
        return bars

    bar_panel(axes[0], sp_v,   "Spearman ρ")
    bar_panel(axes[1], kt_v,   "Kendall τ")
    bar_panel(axes[2], mono_v, "Mono stricte", ref=0.5)
    axes[2].set_ylim(0, 1)
    axes[2].axhline(0.5, color="#888", lw=0.8, ls=":", label="hasard")
    axes[2].legend(fontsize=7, loc="upper right")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=32, ha="right", fontsize=8)

    # Valeurs numériques au-dessus/dessous des barres
    for ax, vals in zip(axes[:2], [sp_v, kt_v]):
        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, fontweight="bold")

    handles = [
        Line2D([0],[0], color=c, lw=8, label=n)
        for n, c in COLORS.items()
        if any(r["backbone"] == n for r in rows)
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="lower left",
                   ncol=2, framealpha=0.9)

    plt.tight_layout()
    p = out / "02_barplot_signed.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — profils MMD² lissés
# ═══════════════════════════════════════════════════════════════════════════════
def fig_profiles(rows, scores, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ranks = np.arange(N_GROUPS)
    win   = min(7, N_GROUPS // 3 * 2 + 1)   # fenêtre Savitzky-Golay impaire

    # ── panneau haut : GMMD ──────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor("#f5f5f5")
    ax0.set_title("GMMD Gram-MMD² dist-vs-dist", fontsize=10, fontweight="bold")

    for r in rows:
        if r["backbone"] == "CMMD":
            continue
        sc   = scores[r["label"]]
        mn, mx = sc.min(), sc.max()
        sc_n = (sc - mn) / (mx - mn + 1e-12)
        sc_s = savgol_filter(sc_n, win, 3) if len(sc_n) > win else sc_n

        color = COLORS[r["backbone"]]
        ax0.plot(ranks, sc_s, color=color, lw=2.2, label=r["label"], alpha=0.9)
        ax0.scatter(ranks, sc_n, color=color, s=18, alpha=0.35, zorder=4)

    ax0.set_ylabel("MMD² normalisé [0–1]", fontsize=10)
    ax0.legend(fontsize=8, framealpha=0.9)
    ax0.grid(True, alpha=0.25)
    ax0.annotate("← pire qualité", xy=(0.02, 0.05), xycoords="axes fraction",
                 fontsize=8, color="#555")
    ax0.annotate("meilleure qualité →", xy=(0.72, 0.05), xycoords="axes fraction",
                 fontsize=8, color="#555")

    # ── panneau bas : CMMD ───────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor("#f5f5f5")
    ax1.set_title("CMMD (CLIP ViT-L/14@336) — corrélation inverse sur RAISE IA",
                  fontsize=10, fontweight="bold", color="#C62828")

    for r in rows:
        if r["backbone"] != "CMMD":
            continue
        sc   = scores[r["label"]]
        mn, mx = sc.min(), sc.max()
        sc_n = (sc - mn) / (mx - mn + 1e-12)
        sc_s = savgol_filter(sc_n, win, 3) if len(sc_n) > win else sc_n
        ls   = "-" if r["gamma_tag"] == "median" else "--"
        glbl = "γ_med" if r["gamma_tag"] == "median" else f"γ={r['gamma_val']:.3f}"
        ax1.plot(ranks, sc_s, color=COLORS["CMMD"], lw=2.2, ls=ls,
                 label=f"CMMD {glbl}  (ρ={r['spearman']:+.2f})", alpha=0.9)
        ax1.scatter(ranks, sc_n, color=COLORS["CMMD"], s=18, alpha=0.35, zorder=4)

    ax1.set_ylabel("score normalisé [0–1]", fontsize=10)
    ax1.set_xlabel("Rang MOS (0 = pire qualité)", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    ax1.annotate(
        "CMMD croît avec la qualité → mauvais sens\n"
        "(attendu : MMD² élevé pour qualité basse)",
        xy=(0.5, 0.82), xycoords="axes fraction",
        fontsize=8, color="#C62828", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fce4e4", ec="#C62828", alpha=0.85)
    )

    plt.tight_layout()
    p = out / "03_profiles_gmmd_vs_cmmd.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — tableau récapitulatif
# ═══════════════════════════════════════════════════════════════════════════════
def fig_table(rows, out: Path):
    sorted_r = sorted(rows, key=lambda x: x["spearman"])

    labels  = [r["label"]        for r in sorted_r]
    sp_v    = [f"{r['spearman']:+.3f}" for r in sorted_r]
    sp_p    = [f"{r['spearman_p']:.2e}" for r in sorted_r]
    kt_v    = [f"{r['kendall']:+.3f}"  for r in sorted_r]
    mono_v  = [f"{r['mono_strict']:.1%}" for r in sorted_r]
    direction = ["✓" if r["spearman"] < 0 else "✗" for r in sorted_r]

    col_labels = ["Label", "Spearman ρ", "p-val", "Kendall τ", "Mono", "Dir."]
    cell_text  = list(zip(labels, sp_v, sp_p, kt_v, mono_v, direction))

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")

    cell_colors = []
    for r, d in zip(sorted_r, direction):
        row_c = ["#ffffff"] * 5
        if d == "✓":
            row_c = ["#e8f5e9"] * 5
        else:
            row_c = ["#ffebee"] * 5
        row_c.append("#e8f5e9" if d == "✓" else "#ffcdd2")
        cell_colors.append(row_c)

    t = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center", loc="center",
        cellColours=cell_colors,
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    for (row, col), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Résultats RAISE IA  ·  1 000 COCO anchor  ·  {N_GROUPS}×{GROUP_SIZE}\n"
        "✓ = direction correcte (ρ < 0),  ✗ = direction incorrecte",
        fontsize=10, fontweight="bold", pad=15
    )
    plt.tight_layout()
    p = out / "04_table.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MOS par groupe (axe X de la régression)
# ═══════════════════════════════════════════════════════════════════════════════

def load_group_mos():
    """Recharge le MOS moyen par groupe depuis les fichiers RAISE."""
    mos_dict = {}
    for cp in [RAISE_ROOT / "ratings" / "train.csv",
               RAISE_ROOT / "ratings" / "test.csv"]:
        with open(cp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = float(row["MOS_Rating"])

    all_imgs = {}
    for folder in [RAISE_ROOT / "images" / "train_images",
                   RAISE_ROOT / "images" / "test_images"]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[Path(p).name] = p

    items = sorted(
        [(path, mos_dict[fname])
         for fname, path in all_imgs.items()
         if fname.startswith("f") and fname in mos_dict],
        key=lambda x: x[1]
    )
    group_mos = np.array([
        np.mean([items[g * GROUP_SIZE + i][1] for i in range(GROUP_SIZE)])
        for g in range(N_GROUPS)
    ])
    return group_mos


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 5 — régression linéaire MMD² ~ MOS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_regression(rows, scores, group_mos, out: Path):
    """
    Un panneau par backbone.
    X = MOS moyen du groupe, Y = MMD² brut (non normalisé).
    Droite de régression + coefficient de régression (pente normalisée).
    """
    # On garde un ordre fixe : GMMD d'abord, CMMD ensuite
    gmmd_rows = [r for r in rows if r["backbone"] != "CMMD"]
    cmmd_rows = [r for r in rows if r["backbone"] == "CMMD"]
    ordered   = gmmd_rows + cmmd_rows

    n = len(ordered)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()

    x = group_mos   # MOS moyen par groupe (croissant)

    for ax, r in zip(axes, ordered):
        sc     = scores[r["label"]].astype(np.float64)
        color  = COLORS.get(r["backbone"], "#9C27B0")
        hatch  = "///" if r["gamma_tag"] == "fixed" else ""

        # Régression linéaire y = a·x + b
        slope, intercept, rval, pval, stderr = linregress(x, sc)

        # Normalise la pente en unités de σ_y / σ_x pour faciliter la comparaison
        slope_norm = slope * (x.std() / (sc.std() + 1e-12))

        ax.set_facecolor("#f5f5f5")
        ax.scatter(x, sc, color=color, s=55, edgecolors="black",
                   linewidths=0.6, zorder=4, alpha=0.85)

        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color=color, lw=2.2, zorder=5, ls="-")

        # Bande de confiance ±1σ autour de la droite
        n_pts  = len(x)
        se_fit = stderr * np.sqrt(
            1 / n_pts + (x_line - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
        )
        ax.fill_between(
            x_line,
            slope * x_line + intercept - se_fit,
            slope * x_line + intercept + se_fit,
            color=color, alpha=0.15, zorder=3,
        )

        # Encadré de stats : haut-gauche pour CMMD, bas-gauche pour GMMD
        sign_ok = "✓" if slope < 0 else "✗"
        is_cmmd = r["backbone"] == "CMMD"
        tx, ty  = (0.03, 0.97) if is_cmmd else (0.03, 0.06)
        va_txt  = "top"        if is_cmmd else "bottom"
        ax.text(
            tx, ty,
            f"slope = {slope:.3e}\n"
            f"R² = {rval**2:.3f}   p = {pval:.2e}\n"
            f"direction : {sign_ok}",
            transform=ax.transAxes,
            ha="left", va=va_txt, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9),
        )

        sp = r["spearman"]
        ax.set_title(f"{r['label']}  (ρ={sp:+.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean group MOS", fontsize=11)
        ax.set_ylabel("MMD²", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.25)

    # Masquer les axes inutilisés
    for ax in axes[len(ordered):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Linear Regression  MMD² ~ MOS\nRAISE AI {N_GROUPS}×{GROUP_SIZE}  ·  1 000 COCO anchor",
        fontsize=23, fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    p = out / "05_regression.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 6 — découpage 1D des groupes sur l'axe MOS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_grouping_1d(out: Path):
    """
    Charge les 480 images RAISE triées par MOS et trace :
    - Un point par image sur l'axe MOS
    - Les bandes de couleur alternées indiquant les groupes
    - La médiane MOS de chaque groupe
    """
    mos_dict = {}
    for cp in [RAISE_ROOT / "ratings" / "train.csv",
               RAISE_ROOT / "ratings" / "test.csv"]:
        with open(cp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = float(row["MOS_Rating"])

    all_imgs = {}
    for folder in [RAISE_ROOT / "images" / "train_images",
                   RAISE_ROOT / "images" / "test_images"]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[Path(p).name] = p

    items = sorted(
        [(mos_dict[fname], fname)
         for fname in all_imgs
         if fname.startswith("f") and fname in mos_dict],
        key=lambda x: x[0]
    )
    all_mos = np.array([it[0] for it in items])
    n_total  = N_GROUPS * GROUP_SIZE
    all_mos  = all_mos[:n_total]   # on ne garde que les images utilisées

    group_ids = np.repeat(np.arange(N_GROUPS), GROUP_SIZE)  # 0,0,...,1,1,...
    cmap      = plt.get_cmap("tab20", N_GROUPS)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_facecolor("#f0f0f0")

    # Bandes de couleur alternées (fond)
    for g in range(N_GROUPS):
        lo = all_mos[g * GROUP_SIZE]
        hi = all_mos[min((g + 1) * GROUP_SIZE - 1, n_total - 1)]
        # légèrement étendu pour couvrir l'espace entre groupes
        mid_lo = (all_mos[g * GROUP_SIZE - 1] + lo) / 2 if g > 0 else lo - 0.2
        mid_hi = (hi + all_mos[(g + 1) * GROUP_SIZE]) / 2 \
                 if (g + 1) * GROUP_SIZE < n_total else hi + 0.2
        alpha = 0.18 if g % 2 == 0 else 0.32
        ax.axvspan(mid_lo, mid_hi, color=cmap(g), alpha=alpha, zorder=1)

    # Points : un par image, y = jitter aléatoire pour la lisibilité
    rng    = np.random.default_rng(0)
    jitter = rng.uniform(-0.35, 0.35, size=n_total)
    for g in range(N_GROUPS):
        idx  = np.where(group_ids == g)[0]
        ax.scatter(all_mos[idx], jitter[idx],
                   color=cmap(g), s=18, alpha=0.75,
                   edgecolors="none", zorder=3)

    # Frontières entre groupes (trait vertical fin)
    for g in range(1, N_GROUPS):
        boundary = (all_mos[g * GROUP_SIZE - 1] + all_mos[g * GROUP_SIZE]) / 2
        ax.axvline(boundary, color="#555", lw=0.6, ls="--", alpha=0.6, zorder=2)

    # Label du numéro de groupe au-dessus de chaque bande
    for g in range(N_GROUPS):
        mid_mos = all_mos[g * GROUP_SIZE: (g + 1) * GROUP_SIZE].mean()
        ax.text(mid_mos, 0.48, str(g),
                ha="center", va="bottom", fontsize=15,
                color=cmap(g), fontweight="bold", zorder=4)

    ax.set_xlabel("MOS (increasing quality →)", fontsize=22)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_yticks([])
    ax.set_ylim(-0.55, 0.7)
    ax.set_xlim(all_mos.min() - 0.5, all_mos.max() + 0.5)
    ax.set_title(
        f"Split of {n_total} RAISE AI images into {N_GROUPS} groups of {GROUP_SIZE}  "
        f"(sorted by ascending MOS)\n"
        f"Each dot = 1 image · MOS ∈ [{all_mos.min():.1f}, {all_mos.max():.1f}]",
        fontsize=22, fontweight="bold",
        pad=12,
    )
    ax.grid(True, axis="x", alpha=0.3, zorder=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_GROUPS - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.025, pad=0.22, aspect=50)
    cbar.set_label("Group index (0 = lowest quality)", fontsize=17)
    cbar.set_ticks(np.linspace(0, N_GROUPS - 1, min(N_GROUPS, 13)))
    cbar.set_ticklabels(
        [str(int(round(t))) for t in np.linspace(0, N_GROUPS - 1, min(N_GROUPS, 13))],
        fontsize=15
    )

    plt.tight_layout()
    p = out / "06_grouping_1d.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"CSV : {CSV_FILE.name}")
    rows      = load_csv()
    scores    = load_scores(rows)
    group_mos = load_group_mos()

    fig_scatter(rows, OUT)
    fig_barplot(rows, OUT)
    fig_profiles(rows, scores, OUT)
    fig_table(rows, OUT)
    fig_regression(rows, scores, group_mos, OUT)
    fig_grouping_1d(OUT)

    print("\nRésumé :")
    print(f"  {'Label':<35} {'ρ':>7}  {'Dir':>4}  {'Mono':>7}")
    print("  " + "-" * 60)
    for r in sorted(rows, key=lambda x: x["spearman"]):
        d = "✓" if r["spearman"] < 0 else "✗"
        print(f"  {r['label']:<35} {r['spearman']:>+7.3f}  {d:>4}  {r['mono_strict']:>7.1%}")
