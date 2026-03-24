"""
plot_kadid_mos_global.py
========================
Visualisation des résultats de kadid_mos_quintile.py (protocole global).

Produit :
  01_barplot_rho.png          — barplot Spearman ρ + strict mono
  02_barplot_strict_mono.png  — barplot strict mono + ρ (info)
  03_barplot_kendall.png      — barplot Kendall τ + strict mono (info)
  04_mmd2_curves.png          — courbes MMD²(groupe) par config
  05_combined.png             — panels 01+02+03 empilés (Figure principale)

Usage :
    .venv/bin/python experiments/plot_kadid_mos_global.py
    .venv/bin/python experiments/plot_kadid_mos_global.py --csv results/kadid_mos_global/global_results_XYZ.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = Path(__file__).resolve().parent.parent
RES_DIR  = BASE_DIR / "results" / "kadid_mos_global"

DATASET_LABEL = "KADID-10k  ·  405 groupes triés par DMOS  ·  anchor = 81 refs"

BACKBONE_COLORS = {
    "sd_vae":        "#2196F3",   # bleu
    "dc_ae":         "#E91E63",   # rose
    "dinov2_vitb14": "#4CAF50",   # vert
    "cmmd":          "#607D8B",   # gris bleuté
}

BACKBONE_LABELS = {
    "sd_vae":        "SD-VAE",
    "dc_ae":         "DC-AE",
    "dinov2_vitb14": "DINOv2",
    "cmmd":          "CMMD",
}


# ===========================================================================
# Chargement
# ===========================================================================

def load_results(csv_path: Path) -> list:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["layer"]            = int(row["layer"])
            row["gamma"]            = float(row["gamma"])
            row["spearman_rho"]     = float(row["spearman_rho"])
            row["strict_mono_rate"] = float(row["strict_mono_rate"])
            row["kendall_tau_mean"] = float(row["kendall_tau_mean"])
            row["n_groups"]         = int(row["n_groups"])
            rows.append(row)
    return rows


def find_latest_csv() -> Path:
    csvs = sorted(RES_DIR.glob("global_results_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Aucun CSV dans {RES_DIR}")
    return csvs[-1]


def _label(r: dict) -> str:
    bb  = BACKBONE_LABELS.get(r["backbone"], r["backbone"])
    lay = r["layer_name"].split(".")[-1] if "." in r["layer_name"] else r["layer_name"]
    return f"{bb}  L{r['layer']}  γ={r['gamma']:.0e}  [{lay}]"


def _color(r: dict) -> str:
    return BACKBONE_COLORS.get(r["backbone"], "#555")


# ===========================================================================
# Panel 1 — barplot ρ (+ strict mono en annotation)
# ===========================================================================

def plot_barplot_rho(rows: list, out_dir: Path) -> Path:
    sorted_rows = sorted(rows, key=lambda r: r["spearman_rho"])
    labels  = [_label(r) for r in sorted_rows]
    rhos    = [r["spearman_rho"]     for r in sorted_rows]
    monos   = [r["strict_mono_rate"] for r in sorted_rows]
    colors  = [_color(r)             for r in sorted_rows]
    n = len(sorted_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, n * 0.55 + 2)))
    fig.suptitle(
        f"Ranked by Spearman ρ — all {n} configurations\n{DATASET_LABEL}",
        fontsize=13, fontweight="bold",
    )

    y = np.arange(n)

    for ax, vals, xlabel, vline, fmt in [
        (ax1, rhos,  "Spearman ρ",              0.0, "+.4f"),
        (ax2, monos, "Strict monotonicity rate", 0.5, ".3f"),
    ]:
        bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white", height=0.6)
        for bar, v in zip(bars, vals):
            x_txt = v + 0.005 if v >= 0 else v - 0.005
            ha = "left" if v >= 0 else "right"
            ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                    format(v, fmt), va="center", ha=ha, fontsize=9)
        ax.axvline(vline, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is ax1 else [""] * n, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        vmax = max(abs(v) for v in vals)
        ax.set_xlim(-vmax * 1.25, vmax * 1.25)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_facecolor("#f9f9f9")

    plt.tight_layout()
    out_path = out_dir / "01_barplot_rho.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ===========================================================================
# Panel 2 — barplot strict mono
# ===========================================================================

def plot_barplot_strict_mono(rows: list, out_dir: Path) -> Path:
    sorted_rows = sorted(rows, key=lambda r: r["strict_mono_rate"])
    labels  = [_label(r) for r in sorted_rows]
    monos   = [r["strict_mono_rate"] for r in sorted_rows]
    rhos    = [r["spearman_rho"]     for r in sorted_rows]
    colors  = [_color(r)             for r in sorted_rows]
    n = len(sorted_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, n * 0.55 + 2)))
    fig.suptitle(
        f"Ranked by STRICT monotonicity — all {n} configurations\n"
        "strict_mono_rate = fraction of consecutive pairs where MMD² increases\n"
        f"{DATASET_LABEL}",
        fontsize=12, fontweight="bold",
    )

    y = np.arange(n)

    for ax, vals, xlabel, vline, fmt in [
        (ax1, monos, "Strict mono rate",  0.5,  ".3f"),
        (ax2, rhos,  "Spearman ρ (info)", 0.0,  "+.4f"),
    ]:
        bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white", height=0.6)
        for bar, v in zip(bars, vals):
            x_txt = v + 0.005 if v >= 0 else v - 0.005
            ha = "left" if v >= 0 else "right"
            ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                    format(v, fmt), va="center", ha=ha, fontsize=9)
        ax.axvline(vline, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is ax1 else [""] * n, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        vmax = max(abs(v) for v in vals)
        margin = vmax * 0.25
        ax.set_xlim(min(vals) - margin, max(vals) + margin)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_facecolor("#f9f9f9")

    plt.tight_layout()
    out_path = out_dir / "02_barplot_strict_mono.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ===========================================================================
# Panel 3 — barplot Kendall τ
# ===========================================================================

def plot_barplot_kendall(rows: list, out_dir: Path) -> Path:
    sorted_rows = sorted(rows, key=lambda r: r["kendall_tau_mean"])
    labels  = [_label(r) for r in sorted_rows]
    taus    = [r["kendall_tau_mean"]  for r in sorted_rows]
    monos   = [r["strict_mono_rate"]  for r in sorted_rows]
    colors  = [_color(r)              for r in sorted_rows]
    n = len(sorted_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, n * 0.55 + 2)))
    fig.suptitle(
        f"Ranked by PARTIAL monotonicity (Kendall τ) — all {n} configurations\n"
        "τ = fraction of concordant rank pairs on the 405-group sequence\n"
        f"{DATASET_LABEL}",
        fontsize=12, fontweight="bold",
    )

    y = np.arange(n)

    for ax, vals, xlabel, vline, fmt in [
        (ax1, taus,  "Kendall τ",             0.0, "+.4f"),
        (ax2, monos, "Strict mono rate (info)", 0.5, ".3f"),
    ]:
        bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white", height=0.6)
        for bar, v in zip(bars, vals):
            x_txt = v + 0.005 if v >= 0 else v - 0.005
            ha = "left" if v >= 0 else "right"
            ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                    format(v, fmt), va="center", ha=ha, fontsize=9)
        ax.axvline(vline, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is ax1 else [""] * n, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        vmax = max(abs(v) for v in vals)
        ax.set_xlim(-vmax * 1.25, vmax * 1.25)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_facecolor("#f9f9f9")

    plt.tight_layout()
    out_path = out_dir / "03_barplot_kendall.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ===========================================================================
# Panel 4 — courbes MMD²(groupe 1..405)
# ===========================================================================

def plot_mmd2_curves(rows: list, res_dir: Path, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f9f9f9")

    # Find .npy files
    npy_files = sorted(res_dir.glob("mmd2_vector_*.npy"))
    loaded = {}
    for p in npy_files:
        # Match filename to row by backbone + layer
        stem = p.stem  # e.g. mmd2_vector_sd_vae_L11_g1e-04_20260311_102415
        for r in rows:
            bb = r["backbone"]
            l  = r["layer"]
            if f"_{bb}_L{l:02d}_" in stem:
                loaded[r["backbone"]] = np.load(str(p))
                break

    x = np.arange(1, 406)

    for r in sorted(rows, key=lambda r: r["spearman_rho"], reverse=True):
        bb  = r["backbone"]
        vec = loaded.get(bb)
        if vec is None:
            continue
        color = _color(r)
        lbl   = f"{_label(r)}  (ρ={r['spearman_rho']:+.3f})"
        # Normalize to [0,1] for visual comparison
        v_min, v_max = vec.min(), vec.max()
        vec_norm = (vec - v_min) / (v_max - v_min + 1e-12)
        ax.plot(x, vec_norm, color=color, linewidth=1.2, alpha=0.85, label=lbl)

    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Group index (1 = lowest DMOS, 405 = highest DMOS)", fontsize=11)
    ax.set_ylabel("MMD² (normalized to [0,1])", fontsize=11)
    ax.set_title(
        "MMD² as a function of quality degradation (405 groups, global DMOS sort)\n"
        f"{DATASET_LABEL}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = out_dir / "04_mmd2_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure principale — panels 01+02+03 empilés verticalement
# ===========================================================================

def plot_combined(out_dir: Path):
    paths = [
        out_dir / "01_barplot_rho.png",
        out_dir / "02_barplot_strict_mono.png",
        out_dir / "03_barplot_kendall.png",
    ]
    for p in paths:
        if not p.exists():
            print(f"  [skip] {p.name} not found")
            return

    imgs       = [mpimg.imread(str(p)) for p in paths]
    heights_px = [img.shape[0] for img in imgs]
    width_px   = max(img.shape[1] for img in imgs)
    total_h_in = sum(h / 150 for h in heights_px)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(width_px / 150, total_h_in),
        gridspec_kw={"height_ratios": heights_px},
    )
    for ax, img in zip(axes, imgs):
        ax.imshow(img, aspect="auto")
        ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)

    out_path = out_dir / "05_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv()
    print(f"CSV : {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dossier sortie : {out_dir}\n")

    rows = load_results(csv_path)
    print(f"  {len(rows)} configurations chargées\n")

    print("--- 1. Barplot Spearman ρ ---")
    plot_barplot_rho(rows, out_dir)

    print("--- 2. Barplot strict monotonicity ---")
    plot_barplot_strict_mono(rows, out_dir)

    print("--- 3. Barplot Kendall τ ---")
    plot_barplot_kendall(rows, out_dir)

    print("--- 4. Courbes MMD²(groupe) ---")
    plot_mmd2_curves(rows, csv_path.parent, out_dir)

    print("--- 5. Figure combinée (01+02+03) ---")
    plot_combined(out_dir)

    print(f"\nFigures dans : {out_dir}")


if __name__ == "__main__":
    main()
