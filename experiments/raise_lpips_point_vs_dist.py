"""
raise_lpips_point_vs_dist.py
============================
Score point-vs-distribution : chaque image RAISE (480 images IA)
est scorée individuellement contre la distribution des 1000 COCO anchors.

  score(q) = 1 + mean_K_rr - 2 * mean_r( K(r, q) )
  (estimateur biaisé mono-point, identique au sweep gamma_sweep_sdvae.py)

Config :
  • Backbone  : lpips_vgg,  layer = 5
  • Gamma     : 10 × médiane heuristique  (1 / (2 · median(d²)))
  • Ancrage   : 1000 images COCO (seed=42)
  • Queries   : 480 images RAISE IA  (toutes, pas de groupes)
  • CMMD      : non

Sorties dans results/raise_lpips_point/ :
  raise_lpips_point_scores_{ts}.csv   — 480 lignes (image, MOS, score)
  raise_lpips_point_summary_{ts}.csv  — ρ Spearman, r Pearson
  plots/
    01_scatter_score_vs_mos.png
    02_sorted_scores.png

Usage :
    .venv/bin/python experiments/raise_lpips_point_vs_dist.py
"""

import csv
import glob as glob_module
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr, pearsonr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Chemins ──────────────────────────────────────────────────────────────────
RAISE_ROOT          = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_TRAIN_IMAGES  = RAISE_ROOT / "images" / "train_images"
RAISE_TEST_IMAGES   = RAISE_ROOT / "images" / "test_images"
RAISE_TRAIN_RATINGS = RAISE_ROOT / "ratings" / "train.csv"
RAISE_TEST_RATINGS  = RAISE_ROOT / "ratings" / "test.csv"
COCO_DIR            = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR             = BASE_DIR / "results" / "raise_lpips_point"

# ─── Config ───────────────────────────────────────────────────────────────────
BACKBONE    = "lpips_vgg"
LAYER       = 5
GAMMA_MULT  = 10.0        # 10× médiane heuristique
N_ANCHOR    = 1000
ANCHOR_SEED = 42
BATCH_SIZE  = 32


# ═══════════════════════════════════════════════════════════════════════════════
# Chargement données
# ═══════════════════════════════════════════════════════════════════════════════

def load_coco_anchor() -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = sorted(rng.choice(len(all_imgs), size=N_ANCHOR, replace=False).tolist())
    paths = [all_imgs[i] for i in idx]
    logger.info(f"Ancrage COCO : {len(paths)} images (seed={ANCHOR_SEED})")
    return paths


def load_mos_dict() -> dict:
    mos = {}
    for csv_path in [RAISE_TRAIN_RATINGS, RAISE_TEST_RATINGS]:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos[row["filename"]] = float(row["MOS_Rating"])
    return mos


def load_raise_images(mos_dict: dict) -> list:
    """480 images IA (f*.png/jpg), triées par MOS croissant."""
    all_imgs = {}
    for folder in [RAISE_TRAIN_IMAGES, RAISE_TEST_IMAGES]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[os.path.basename(p)] = p
    items = [
        (path, mos_dict[fname])
        for fname, path in all_imgs.items()
        if fname.startswith("f") and fname in mos_dict
    ]
    items.sort(key=lambda x: x[1])
    logger.info(f"RAISE images IA : {len(items)}  "
                f"MOS [{items[0][1]:.2f} … {items[-1][1]:.2f}]")
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Distances et gamma
# ═══════════════════════════════════════════════════════════════════════════════

def sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    return np.maximum(
        (A * A).sum(1, keepdims=True) + (B * B).sum(1, keepdims=True).T
        - 2.0 * (A @ B.T), 0.0
    )


def median_gamma(feat: np.ndarray) -> float:
    """1 / (2 · median(d²)) sur le triangle supérieur strict."""
    D = sq_dist(feat, feat)
    triu = D[np.triu_indices(D.shape[0], k=1)]
    return float(1.0 / (2.0 * float(np.median(triu)) + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════════
# Score point-vs-distribution
# ═══════════════════════════════════════════════════════════════════════════════

def score_points(anchor_feat: np.ndarray, query_feat: np.ndarray,
                 gamma: float) -> np.ndarray:
    """
    Retourne scores (N_q,) :
      score(q) = 1 + mean_K_rr - 2 * mean_r( K(r, q) )
    """
    n_r = anchor_feat.shape[0]
    D_rr = sq_dist(anchor_feat, anchor_feat)
    K_rr_mean = float(np.exp(-gamma * D_rr).mean())

    D_rq = sq_dist(anchor_feat, query_feat)    # (n_r, n_q)
    K_rq_col = np.exp(-gamma * D_rq).mean(axis=0)  # (n_q,)

    return 1.0 + K_rr_mean - 2.0 * K_rq_col


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scatter(scores: np.ndarray, mos: np.ndarray,
                 rho: float, r_pearson: float,
                 gamma: float, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#f9f9f9")
    ax.scatter(mos, scores, s=18, alpha=0.55, color="#9C27B0",
               edgecolors="none", label="RAISE IA (480 imgs)")

    # régression linéaire
    m, b = np.polyfit(mos, scores, 1)
    x_line = np.linspace(mos.min(), mos.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#E91E63", linewidth=1.5,
            label=f"Linear fit  (r={r_pearson:+.3f})")

    ax.set_xlabel("MOS Rating", fontsize=12)
    ax.set_ylabel("MMD² score", fontsize=12)
    ax.set_title(
        f"LPIPS-VGG L{LAYER} — point-vs-distribution\n"
        f"γ = {GAMMA_MULT:.0f}× median  ({gamma:.3e})  ·  "
        f"Spearman ρ = {rho:+.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / "01_scatter_score_vs_mos.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


def plot_sorted(scores: np.ndarray, mos: np.ndarray, out_dir: Path):
    """Images triées par MOS (x) et score MMD² (y)."""
    idx_mos = np.argsort(mos)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_facecolor("#f9f9f9")

    x = np.arange(len(mos))
    color_score = "#9C27B0"
    color_mos   = "#FF9800"

    ax1.plot(x, scores[idx_mos], color=color_score, linewidth=1.2, label="MMD² score")
    ax1.set_xlabel("Images triées par MOS croissant", fontsize=11)
    ax1.set_ylabel("MMD² score", color=color_score, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_score)

    ax2 = ax1.twinx()
    ax2.plot(x, mos[idx_mos], color=color_mos, linewidth=1.0,
             linestyle="--", alpha=0.7, label="MOS")
    ax2.set_ylabel("MOS Rating", color=color_mos, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_mos)

    ax1.set_title(
        f"LPIPS-VGG L{LAYER} — MMD² et MOS triés par MOS  "
        f"(480 images RAISE IA)",
        fontsize=11, fontweight="bold",
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    out = out_dir / "02_sorted_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"RAISE Point-vs-Distribution  —  {BACKBONE} L{LAYER}")
    logger.info(f"  Gamma      : {GAMMA_MULT:.0f}× median heuristic")
    logger.info(f"  N anchor   : {N_ANCHOR} COCO (seed={ANCHOR_SEED})")
    logger.info(f"  N queries  : 480 RAISE IA images")
    logger.info("=" * 60)

    anchor_paths = load_coco_anchor()
    mos_dict     = load_mos_dict()
    items        = load_raise_images(mos_dict)   # 480 images triées par MOS

    all_paths = [item[0] for item in items]
    all_mos   = np.array([item[1] for item in items])

    # ── Feature extraction ──────────────────────────────────────────────
    extractor = FeatureExtractor(
        backbone=BACKBONE,
        layer=LAYER,
        transform_config={
            "name":           "gram_spatial",
            "use_gram":       True,
            "use_pca":        False,
            "gram_patches":   False,
            "gram_averaging": "spatial",
        },
    )

    logger.info(f"\nExtraction ancrage ({N_ANCHOR} images COCO)…")
    anchor_feat = extractor.extract(anchor_paths, fit_transform=True,
                                    batch_size=BATCH_SIZE)
    logger.info(f"  anchor_feat : {anchor_feat.shape}")

    logger.info(f"Extraction queries (480 images RAISE)…")
    query_feat = extractor.extract(
        [Image.open(p).convert("RGB") for p in all_paths],
        fit_transform=False, batch_size=BATCH_SIZE
    )
    logger.info(f"  query_feat  : {query_feat.shape}")

    extractor.cleanup()
    del extractor
    torch.cuda.empty_cache()

    # ── Gamma ───────────────────────────────────────────────────────────
    g_med = median_gamma(anchor_feat)
    gamma = GAMMA_MULT * g_med
    logger.info(f"\nGamma médiane = {g_med:.4e}")
    logger.info(f"Gamma utilisé = {GAMMA_MULT:.0f}× médiane = {gamma:.4e}")

    # ── Scores ──────────────────────────────────────────────────────────
    logger.info("\nCalcul scores point-vs-distribution…")
    scores = score_points(anchor_feat, query_feat, gamma)
    logger.info(f"  scores : min={scores.min():.5f}  max={scores.max():.5f}  "
                f"mean={scores.mean():.5f}")

    # ── Statistiques ────────────────────────────────────────────────────
    rho, p_rho = spearmanr(scores, all_mos)
    r,   p_r   = pearsonr(scores, all_mos)
    logger.info(f"\n  Spearman ρ = {rho:+.4f}  (p={p_rho:.3e})")
    logger.info(f"  Pearson  r = {r:+.4f}  (p={p_r:.3e})")

    # ── Sauvegarde CSV ──────────────────────────────────────────────────
    detail_rows = [
        {
            "filename": os.path.basename(all_paths[i]),
            "mos":      float(all_mos[i]),
            "mmd_score": float(scores[i]),
        }
        for i in range(len(scores))
    ]
    detail_path = OUT_DIR / f"raise_lpips_point_scores_{ts}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "mos", "mmd_score"])
        w.writeheader()
        w.writerows(detail_rows)
    logger.info(f"  Saved: {detail_path}")

    summary_path = OUT_DIR / f"raise_lpips_point_summary_{ts}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "backbone", "layer", "gamma_mult", "gamma",
            "n_anchor", "n_queries",
            "spearman_rho", "p_spearman", "pearson_r", "p_pearson",
        ])
        w.writeheader()
        w.writerow({
            "backbone":     BACKBONE,
            "layer":        LAYER,
            "gamma_mult":   GAMMA_MULT,
            "gamma":        gamma,
            "n_anchor":     N_ANCHOR,
            "n_queries":    len(scores),
            "spearman_rho": float(rho),
            "p_spearman":   float(p_rho),
            "pearson_r":    float(r),
            "p_pearson":    float(p_r),
        })
    logger.info(f"  Saved: {summary_path}")

    # ── Figures ─────────────────────────────────────────────────────────
    logger.info("\nFigures…")
    plot_scatter(scores, all_mos, rho, r, gamma, plots_dir)
    plot_sorted(scores, all_mos, plots_dir)

    logger.info("\n" + "=" * 60)
    logger.info(f"  {BACKBONE} L{LAYER}  γ={GAMMA_MULT:.0f}×med={gamma:.3e}")
    logger.info(f"  Spearman ρ = {rho:+.4f}  (p={p_rho:.3e})")
    logger.info(f"  Pearson  r = {r:+.4f}  (p={p_r:.3e})")
    logger.info(f"Résultats dans : {OUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
