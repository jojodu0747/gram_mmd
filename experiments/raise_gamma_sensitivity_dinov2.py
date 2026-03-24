"""
raise_gamma_sensitivity_dinov2.py
==================================
Sensibilité au gamma pour DINOv2 ViT-B/14 L5 sur RAISE.

Protocole :
  - 24 groupes × 20 images (= 480), triées par MOS_Rating croissant
  - Ancrage : 1000 images COCO (seed=42)
  - Gammas testés (tous dérivés de l'ancrage) :
      • γ_median   = 1 / (2 · median(d²))
      • 10×median
      • 50×median

Usage :
    .venv/bin/python experiments/raise_gamma_sensitivity_dinov2.py
"""

import csv
import gc
import glob as glob_module
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, pearsonr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

RAISE_ROOT          = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_TRAIN_IMAGES  = RAISE_ROOT / "images" / "train_images"
RAISE_TEST_IMAGES   = RAISE_ROOT / "images" / "test_images"
RAISE_TRAIN_RATINGS = RAISE_ROOT / "ratings" / "train.csv"
RAISE_TEST_RATINGS  = RAISE_ROOT / "ratings" / "test.csv"
COCO_DIR            = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR             = BASE_DIR / "results" / "raise_gamma_sensitivity_dinov2"

N_ANCHOR    = 1000
ANCHOR_SEED = 42
N_GROUPS    = 24
GROUP_SIZE  = 20   # 24 × 20 = 480

BACKBONE    = "dinov2_vitb14"
LAYER       = 5
BATCH_SIZE  = 8


# ═══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════

def load_coco_anchor() -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if not all_imgs:
        raise FileNotFoundError(f"Aucune image dans {COCO_DIR}")
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = sorted(rng.choice(len(all_imgs), size=N_ANCHOR, replace=False).tolist())
    paths = [all_imgs[i] for i in idx]
    logger.info(f"Ancrage COCO : {len(paths)} images (seed={ANCHOR_SEED})")
    return paths


def load_mos_dict() -> dict:
    mos_dict = {}
    for csv_path in [RAISE_TRAIN_RATINGS, RAISE_TEST_RATINGS]:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = float(row["MOS_Rating"])
    return mos_dict


def load_raise_fake(mos_dict: dict) -> list:
    """480 images IA (f*.png), triées par MOS_Rating croissant."""
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
    logger.info(f"Images IA RAISE : {len(items)}  MOS [{items[0][1]:.2f} … {items[-1][1]:.2f}]")
    return items


def make_groups(items: list) -> list:
    assert len(items) >= N_GROUPS * GROUP_SIZE, \
        f"Besoin de {N_GROUPS * GROUP_SIZE} images, trouvé {len(items)}"
    groups = []
    for i in range(N_GROUPS):
        g = items[i * GROUP_SIZE: (i + 1) * GROUP_SIZE]
        groups.append({
            "paths":   [x[0] for x in g],
            "mos":     float(np.mean([x[1] for x in g])),
            "mos_min": g[0][1],
            "mos_max": g[-1][1],
        })
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  MMD²
# ═══════════════════════════════════════════════════════════════════════

def sq_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True)
    return np.maximum(sq_A + sq_B.T - 2.0 * (A @ B.T), 0.0)


def compute_term1_rr(D_rr: np.ndarray, gamma: float) -> float:
    n = D_rr.shape[0]
    K_rr = np.exp(-gamma * D_rr)
    return float((np.sum(K_rr) - np.trace(K_rr)) / (n * (n - 1)))


def mmd2_unbiased(anchor_feat: np.ndarray, group_feat: np.ndarray,
                  gamma: float, term1_rr: float) -> float:
    n_r = anchor_feat.shape[0]
    n_g = group_feat.shape[0]
    D_rg = sq_dist_matrix(anchor_feat, group_feat)
    D_gg = sq_dist_matrix(group_feat, group_feat)
    K_rg = np.exp(-gamma * D_rg)
    K_gg = np.exp(-gamma * D_gg)
    term2 = (np.sum(K_gg) - np.trace(K_gg)) / (n_g * (n_g - 1)) if n_g > 1 else 1.0
    term3 = 2.0 * np.sum(K_rg) / (n_r * n_g)
    return float(term1_rr + term2 - term3)


# ═══════════════════════════════════════════════════════════════════════
#  GAMMA HEURISTIQUES
# ═══════════════════════════════════════════════════════════════════════

def compute_anchor_gammas(D_rr: np.ndarray) -> dict:
    triu = D_rr[np.triu_indices(D_rr.shape[0], k=1)]
    median_d2 = float(np.median(triu))

    gamma_median = 1.0 / (2.0 * median_d2 + 1e-8)

    gammas = {
        "median":     gamma_median,
        "10x_median": 10.0  * gamma_median,
        "50x_median": 50.0  * gamma_median,
    }

    logger.info(f"  median(d²) = {median_d2:.4e}  → γ_median  = {gamma_median:.4e}")
    for tag, g in gammas.items():
        logger.info(f"    {tag:15s}: γ = {g:.4e}")
    return gammas


# ═══════════════════════════════════════════════════════════════════════
#  STATISTIQUES
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(scores: list, groups: list, label: str) -> dict:
    s   = np.array(scores)
    mos = np.array([g["mos"] for g in groups])
    rho, p_rho = spearmanr(s, mos)
    r,   p_r   = pearsonr(s, mos)
    logger.info(
        f"  {label:50s}  ρ={rho:+.4f}  r={r:+.4f}  "
        f"p_ρ={p_rho:.3e}  p_r={p_r:.3e}"
    )
    return {
        "label":        label,
        "n_groups":     len(scores),
        "spearman_rho": float(rho),
        "neg_rho":      float(-rho),
        "p_spearman":   float(p_rho),
        "pearson_r":    float(r),
        "p_pearson":    float(p_r),
    }


# ═══════════════════════════════════════════════════════════════════════
#  SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════

DETAIL_FIELDS  = ["label", "gamma_tag", "gamma_val", "group_idx",
                  "mos_mean", "mos_min", "mos_max", "mmd_score"]
SUMMARY_FIELDS = ["label", "gamma_tag", "gamma_val", "n_groups",
                  "spearman_rho", "neg_rho", "p_spearman",
                  "pearson_r", "p_pearson"]


def save_csv(rows: list, path: Path, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  Sauvegardé : {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 65)
    logger.info(f"RAISE Gamma Sensitivity — DINOv2 ViT-B/14 L{LAYER}")
    logger.info(f"  Groupes   : {N_GROUPS} × {GROUP_SIZE} = {N_GROUPS * GROUP_SIZE} images")
    logger.info(f"  Ancrage   : {N_ANCHOR} COCO (seed={ANCHOR_SEED})")
    logger.info(f"  Gammas    : median, 10×median, 50×median")
    logger.info(f"  Device    : {device}")
    logger.info("=" * 65)

    anchor_paths = load_coco_anchor()
    mos_dict     = load_mos_dict()
    items        = load_raise_fake(mos_dict)
    groups       = make_groups(items)

    # ── Extraction features ────────────────────────────────────────────
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

    logger.info(f"\nExtraction ancrage ({N_ANCHOR} images)...")
    anchor_feat = extractor.extract(anchor_paths, fit_transform=True, batch_size=BATCH_SIZE)
    logger.info(f"  anchor_feat : {anchor_feat.shape}")

    logger.info(f"\nExtraction RAISE ({len(items)} images)...")
    all_paths = [item[0] for item in items]
    raise_feat = extractor.extract(
        [Image.open(p).convert("RGB") for p in all_paths],
        fit_transform=False, batch_size=BATCH_SIZE
    )
    logger.info(f"  raise_feat : {raise_feat.shape}")

    # ── Calcul gammas heuristiques depuis l'ancrage ────────────────────
    logger.info("\nCalcul des gammas heuristiques (ancrage)...")
    D_rr_anchor = sq_dist_matrix(anchor_feat, anchor_feat)
    gammas = compute_anchor_gammas(D_rr_anchor)

    extractor.cleanup()
    gc.collect()
    torch.cuda.empty_cache()

    # ── Évaluation pour chaque gamma ──────────────────────────────────
    detail_rows  = []
    summary_rows = []

    for gamma_tag, gamma_val in gammas.items():
        logger.info(f"\n── γ = {gamma_val:.4e}  ({gamma_tag}) ──")
        label = f"DINOv2_vitb14_L{LAYER}_{gamma_tag}"

        term1_rr = compute_term1_rr(D_rr_anchor, gamma_val)
        logger.info(f"  term1_rr = {term1_rr:.6f}")

        scores = []
        for i, grp in enumerate(groups):
            start = i * GROUP_SIZE
            end   = start + GROUP_SIZE
            gf    = raise_feat[start:end]
            score = mmd2_unbiased(anchor_feat, gf, gamma_val, term1_rr)
            scores.append(score)

        stats = compute_stats(scores, groups, label)
        stats["gamma_tag"] = gamma_tag
        stats["gamma_val"] = f"{gamma_val:.4e}"

        for i, (score, grp) in enumerate(zip(scores, groups)):
            detail_rows.append({
                "label":     label,
                "gamma_tag": gamma_tag,
                "gamma_val": f"{gamma_val:.4e}",
                "group_idx": i + 1,
                "mos_mean":  grp["mos"],
                "mos_min":   grp["mos_min"],
                "mos_max":   grp["mos_max"],
                "mmd_score": score,
            })
        summary_rows.append(stats)

    # ── Résumé final ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info(f"RÉSUMÉ FINAL  [{N_GROUPS} groupes × {GROUP_SIZE} images]")
    logger.info(f"{'Gamma':15s}  {'γ val':>12s}  {'ρ':>8s}  {'−ρ':>8s}  {'p_ρ':>10s}")
    logger.info("-" * 65)
    for r in sorted(summary_rows, key=lambda x: -x["neg_rho"]):
        sig = " ✱" if r["p_spearman"] < 0.05 else ""
        logger.info(
            f"  {r['gamma_tag']:15s}  {r['gamma_val']:>12s}  "
            f"{r['spearman_rho']:+8.4f}  {r['neg_rho']:+8.4f}  "
            f"{r['p_spearman']:.3e}{sig}"
        )
    logger.info("=" * 65)

    # ── Sauvegarde CSV ────────────────────────────────────────────────
    tag = f"dinov2_vitb14_L{LAYER}_{N_GROUPS}x{GROUP_SIZE}"
    det_path = OUT_DIR / f"gamma_sensitivity_{tag}_detailed_{ts}.csv"
    sum_path = OUT_DIR / f"gamma_sensitivity_{tag}_summary_{ts}.csv"
    save_csv(detail_rows,  det_path, DETAIL_FIELDS)
    save_csv(summary_rows, sum_path, SUMMARY_FIELDS)
    logger.info(f"\nRésultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
