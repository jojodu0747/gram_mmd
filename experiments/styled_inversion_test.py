"""
styled_inversion_test.py
========================
Test d'inversion avec images KITTI stylisées (histogram matching L*a*b*).

  Anchor  : 1000 images KITTI originales (dataset/kitti/images/)
  Query A : 1000 images KITTI stylisées  (dataset/kitti_styled/)
            → même sémantique, texture radicalement différente
  Query B : 1000 images COCO

Hypothèse :
  - CMMD (CLIP)       : KITTI stylisé ≈ KITTI original (CLIP invariant au style)
                        → faible distance → INVERSION (styled < COCO)
  - Gram-MMD (SD-VAE) : texture très différente → grande distance
                        → PAS d'inversion (styled > COCO)

Usage:
    .venv/bin/python experiments/styled_inversion_test.py
"""

import gc
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
KITTI_DIR   = BASE_DIR / "dataset" / "kitti" / "images"
STYLED_DIR  = BASE_DIR / "dataset" / "kitti_styled"
COCO_DIR    = BASE_DIR / "dataset" / "coco_5000"

N_ANCHOR    = 1000
N_QUERY     = 1000
SEED        = 42
BATCH_SIZE  = 16

BACKBONE = "sd_vae"
LAYER    = 11
GAMMA_MULTIPLIERS = {"γ/10": 0.1, "γ_med": 1.0, "10×γ": 10.0, "100×γ": 100.0}
CMMD_GAMMA = 0.005


# ── Utils ────────────────────────────────────────────────────

def sample_paths(paths, n, seed):
    rng = np.random.default_rng(seed)
    paths = sorted(paths)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(idx)]


def median_gamma(feats, max_samples=200, max_dims=512):
    rng = np.random.default_rng(0)
    n, d = feats.shape
    idx = rng.choice(n, size=min(n, max_samples), replace=False)
    sample = feats[idx]
    if d > max_dims:
        dim_idx = rng.choice(d, size=max_dims, replace=False)
        sample = sample[:, dim_idx]
    diff = sample[:, None, :] - sample[None, :, :]
    sq = (diff ** 2).sum(-1)
    med = np.median(sq[sq > 0])
    return float(1.0 / (2 * med + 1e-8))


def sq_dists_fast(A, B):
    """||a_i - b_j||² via dot product trick."""
    aa = (A ** 2).sum(1)
    bb = (B ** 2).sum(1)
    return aa[:, None] + bb[None, :] - 2 * (A @ B.T)


def mmd2_unbiased(X, Y, gamma):
    n, m = len(X), len(Y)
    Kxx = np.exp(-gamma * sq_dists_fast(X, X)); np.fill_diagonal(Kxx, 0)
    Kyy = np.exp(-gamma * sq_dists_fast(Y, Y)); np.fill_diagonal(Kyy, 0)
    Kxy = np.exp(-gamma * sq_dists_fast(X, Y))
    return float(Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean())


def get_gram(paths, ext, fit_transform=False):
    return ext.extract([str(p) for p in paths],
                       fit_transform=fit_transform,
                       batch_size=BATCH_SIZE).astype(np.float32)


def get_clip_feats(paths, clip_model, clip_preprocess, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [clip_preprocess(Image.open(str(p)).convert("RGB"))
                 for p in paths[i:i+batch_size]]
        t = torch.stack(batch).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(t).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def print_row(label, vv, cv, width=22):
    r = vv / cv if cv > 1e-12 else float("inf")
    inv = "OUI ★" if r < 1 else "non"
    logger.info(f"  {label:<{width}} {vv:>14.4e}  {cv:>14.4e}  {r:>7.3f}  {inv}")


# ── Main ─────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    # Anchor = KITTI originaux (on exclut les images qui ont été stylisées)
    styled_stems = {p.stem.replace("_styled", "") for p in STYLED_DIR.glob("*.png")}
    all_kitti = [p for p in sorted(KITTI_DIR.glob("*.png"))
                 if p.stem not in styled_stems]
    # Si pas assez, prendre tous
    if len(all_kitti) < N_ANCHOR:
        all_kitti = sorted(KITTI_DIR.glob("*.png"))

    anchor_paths = sample_paths(all_kitti, N_ANCHOR, SEED)
    styled_paths = sample_paths(list(STYLED_DIR.glob("*.png")), N_QUERY, SEED+1)
    coco_paths   = sample_paths(
        list(COCO_DIR.glob("*.jpg")) + list(COCO_DIR.glob("*.png")), N_QUERY, SEED+2)

    logger.info(f"Anchor  : {len(anchor_paths)} images KITTI originales")
    logger.info(f"Query A : {len(styled_paths)} images KITTI stylisées")
    logger.info(f"Query B : {len(coco_paths)} images COCO")

    # ── Gram extractors ──
    transform_cfg = {"name": "gram_spatial", "use_gram": True,
                     "use_pca": False, "gram_patches": False, "gram_averaging": "spatial"}

    ext_std = FeatureExtractor(backbone=BACKBONE, layer=LAYER,
                               transform_config=transform_cfg, device=device)
    ext_raw = FeatureExtractor(backbone=BACKBONE, layer=LAYER,
                               transform_config=transform_cfg, device=device)

    logger.info("\nExtraction Gram anchor (std)...")
    anchor_std = get_gram(anchor_paths, ext_std, fit_transform=True)
    logger.info("Extraction Gram anchor (raw)...")
    anchor_raw = get_gram(anchor_paths, ext_raw, fit_transform=False)

    logger.info("Extraction Gram styled...")
    styled_std = get_gram(styled_paths, ext_std)
    styled_raw = get_gram(styled_paths, ext_raw)

    logger.info("Extraction Gram COCO...")
    coco_std = get_gram(coco_paths, ext_std)
    coco_raw = get_gram(coco_paths, ext_raw)

    g_std = median_gamma(anchor_std)
    g_raw = median_gamma(anchor_raw)

    # ── CLIP ──
    logger.info("\nChargement CLIP...")
    import clip
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()

    logger.info("Extraction CLIP anchor...")
    anchor_clip = get_clip_feats(anchor_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP styled...")
    styled_clip = get_clip_feats(styled_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP COCO...")
    coco_clip = get_clip_feats(coco_paths, clip_model, clip_preprocess, device)

    g_clip = median_gamma(anchor_clip)

    # ── Compute distances ──
    logger.info("\nCalcul des distances...")

    gram_std_styled = {t: mmd2_unbiased(anchor_std, styled_std, g_std*m)
                       for t, m in GAMMA_MULTIPLIERS.items()}
    gram_std_coco   = {t: mmd2_unbiased(anchor_std, coco_std,   g_std*m)
                       for t, m in GAMMA_MULTIPLIERS.items()}
    gram_raw_styled = {t: mmd2_unbiased(anchor_raw, styled_raw, g_raw*m)
                       for t, m in GAMMA_MULTIPLIERS.items()}
    gram_raw_coco   = {t: mmd2_unbiased(anchor_raw, coco_raw,   g_raw*m)
                       for t, m in GAMMA_MULTIPLIERS.items()}

    cmmd_styled_fixed = mmd2_unbiased(anchor_clip, styled_clip, CMMD_GAMMA)
    cmmd_coco_fixed   = mmd2_unbiased(anchor_clip, coco_clip,   CMMD_GAMMA)
    cmmd_styled_med   = mmd2_unbiased(anchor_clip, styled_clip, g_clip)
    cmmd_coco_med     = mmd2_unbiased(anchor_clip, coco_clip,   g_clip)

    # ── Print results ──
    sep = "=" * 75
    logger.info(f"\n{sep}")
    logger.info("STYLED KITTI INVERSION TEST")
    logger.info(f"  Anchor: KITTI orig | Query A: KITTI stylisé | Query B: COCO")
    logger.info(sep)
    logger.info(f"  {'Méthode':<22} {'Styled':>14}  {'COCO':>14}  {'Ratio':>7}  INV?")
    logger.info(f"  {'-'*68}")
    for tag in GAMMA_MULTIPLIERS:
        print_row(f"Gram_std {tag}", gram_std_styled[tag], gram_std_coco[tag])
    for tag in GAMMA_MULTIPLIERS:
        print_row(f"Gram_raw {tag}", gram_raw_styled[tag], gram_raw_coco[tag])
    print_row("CMMD γ=0.005", cmmd_styled_fixed, cmmd_coco_fixed)
    print_row("CMMD γ_med",   cmmd_styled_med,   cmmd_coco_med)
    logger.info(sep)

    # Summary
    cmmd_inv = cmmd_styled_fixed < cmmd_coco_fixed or cmmd_styled_med < cmmd_coco_med
    gram_inv = any(gram_std_styled[t] < gram_std_coco[t] for t in GAMMA_MULTIPLIERS) or \
               any(gram_raw_styled[t] < gram_raw_coco[t] for t in GAMMA_MULTIPLIERS)

    logger.info("")
    if cmmd_inv and not gram_inv:
        logger.info("  ★ SUCCÈS : CMMD inverse, Gram-MMD ne s'inverse PAS ★")
    elif cmmd_inv and gram_inv:
        logger.info("  ~ Les deux inversent (CMMD et Gram-MMD)")
    elif not cmmd_inv and not gram_inv:
        logger.info("  ~ Aucune inversion (ni CMMD ni Gram-MMD)")
    else:
        logger.info("  ~ Gram-MMD inverse mais pas CMMD (inverse de l'objectif)")
    logger.info(sep)


if __name__ == "__main__":
    main()
