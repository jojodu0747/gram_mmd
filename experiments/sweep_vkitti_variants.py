"""
sweep_vkitti_variants.py
========================
Anchor  : 1000 images KITTI (dataset/kitti/images/)
Queries : chaque variant VKITTI2 (clone, fog, rain, morning, sunset, all_mix)
          + 1000 COCO
Métriques : Gram-MMD (std + raw, 4 gammas) + CMMD (γ=0.005, γ_med)

Objectif : trouver un variant où CMMD inverse (VK < COCO) mais pas Gram-MMD.
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
KITTI_DIR  = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR = BASE_DIR / "dataset" / "virtual_kitti"
COCO_DIR   = BASE_DIR / "dataset" / "coco_5000"

N_ANCHOR   = 1000
N_QUERY    = 1000
SEED       = 42
BATCH_SIZE = 16

BACKBONE = "sd_vae"
LAYER    = 11
GAMMA_MULTIPLIERS = {"γ/10": 0.1, "γ_med": 1.0, "10×γ": 10.0, "100×γ": 100.0}

CMMD_GAMMA = 0.005

VKITTI_VARIANTS = {
    "clone":   ["clone_rgb"],
    "fog":     ["fog_rgb"],
    "rain":    ["rain_rgb"],
    "morning": ["morning_rgb"],
    "sunset":  ["sunset_rgb"],
    "all_mix": ["clone_rgb", "fog_rgb", "rain_rgb", "morning_rgb", "sunset_rgb"],
}


# ── Helpers ─────────────────────────────────────────────────

def sample_paths(paths, n, seed):
    rng = np.random.default_rng(seed)
    paths = sorted(paths)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(idx)]


def median_gamma(feats, max_samples=200, max_dims=512):
    """Median heuristic sur un sous-échantillon et sous-espace réduit."""
    rng = np.random.default_rng(0)
    n, d = feats.shape
    idx = rng.choice(n, size=min(n, max_samples), replace=False)
    sample = feats[idx]
    # Réduire la dimension si nécessaire
    if d > max_dims:
        dim_idx = rng.choice(d, size=max_dims, replace=False)
        sample = sample[:, dim_idx]
    diff = sample[:, None, :] - sample[None, :, :]
    sq = (diff ** 2).sum(-1)
    med = np.median(sq[sq > 0])
    return float(1.0 / (2 * med + 1e-8))


def sq_dists_fast(A, B):
    """||a_i - b_j||² via dot product trick. Returns (n, m) array."""
    aa = (A ** 2).sum(1)       # (n,)
    bb = (B ** 2).sum(1)       # (m,)
    ab = A @ B.T               # (n, m)
    return aa[:, None] + bb[None, :] - 2 * ab


def mmd2_unbiased(X, Y, gamma):
    """Unbiased MMD² with RBF kernel using dot-product trick (memory efficient)."""
    n, m = len(X), len(Y)
    sq_xx = sq_dists_fast(X, X)
    np.fill_diagonal(sq_xx, 0)
    Kxx = np.exp(-gamma * sq_xx)
    np.fill_diagonal(Kxx, 0)

    sq_yy = sq_dists_fast(Y, Y)
    np.fill_diagonal(sq_yy, 0)
    Kyy = np.exp(-gamma * sq_yy)
    np.fill_diagonal(Kyy, 0)

    Kxy = np.exp(-gamma * sq_dists_fast(X, Y))
    return float(Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean())


def extract_gram(paths, ext, fit_transform=False):
    return ext.extract([str(p) for p in paths], fit_transform=fit_transform,
                       batch_size=BATCH_SIZE).astype(np.float32)


def get_clip_feats(paths, clip_model, clip_preprocess, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [clip_preprocess(Image.open(str(p)).convert("RGB"))
                 for p in paths[i:i+batch_size]]
        batch_t = torch.stack(batch).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(batch_t).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


# ── Main ─────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Anchor
    anchor_paths = sample_paths(list(KITTI_DIR.glob("*.png")), N_ANCHOR, SEED)
    logger.info(f"Anchor: {len(anchor_paths)} images KITTI")

    # COCO
    coco_paths = sample_paths(
        list(COCO_DIR.glob("*.jpg")) + list(COCO_DIR.glob("*.png")), N_QUERY, SEED+1)
    logger.info(f"COCO: {len(coco_paths)} images")

    # Gram extractors
    ext_std = FeatureExtractor(
        backbone=BACKBONE, layer=LAYER,
        transform_config={"name": "gram_spatial", "use_gram": True,
                          "use_pca": False, "gram_patches": False, "gram_averaging": "spatial"},
        device=device,
    )
    ext_raw = FeatureExtractor(
        backbone=BACKBONE, layer=LAYER,
        transform_config={"name": "gram_spatial", "use_gram": True,
                          "use_pca": False, "gram_patches": False, "gram_averaging": "spatial"},
        device=device,
    )

    logger.info("Extracting anchor Gram (std)...")
    anchor_std = extract_gram(anchor_paths, ext_std, fit_transform=True)
    logger.info("Extracting anchor Gram (raw)...")
    anchor_raw = extract_gram(anchor_paths, ext_raw, fit_transform=False)

    logger.info("Extracting COCO Gram...")
    coco_std = extract_gram(coco_paths, ext_std, fit_transform=False)
    coco_raw = extract_gram(coco_paths, ext_raw, fit_transform=False)

    g_med_std = median_gamma(anchor_std)
    g_med_raw = median_gamma(anchor_raw)

    # COCO Gram MMD
    coco_gram_std = {t: mmd2_unbiased(anchor_std, coco_std, g_med_std * m)
                     for t, m in GAMMA_MULTIPLIERS.items()}
    coco_gram_raw = {t: mmd2_unbiased(anchor_raw, coco_raw, g_med_raw * m)
                     for t, m in GAMMA_MULTIPLIERS.items()}

    # CLIP
    logger.info("Loading CLIP...")
    import clip
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()

    logger.info("Extracting anchor CLIP feats...")
    anchor_clip = get_clip_feats(anchor_paths, clip_model, clip_preprocess, device)
    logger.info("Extracting COCO CLIP feats...")
    coco_clip = get_clip_feats(coco_paths, clip_model, clip_preprocess, device)

    g_med_clip = median_gamma(anchor_clip)
    coco_cmmd_fixed = mmd2_unbiased(anchor_clip, coco_clip, CMMD_GAMMA)
    coco_cmmd_med   = mmd2_unbiased(anchor_clip, coco_clip, g_med_clip)
    logger.info(f"COCO CMMD fixed={coco_cmmd_fixed:.4e}  med={coco_cmmd_med:.4e}")

    # ── Per-variant loop ──
    results = {}
    for vname, vlist in VKITTI_VARIANTS.items():
        logger.info(f"\n{'='*70}\nVARIANT: {vname}  {vlist}\n{'='*70}")

        vk_all = []
        for v in vlist:
            vk_all += list(VKITTI_DIR.glob(f"*{v}*.jpg"))
        vk_paths = sample_paths(vk_all, N_QUERY, SEED+2)
        logger.info(f"  Pool: {len(vk_all)} → {len(vk_paths)} sélectionnées")

        # Gram
        vk_std = extract_gram(vk_paths, ext_std, fit_transform=False)
        vk_raw = extract_gram(vk_paths, ext_raw, fit_transform=False)
        vk_gram_std = {t: mmd2_unbiased(anchor_std, vk_std, g_med_std * m)
                       for t, m in GAMMA_MULTIPLIERS.items()}
        vk_gram_raw = {t: mmd2_unbiased(anchor_raw, vk_raw, g_med_raw * m)
                       for t, m in GAMMA_MULTIPLIERS.items()}

        # CMMD
        logger.info("  Extracting CLIP feats...")
        vk_clip = get_clip_feats(vk_paths, clip_model, clip_preprocess, device)
        vk_cmmd_fixed = mmd2_unbiased(anchor_clip, vk_clip, CMMD_GAMMA)
        vk_cmmd_med   = mmd2_unbiased(anchor_clip, vk_clip, g_med_clip)

        results[vname] = {
            "gram_std": vk_gram_std, "gram_raw": vk_gram_raw,
            "cmmd_fixed": vk_cmmd_fixed, "cmmd_med": vk_cmmd_med,
        }

        # Print
        logger.info(f"\n  {'Méthode':<22} {'VK':>14} {'COCO':>14} {'Ratio':>8}  INV?")
        logger.info(f"  {'-'*65}")
        for tag in GAMMA_MULTIPLIERS:
            vv, cv = vk_gram_std[tag], coco_gram_std[tag]
            r = vv/cv if cv > 1e-12 else float("inf")
            logger.info(f"  Gram_std {tag:<12} {vv:>14.4e} {cv:>14.4e} {r:>8.3f}  {'OUI' if r<1 else 'non'}")
        for tag in GAMMA_MULTIPLIERS:
            vv, cv = vk_gram_raw[tag], coco_gram_raw[tag]
            r = vv/cv if cv > 1e-12 else float("inf")
            logger.info(f"  Gram_raw {tag:<12} {vv:>14.4e} {cv:>14.4e} {r:>8.3f}  {'OUI' if r<1 else 'non'}")
        for vv, cv, lbl in [(vk_cmmd_fixed, coco_cmmd_fixed, "CMMD γ=0.005"),
                             (vk_cmmd_med,   coco_cmmd_med,   "CMMD γ_med")]:
            r = vv/cv if cv > 1e-12 else float("inf")
            logger.info(f"  {lbl:<22} {vv:>14.4e} {cv:>14.4e} {r:>8.3f}  {'OUI' if r<1 else 'non'}")

        del vk_std, vk_raw, vk_clip
        gc.collect()

    # ── Final summary ──
    sep = "=" * 80
    logger.info(f"\n{sep}")
    logger.info("RÉSUMÉ — CMMD inversion (VK < COCO ?)")
    logger.info(sep)
    logger.info(f"  {'Variant':<12} {'CMMD_fix':>12} {'CMMD_med':>12}  fix/med")
    logger.info(f"  {'-'*50}")
    for vname, r in results.items():
        cf, cm = r["cmmd_fixed"], r["cmmd_med"]
        inv_f = "OUI" if cf < coco_cmmd_fixed else "non"
        inv_m = "OUI" if cm < coco_cmmd_med   else "non"
        star = "  ★" if inv_f == "OUI" or inv_m == "OUI" else ""
        logger.info(f"  {vname:<12} {cf:>12.4e} {cm:>12.4e}  {inv_f}/{inv_m}{star}")

    logger.info(f"\n{sep}")
    logger.info("RÉSUMÉ — Gram-MMD inversion (VK < COCO ?) [γ_med]")
    logger.info(sep)
    logger.info(f"  {'Variant':<12} {'std_γmed':>12} {'raw_γmed':>12}  std/raw")
    logger.info(f"  {'-'*50}")
    coco_s = coco_gram_std["γ_med"]; coco_r = coco_gram_raw["γ_med"]
    for vname, r in results.items():
        vs = r["gram_std"]["γ_med"]; vr = r["gram_raw"]["γ_med"]
        is_ = "OUI" if vs < coco_s else "non"
        ir  = "OUI" if vr < coco_r else "non"
        star = "  ★" if is_ == "OUI" or ir == "OUI" else ""
        logger.info(f"  {vname:<12} {vs:>12.4e} {vr:>12.4e}  {is_}/{ir}{star}")
    logger.info(sep)


if __name__ == "__main__":
    main()
