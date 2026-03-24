"""
kitti_stanford_vkitti_samesize_test.py
=======================================
Variante de kitti_stanford_vkitti_test.py avec filtrage des Stanford Cars
par taille (largeur >= MIN_WIDTH px) pour se rapprocher du format KITTI (~1242x375).

KITTI   : ~1242x375  (ratio ~3.3)
Stanford: médiane 582x371 → on garde W >= 600 (4677 images dispo sur 8144)

Anchor  : 1000 images KITTI originales (dataset/kitti/images/)
Query A : 1000 images VKITTI2 variées (toutes conditions, tous scènes)
Query B : 1000 images Stanford Cars filtrées W >= MIN_WIDTH

Hypothèse :
  - CMMD (CLIP) : Stanford Cars ≈ KITTI sémantiquement (voitures réelles)
                  → CMMD(KITTI, Stanford) < CMMD(KITTI, VKITTI2) ? INVERSION ?
  - Gram-MMD    : VKITTI2 synthétique → textures très différentes de KITTI réel
                  → Gram(KITTI, VKITTI2) > Gram(KITTI, Stanford)  PAS d'inversion
"""

import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KITTI_DIR        = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR       = BASE_DIR / "dataset" / "vkitti_2.0 _rgb"
STANFORD_DIR     = BASE_DIR / "dataset" / "stanford_cars"

N_ANCHOR    = 1000
N_QUERY     = 1000
SEED        = 42
BATCH_SIZE  = 16

BACKBONE = "sd_vae"
LAYER    = 11
GAMMA_MULTIPLIERS = {"γ/10": 0.1, "γ_med": 1.0, "10×γ": 10.0, "100×γ": 100.0}
CMMD_GAMMA = 0.005
MIN_WIDTH  = 600   # filtre Stanford Cars : largeur minimale en pixels

RESULTS_DIR = BASE_DIR / "experiments" / "results"


def filter_by_width(paths, min_width):
    """Garde uniquement les images dont la largeur >= min_width."""
    from PIL import Image as _Image
    kept = []
    for p in paths:
        try:
            w, _ = _Image.open(p).size
            if w >= min_width:
                kept.append(p)
        except Exception:
            pass
    return kept


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
        sample = sample[:, rng.choice(d, size=max_dims, replace=False)]
    sq = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1)
    return float(1.0 / (2 * np.median(sq[sq > 0]) + 1e-8))


def sq_dists_fast(A, B):
    return (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2*(A @ B.T)


def mmd2_unbiased(X, Y, gamma):
    n, m = len(X), len(Y)
    Kxx = np.exp(-gamma * sq_dists_fast(X, X)); np.fill_diagonal(Kxx, 0)
    Kyy = np.exp(-gamma * sq_dists_fast(Y, Y)); np.fill_diagonal(Kyy, 0)
    Kxy = np.exp(-gamma * sq_dists_fast(X, Y))
    return float(Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2*Kxy.mean())


def get_gram(paths, ext, fit_transform=False):
    return ext.extract([str(p) for p in paths],
                       fit_transform=fit_transform, batch_size=BATCH_SIZE).astype(np.float32)


def get_clip_feats(paths, clip_model, clip_preprocess, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [clip_preprocess(Image.open(str(p)).convert("RGB")) for p in paths[i:i+batch_size]]
        with torch.no_grad():
            f = clip_model.encode_image(torch.stack(batch).to(device)).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def print_row(label, vv, cv, width=24):
    r = vv / cv if cv > 1e-12 else float("inf")
    logger.info(f"  {label:<{width}} {vv:>14.4e}  {cv:>14.4e}  {r:>7.3f}  {'OUI ★' if r < 1 else 'non'}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Collect paths ────────────────────────────────────────────
    anchor_paths = sample_paths(list(KITTI_DIR.glob("*.png")), N_ANCHOR, SEED)
    logger.info(f"Anchor  : {len(anchor_paths)} KITTI originales")

    # VKITTI2 : all variants, all scenes
    vkitti_all = list(VKITTI_DIR.rglob("*.jpg"))
    vkitti_paths = sample_paths(vkitti_all, N_QUERY, SEED + 1)
    logger.info(f"Query A : {len(vkitti_paths)} VKITTI2 variées (pool={len(vkitti_all)})")

    stanford_all_raw = list(STANFORD_DIR.glob("*.jpg")) + list(STANFORD_DIR.glob("*.png"))
    logger.info(f"Filtrage Stanford Cars W>={MIN_WIDTH}px ({len(stanford_all_raw)} total)...")
    stanford_all = filter_by_width(stanford_all_raw, MIN_WIDTH)
    stanford_paths = sample_paths(stanford_all, N_QUERY, SEED + 2)
    logger.info(f"Query B : {len(stanford_paths)} Stanford Cars filtrées (pool={len(stanford_all)})")

    if len(stanford_paths) == 0:
        logger.error("Stanford Cars dataset not found! Run the download script first.")
        return

    # ── Gram extractors ──────────────────────────────────────────
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

    logger.info("Extraction Gram VKITTI2...")
    vk_std = get_gram(vkitti_paths, ext_std)
    vk_raw = get_gram(vkitti_paths, ext_raw)

    logger.info("Extraction Gram Stanford Cars...")
    sc_std = get_gram(stanford_paths, ext_std)
    sc_raw = get_gram(stanford_paths, ext_raw)

    g_std = median_gamma(anchor_std)
    g_raw = median_gamma(anchor_raw)

    gram_std_vk = {t: mmd2_unbiased(anchor_std, vk_std, g_std*m) for t, m in GAMMA_MULTIPLIERS.items()}
    gram_std_sc = {t: mmd2_unbiased(anchor_std, sc_std, g_std*m) for t, m in GAMMA_MULTIPLIERS.items()}
    gram_raw_vk = {t: mmd2_unbiased(anchor_raw, vk_raw, g_raw*m) for t, m in GAMMA_MULTIPLIERS.items()}
    gram_raw_sc = {t: mmd2_unbiased(anchor_raw, sc_raw, g_raw*m) for t, m in GAMMA_MULTIPLIERS.items()}

    del vk_std, vk_raw, sc_std, sc_raw
    gc.collect()

    # ── CLIP ─────────────────────────────────────────────────────
    logger.info("\nChargement CLIP...")
    import clip
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()

    logger.info("Extraction CLIP anchor...")
    anchor_clip = get_clip_feats(anchor_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP VKITTI2...")
    vk_clip = get_clip_feats(vkitti_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP Stanford Cars...")
    sc_clip = get_clip_feats(stanford_paths, clip_model, clip_preprocess, device)

    g_clip = median_gamma(anchor_clip)
    cmmd_vk_fixed = mmd2_unbiased(anchor_clip, vk_clip,  CMMD_GAMMA)
    cmmd_sc_fixed = mmd2_unbiased(anchor_clip, sc_clip,  CMMD_GAMMA)
    cmmd_vk_med   = mmd2_unbiased(anchor_clip, vk_clip,  g_clip)
    cmmd_sc_med   = mmd2_unbiased(anchor_clip, sc_clip,  g_clip)

    # ── Print results ─────────────────────────────────────────────
    sep = "=" * 78
    logger.info(f"\n{sep}")
    logger.info("KITTI → VKITTI2 variés  vs  Stanford Cars")
    logger.info(f"  Anchor: KITTI orig | Query A: VKITTI2 | Query B: Stanford Cars")
    logger.info(sep)
    logger.info(f"  {'Méthode':<24} {'VKITTI2':>14}  {'Stanford':>14}  {'Ratio':>7}  INV?")
    logger.info(f"  {'-'*70}")
    for tag in GAMMA_MULTIPLIERS:
        print_row(f"Gram_std {tag}", gram_std_vk[tag], gram_std_sc[tag])
    for tag in GAMMA_MULTIPLIERS:
        print_row(f"Gram_raw {tag}", gram_raw_vk[tag], gram_raw_sc[tag])
    print_row("CMMD γ=0.005",  cmmd_vk_fixed, cmmd_sc_fixed)
    print_row("CMMD γ_med",    cmmd_vk_med,   cmmd_sc_med)
    logger.info(sep)

    cmmd_inv   = cmmd_vk_fixed < cmmd_sc_fixed or cmmd_vk_med < cmmd_sc_med
    gram_inv_g10 = (gram_std_vk["γ/10"] < gram_std_sc["γ/10"] or
                    gram_raw_vk["γ/10"] < gram_raw_sc["γ/10"])
    gram_inv_any = (any(gram_std_vk[t] < gram_std_sc[t] for t in GAMMA_MULTIPLIERS) or
                    any(gram_raw_vk[t] < gram_raw_sc[t] for t in GAMMA_MULTIPLIERS))

    if cmmd_inv and not gram_inv_g10:
        verdict = "SUCCESS: CMMD inverts (VKITTI2 < Stanford), Gram-MMD (γ/10) does NOT"
    elif cmmd_inv and gram_inv_any:
        verdict = "BOTH invert"
    elif not cmmd_inv and not gram_inv_any:
        verdict = "NO inversion"
    else:
        verdict = "Gram-MMD inverts but not CMMD"

    logger.info("")
    logger.info(f"  ▶ {verdict}")
    logger.info(sep)

    # ── Save results ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"kitti_stanford_vkitti_samesize_{ts}.json"

    results = {
        "experiment": "kitti_stanford_vkitti_samesize_test",
        "timestamp": ts,
        "config": {
            "backbone": BACKBONE, "layer": LAYER,
            "n_anchor": N_ANCHOR, "n_query": N_QUERY,
            "seed": SEED, "cmmd_gamma": CMMD_GAMMA,
            "stanford_min_width": MIN_WIDTH,
            "gamma_multipliers": GAMMA_MULTIPLIERS,
        },
        "paths": {
            "anchor": [str(p) for p in anchor_paths],
            "vkitti": [str(p) for p in vkitti_paths],
            "stanford": [str(p) for p in stanford_paths],
        },
        "gamma_med": {"std": g_std, "raw": g_raw, "clip": g_clip},
        "gram_std_vkitti":  gram_std_vk,
        "gram_std_stanford": gram_std_sc,
        "gram_raw_vkitti":  gram_raw_vk,
        "gram_raw_stanford": gram_raw_sc,
        "cmmd_vkitti_fixed":  cmmd_vk_fixed,
        "cmmd_stanford_fixed": cmmd_sc_fixed,
        "cmmd_vkitti_med":    cmmd_vk_med,
        "cmmd_stanford_med":  cmmd_sc_med,
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
