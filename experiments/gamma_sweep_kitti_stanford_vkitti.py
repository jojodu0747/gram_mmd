"""
gamma_sweep_kitti_stanford_vkitti.py
=====================================
Sweep gamma sur [γ_med/1000 … 100×γ_med] pour comparer :
  MMD²(KITTI_5k, VKITTI2_1k)  vs  MMD²(KITTI_5k, Stanford_1k)

Astuce : pré-calcul des matrices de distances carrées une seule fois,
puis application de exp(-γ·D) pour chaque γ → sweep rapide.

Produit :
  - experiments/results/gamma_sweep_kitti_stanford_vkitti_<ts>.json
  - experiments/results/gamma_sweep_kitti_stanford_vkitti_<ts>.png
"""

import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KITTI_DIR    = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR   = BASE_DIR / "dataset" / "vkitti_2.0 _rgb"
STANFORD_DIR = BASE_DIR / "dataset" / "stanford_cars"
RESULTS_DIR  = BASE_DIR / "experiments" / "results"

N_ANCHOR   = 5000
N_QUERY    = 1000
SEED       = 42
BATCH_SIZE = 16

BACKBONE   = "sd_vae"
LAYER      = 11
CMMD_GAMMA = 0.005

# Grille de multiplicateurs (log-space, 40 points entre γ/1000 et 100×γ)
N_GAMMA_STEPS = 40
GAMMA_MIN_MUL = 1e-3   # γ_med / 1000
GAMMA_MAX_MUL = 1e2    # 100 × γ_med


def sample_paths(paths, n, seed):
    rng = np.random.default_rng(seed)
    paths = sorted(paths)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(idx)]


def median_gamma(feats, max_samples=300, max_dims=512):
    rng = np.random.default_rng(0)
    n, d = feats.shape
    idx = rng.choice(n, size=min(n, max_samples), replace=False)
    sample = feats[idx]
    if d > max_dims:
        sample = sample[:, rng.choice(d, size=max_dims, replace=False)]
    sq = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1)
    return float(1.0 / (2 * np.median(sq[sq > 0]) + 1e-8))


def sq_dists(A, B):
    """||a - b||² via dot-product trick. Retourne (n, m)."""
    return (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * (A @ B.T)


def precompute_dists(X, Y):
    """Retourne D_xx (n,n), D_yy (m,m), D_xy (n,m)."""
    return sq_dists(X, X), sq_dists(Y, Y), sq_dists(X, Y)


def mmd2_from_dists(D_xx, D_yy, D_xy, gamma):
    n, m = D_xx.shape[0], D_yy.shape[0]
    Kxx = np.exp(-gamma * D_xx); np.fill_diagonal(Kxx, 0.0)
    Kyy = np.exp(-gamma * D_yy); np.fill_diagonal(Kyy, 0.0)
    Kxy = np.exp(-gamma * D_xy)
    return float(Kxx.sum() / (n * (n - 1)) +
                 Kyy.sum() / (m * (m - 1)) -
                 2 * Kxy.mean())


def get_gram(paths, ext, fit_transform=False):
    return ext.extract([str(p) for p in paths],
                       fit_transform=fit_transform,
                       batch_size=BATCH_SIZE).astype(np.float32)


def get_clip_feats(paths, model, preprocess, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [preprocess(Image.open(str(p)).convert("RGB"))
                 for p in paths[i:i + batch_size]]
        with torch.no_grad():
            f = model.encode_image(torch.stack(batch).to(device)).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def sweep_mmd(D_xx, D_yy_vk, D_xy_vk, D_yy_sc, D_xy_sc, gamma_med, multipliers):
    gammas = gamma_med * multipliers
    mmd_vk = np.array([mmd2_from_dists(D_xx, D_yy_vk, D_xy_vk, g) for g in gammas])
    mmd_sc = np.array([mmd2_from_dists(D_xx, D_yy_sc, D_xy_sc, g) for g in gammas])
    return gammas, mmd_vk, mmd_sc


def plot_sweep(multipliers, mmd_vk_gram, mmd_sc_gram,
               mmd_vk_clip, mmd_sc_clip, cmmd_fixed_vk, cmmd_fixed_sc,
               g_med_gram, g_med_clip, out_path):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MMD² sweep gamma — KITTI 5k anchor\nVKITTI2 vs Stanford Cars",
                 fontsize=13, fontweight="bold")

    for ax, mmd_vk, mmd_sc, g_med, label, extra in [
        (axes[0], mmd_vk_gram, mmd_sc_gram, g_med_gram, "Gram-MMD (SD-VAE L11)", None),
        (axes[1], mmd_vk_clip, mmd_sc_clip, g_med_clip, "CMMD (CLIP ViT-L/14)",
         (CMMD_GAMMA, cmmd_fixed_vk, cmmd_fixed_sc)),
    ]:
        ax.semilogx(multipliers, mmd_vk, "b-o", markersize=3, label="VKITTI2 (synthetic)")
        ax.semilogx(multipliers, mmd_sc, "r-s", markersize=3, label="Stanford Cars (real)")
        ax.fill_between(multipliers,
                        np.minimum(mmd_vk, mmd_sc),
                        np.maximum(mmd_vk, mmd_sc),
                        alpha=0.12,
                        color="green" if (mmd_vk > mmd_sc).all() else "orange")

        # Références γ/10, γ_med, 10×γ, 100×γ
        for mul, name in [(0.1, "γ/10"), (1.0, "γ_med"), (10.0, "10γ"), (100.0, "100γ")]:
            ax.axvline(mul, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
            ax.text(mul, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    name, rotation=90, fontsize=7, va="top", ha="right", color="gray")

        # Ligne CMMD fixe γ=0.005
        if extra is not None:
            g_fixed, vk_f, sc_f = extra
            rel = g_fixed / g_med
            ax.axvline(rel, color="purple", linestyle=":", linewidth=1.2, label=f"γ=0.005 (×{rel:.2e})")
            ax.plot(rel, vk_f, "b*", markersize=8)
            ax.plot(rel, sc_f, "r*", markersize=8)

        # Zone d'inversion : VKITTI2 (synth) plus proche que Stanford (réel)
        inv_mask = mmd_vk < mmd_sc
        if inv_mask.any():
            inv_muls = multipliers[inv_mask]
            ax.axvspan(inv_muls.min(), inv_muls.max(), alpha=0.08, color="red",
                       label="zone inversion")

        ax.set_xlabel("Multiplicateur × γ_med (log scale)")
        ax.set_ylabel("MMD² (unbiased)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Plot saved → {out_path}")
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Paths ────────────────────────────────────────────────────
    anchor_paths  = sample_paths(list(KITTI_DIR.glob("*.png")), N_ANCHOR, SEED)
    vkitti_all    = list(VKITTI_DIR.rglob("*.jpg"))
    vkitti_paths  = sample_paths(vkitti_all, N_QUERY, SEED + 1)
    stanford_all  = list(STANFORD_DIR.glob("*.jpg")) + list(STANFORD_DIR.glob("*.png"))
    stanford_paths = sample_paths(stanford_all, N_QUERY, SEED + 2)

    logger.info(f"Anchor : {len(anchor_paths)} KITTI")
    logger.info(f"QueryA : {len(vkitti_paths)} VKITTI2  (pool={len(vkitti_all)})")
    logger.info(f"QueryB : {len(stanford_paths)} Stanford Cars")

    # ── Gram features ────────────────────────────────────────────
    transform_cfg = {"name": "gram_spatial", "use_gram": True,
                     "use_pca": False, "gram_patches": False, "gram_averaging": "spatial"}
    ext = FeatureExtractor(backbone=BACKBONE, layer=LAYER,
                           transform_config=transform_cfg, device=device)

    logger.info("Extraction Gram anchor (fit)...")
    anc_gram = get_gram(anchor_paths, ext, fit_transform=True)
    logger.info("Extraction Gram VKITTI2...")
    vk_gram  = get_gram(vkitti_paths, ext)
    logger.info("Extraction Gram Stanford Cars...")
    sc_gram  = get_gram(stanford_paths, ext)

    g_med_gram = median_gamma(anc_gram)
    logger.info(f"γ_med Gram = {g_med_gram:.4e}")

    logger.info("Pré-calcul distances Gram...")
    D_xx_g, D_yy_vk_g, D_xy_vk_g = precompute_dists(anc_gram, vk_gram)
    D_yy_sc_g = sq_dists(sc_gram, sc_gram)
    D_xy_sc_g = sq_dists(anc_gram, sc_gram)
    del anc_gram, vk_gram, sc_gram; gc.collect()

    # ── CLIP features ─────────────────────────────────────────────
    logger.info("Chargement CLIP...")
    import clip
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()

    logger.info("Extraction CLIP anchor...")
    anc_clip = get_clip_feats(anchor_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP VKITTI2...")
    vk_clip  = get_clip_feats(vkitti_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP Stanford Cars...")
    sc_clip  = get_clip_feats(stanford_paths, clip_model, clip_preprocess, device)

    g_med_clip = median_gamma(anc_clip)
    logger.info(f"γ_med CLIP = {g_med_clip:.4e}")

    # CMMD fixe γ=0.005
    cmmd_fixed_vk = mmd2_from_dists(*precompute_dists(anc_clip, vk_clip), CMMD_GAMMA)
    cmmd_fixed_sc = mmd2_from_dists(*precompute_dists(anc_clip, sc_clip), CMMD_GAMMA)

    logger.info("Pré-calcul distances CLIP...")
    D_xx_c, D_yy_vk_c, D_xy_vk_c = precompute_dists(anc_clip, vk_clip)
    D_yy_sc_c = sq_dists(sc_clip, sc_clip)
    D_xy_sc_c = sq_dists(anc_clip, sc_clip)
    del anc_clip, vk_clip, sc_clip; gc.collect()

    # ── Sweep gamma ───────────────────────────────────────────────
    multipliers = np.logspace(np.log10(GAMMA_MIN_MUL), np.log10(GAMMA_MAX_MUL), N_GAMMA_STEPS)

    logger.info("Sweep Gram-MMD...")
    _, mmd_vk_gram, mmd_sc_gram = sweep_mmd(
        D_xx_g, D_yy_vk_g, D_xy_vk_g, D_yy_sc_g, D_xy_sc_g, g_med_gram, multipliers)

    logger.info("Sweep CMMD...")
    _, mmd_vk_clip, mmd_sc_clip = sweep_mmd(
        D_xx_c, D_yy_vk_c, D_xy_vk_c, D_yy_sc_c, D_xy_sc_c, g_med_clip, multipliers)

    # ── Affichage résumé ──────────────────────────────────────────
    sep = "=" * 70
    logger.info(f"\n{sep}")
    logger.info("GAMMA SWEEP — KITTI 5k | VKITTI2 vs Stanford Cars")
    logger.info(f"  γ_med Gram = {g_med_gram:.4e}   γ_med CLIP = {g_med_clip:.4e}")
    logger.info(f"  CMMD fixe (γ=0.005) : VKITTI2={cmmd_fixed_vk:.4e}  Stanford={cmmd_fixed_sc:.4e}")

    inv_gram = int((mmd_vk_gram < mmd_sc_gram).sum())
    inv_clip = int((mmd_vk_clip < mmd_sc_clip).sum())
    logger.info(f"  Gram-MMD : inversion (VKITTI2 < Stanford) pour {inv_gram}/{N_GAMMA_STEPS} gammas")
    logger.info(f"  CMMD     : inversion (VKITTI2 < Stanford) pour {inv_clip}/{N_GAMMA_STEPS} gammas")
    logger.info(sep)

    # ── Plot ──────────────────────────────────────────────────────
    plot_path = RESULTS_DIR / f"gamma_sweep_kitti_stanford_vkitti_{ts}.png"
    plot_sweep(multipliers, mmd_vk_gram, mmd_sc_gram,
               mmd_vk_clip, mmd_sc_clip, cmmd_fixed_vk, cmmd_fixed_sc,
               g_med_gram, g_med_clip, plot_path)

    # ── Save JSON ─────────────────────────────────────────────────
    out_json = RESULTS_DIR / f"gamma_sweep_kitti_stanford_vkitti_{ts}.json"
    results = {
        "experiment": "gamma_sweep_kitti_stanford_vkitti",
        "timestamp": ts,
        "config": {
            "backbone": BACKBONE, "layer": LAYER,
            "n_anchor": N_ANCHOR, "n_query": N_QUERY,
            "seed": SEED, "cmmd_gamma": CMMD_GAMMA,
            "gamma_min_mul": GAMMA_MIN_MUL, "gamma_max_mul": GAMMA_MAX_MUL,
            "n_gamma_steps": N_GAMMA_STEPS,
        },
        "paths": {
            "anchor": [str(p) for p in anchor_paths],
            "vkitti": [str(p) for p in vkitti_paths],
            "stanford": [str(p) for p in stanford_paths],
        },
        "gamma_med": {"gram": g_med_gram, "clip": g_med_clip},
        "cmmd_fixed_vkitti": cmmd_fixed_vk,
        "cmmd_fixed_stanford": cmmd_fixed_sc,
        "sweep": {
            "multipliers": multipliers.tolist(),
            "gram_vkitti":   mmd_vk_gram.tolist(),
            "gram_stanford": mmd_sc_gram.tolist(),
            "clip_vkitti":   mmd_vk_clip.tolist(),
            "clip_stanford": mmd_sc_clip.tolist(),
        },
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved → {out_json}")


if __name__ == "__main__":
    main()
