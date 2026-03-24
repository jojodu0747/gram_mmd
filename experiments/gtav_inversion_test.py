"""
gtav_inversion_test.py
======================
Test d'inversion CMMD vs Gram-MMD : GTA V (synthétique conduite) vs COCO.

  Ancrage : 1000 images KITTI Tracking (toutes séquences, aléatoire)
  Query A : ~456 GTA V Vehicle Dataset (synthétique, domaine conduite)
  Query B :  456 COCO (domaine différent, réel)

Hypothèse :
  - CMMD (CLIP sémantique) : GTA V "conduite" ≈ ancrage "conduite" → INVERSION
  - GMMD (Gram texture)    : texture jeu vidéo ≠ réel → PAS d'inversion

Usage:
    .venv/bin/python experiments/gtav_inversion_test.py
"""

import csv
import gc
import glob as glob_module
import logging
import sys
import time
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

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

TRACKING_DIR = BASE_DIR / "dataset" / "data_tracking_image_2" / "training" / "image_02"
GTAV_DIR     = BASE_DIR / "dataset" / "GTA-V-Vehicle-Dataset-1"
COCO_DIR     = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR      = BASE_DIR / "results" / "gtav_inversion"

N_ANCHOR = 1000
SEED     = 42
BATCH_SIZE = 16

# Gram-MMD
BACKBONE = "sd_vae"
LAYER    = 11
GAMMA_MULTIPLIERS = {
    "γ/10":   0.1,
    "γ_med":  1.0,
    "10×γ":  10.0,
    "100×γ": 100.0,
}

# CLIP / CMMD
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32


# ════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ════════════════════════════════════════════════════════════

def load_anchor_paths():
    """Load all tracking frames (all sequences)."""
    paths = []
    for seq_dir in sorted(TRACKING_DIR.iterdir()):
        paths.extend(sorted(glob_module.glob(str(seq_dir / "*.png"))))
    logger.info(f"  Anchor pool (all tracking seqs): {len(paths)} frames")
    return paths


def load_gtav_paths():
    """Load all GTA V images from train/test/valid splits."""
    paths = []
    for split in ["train", "test", "valid"]:
        img_dir = GTAV_DIR / split / "images"
        if img_dir.exists():
            paths.extend(sorted(
                glob_module.glob(str(img_dir / "*.jpg")) +
                glob_module.glob(str(img_dir / "*.png"))
            ))
    logger.info(f"  GTA V pool: {len(paths)} images")
    return paths


def load_coco_paths():
    paths = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    logger.info(f"  COCO pool: {len(paths)} images")
    return paths


def subsample(paths, n, seed):
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(paths), size=min(n, len(paths)), replace=False).tolist())
    return [paths[i] for i in idx]


# ════════════════════════════════════════════════════════════
#  DISTANCE UTILS
# ════════════════════════════════════════════════════════════

def squared_l2_matrix(X, Y):
    X_sq = np.einsum("ij,ij->i", X, X)
    Y_sq = np.einsum("ij,ij->i", Y, Y)
    return X_sq[:, None] + Y_sq[None, :] - 2.0 * (X @ Y.T)


def mmd2_unbiased(ref, query, gamma):
    n_r, n_q = len(ref), len(query)
    D_rr = squared_l2_matrix(ref, ref)
    D_qq = squared_l2_matrix(query, query)
    D_rq = squared_l2_matrix(ref, query)
    K_rr = np.exp(-gamma * D_rr)
    K_qq = np.exp(-gamma * D_qq)
    K_rq = np.exp(-gamma * D_rq)
    t1 = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))
    t2 = (np.sum(K_qq) - np.trace(K_qq)) / (n_q * (n_q - 1))
    t3 = 2.0 * np.sum(K_rq) / (n_r * n_q)
    return float(t1 + t2 - t3)


def median_gamma(features):
    f_sq = np.einsum("ij,ij->i", features, features)
    D = f_sq[:, None] + f_sq[None, :] - 2.0 * (features @ features.T)
    triu = D[np.triu_indices(len(features), k=1)]
    return 1.0 / (2.0 * float(np.median(triu)) + 1e-8)


def cosine_distance(feat_ref, feat_query):
    ref_n = feat_ref / (np.linalg.norm(feat_ref, axis=1, keepdims=True) + 1e-8)
    q_n = feat_query / (np.linalg.norm(feat_query, axis=1, keepdims=True) + 1e-8)
    return float(1.0 - np.mean(ref_n @ q_n.T))


def compute_fid(feat_ref, feat_query):
    mu_r, mu_q = np.mean(feat_ref, axis=0), np.mean(feat_query, axis=0)
    diff = mu_r - mu_q
    mean_term = np.sum(diff ** 2)
    d = feat_ref.shape[1]
    cov_r = np.cov(feat_ref, rowvar=False) + np.eye(d) * 1e-6
    cov_q = np.cov(feat_query, rowvar=False) + np.eye(d) * 1e-6
    from scipy import linalg
    covmean, _ = linalg.sqrtm(cov_r @ cov_q, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(mean_term + np.trace(cov_r + cov_q - 2 * covmean))


def clip_embed(paths, model, proc, device):
    embs = []
    for i in range(0, len(paths), BATCH_SIZE_CLIP):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE_CLIP]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


# ════════════════════════════════════════════════════════════
#  EVALUATION
# ════════════════════════════════════════════════════════════

def evaluate_gram(anchor_paths, gtav_paths, coco_paths, device, use_scaler=True):
    ext = FeatureExtractor(
        backbone=BACKBONE, layer=LAYER,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )

    gram_ref = ext.extract(anchor_paths, fit_transform=use_scaler,
                           batch_size=BATCH_SIZE).astype(np.float32)
    gram_gta = ext.extract(gtav_paths, fit_transform=False,
                           batch_size=BATCH_SIZE).astype(np.float32)
    gram_co  = ext.extract(coco_paths, fit_transform=False,
                           batch_size=BATCH_SIZE).astype(np.float32)
    ext.cleanup()
    del ext
    torch.cuda.empty_cache()
    gc.collect()

    g_med = median_gamma(gram_ref)
    rows = []
    for tag, mult in GAMMA_MULTIPLIERS.items():
        gamma = mult * g_med
        m_gta = mmd2_unbiased(gram_ref, gram_gta, gamma)
        m_co = mmd2_unbiased(gram_ref, gram_co, gamma)
        rows.append({
            "method": f"Gram{'_std' if use_scaler else '_raw'} {tag}",
            "gtav": m_gta, "coco": m_co, "gamma": gamma,
            "inversion": m_gta < m_co,
        })

    del gram_ref, gram_gta, gram_co
    gc.collect()
    return rows


def evaluate_clip(anchor_paths, gtav_paths, coco_paths, device,
                  clip_model, clip_proc):
    clip_ref = clip_embed(anchor_paths, clip_model, clip_proc, device)
    clip_gta = clip_embed(gtav_paths, clip_model, clip_proc, device)
    clip_co  = clip_embed(coco_paths, clip_model, clip_proc, device)

    rows = []

    # CMMD γ=0.005
    m_gta = mmd2_unbiased(clip_ref, clip_gta, CMMD_GAMMA)
    m_co = mmd2_unbiased(clip_ref, clip_co, CMMD_GAMMA)
    rows.append({"method": "CMMD γ=0.005", "gtav": m_gta, "coco": m_co,
                 "gamma": CMMD_GAMMA, "inversion": m_gta < m_co})

    # CMMD γ_median
    g_med = median_gamma(clip_ref)
    m_gta = mmd2_unbiased(clip_ref, clip_gta, g_med)
    m_co = mmd2_unbiased(clip_ref, clip_co, g_med)
    rows.append({"method": "CMMD γ_med", "gtav": m_gta, "coco": m_co,
                 "gamma": g_med, "inversion": m_gta < m_co})

    # Cosine
    cos_gta = cosine_distance(clip_ref, clip_gta)
    cos_co = cosine_distance(clip_ref, clip_co)
    rows.append({"method": "Cosine CLIP", "gtav": cos_gta, "coco": cos_co,
                 "gamma": None, "inversion": cos_gta < cos_co})

    # FID
    fid_gta = compute_fid(clip_ref, clip_gta)
    fid_co = compute_fid(clip_ref, clip_co)
    rows.append({"method": "FID CLIP", "gtav": fid_gta, "coco": fid_co,
                 "gamma": None, "inversion": fid_gta < fid_co})

    del clip_ref, clip_gta, clip_co
    gc.collect()
    return rows


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_total = time.time()

    # ── Load pools ────────────────────────────────────────
    anchor_pool = load_anchor_paths()
    gtav_paths  = load_gtav_paths()      # all ~456 images
    coco_pool   = load_coco_paths()

    n_query = len(gtav_paths)            # match COCO size to GTA V
    anchor_paths = subsample(anchor_pool, N_ANCHOR, SEED)
    coco_paths   = subsample(coco_pool, n_query, SEED)

    logger.info("=" * 90)
    logger.info("GTA V INVERSION TEST")
    logger.info(f"  Anchor:  {len(anchor_paths)} KITTI Tracking (all seqs)")
    logger.info(f"  Query A: {len(gtav_paths)} GTA V")
    logger.info(f"  Query B: {len(coco_paths)} COCO")
    logger.info(f"  Device: {device}")
    logger.info("=" * 90)

    all_rows = []

    # Gram-MMD with scaler
    logger.info("\n  Gram-MMD + StandardScaler")
    rows = evaluate_gram(anchor_paths, gtav_paths, coco_paths, device, use_scaler=True)
    for r in rows:
        logger.info(f"    {r['method']:<18s}  GTA={r['gtav']:.4e}  CO={r['coco']:.4e}  "
                     f"ratio={r['gtav']/r['coco']:.3f}  {'INV!' if r['inversion'] else 'ok'}")
    all_rows.extend(rows)

    # Gram-MMD raw
    logger.info("\n  Gram-MMD raw (no scaler)")
    rows = evaluate_gram(anchor_paths, gtav_paths, coco_paths, device, use_scaler=False)
    for r in rows:
        logger.info(f"    {r['method']:<18s}  GTA={r['gtav']:.4e}  CO={r['coco']:.4e}  "
                     f"ratio={r['gtav']/r['coco']:.3f}  {'INV!' if r['inversion'] else 'ok'}")
    all_rows.extend(rows)

    # CLIP metrics
    logger.info("\n  CMMD / CLIP metrics")
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    rows = evaluate_clip(anchor_paths, gtav_paths, coco_paths, device, clip_model, clip_proc)
    for r in rows:
        logger.info(f"    {r['method']:<18s}  GTA={r['gtav']:.4e}  CO={r['coco']:.4e}  "
                     f"ratio={r['gtav']/r['coco']:.3f}  {'INV!' if r['inversion'] else 'ok'}")
    all_rows.extend(rows)

    del clip_model, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════
    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")

    print("\n" + "=" * 90)
    print("GTA V INVERSION TEST")
    print(f"Anchor = {len(anchor_paths)} KITTI Tracking | Query A = {len(gtav_paths)} GTA V | Query B = {len(coco_paths)} COCO")
    print("=" * 90)
    print(f"  {'Méthode':<20s}  {'GTA V':>14s}  {'COCO':>14s}  {'Ratio':>8s}  {'INV?':>5s}")
    print("  " + "-" * 65)
    for r in all_rows:
        ratio = r["gtav"] / r["coco"] if r["coco"] > 0 else float("inf")
        print(f"  {r['method']:<20s}  {r['gtav']:>14.4e}  {r['coco']:>14.4e}  "
              f"{ratio:>8.3f}  {'OUI' if r['inversion'] else 'non':>5s}")

    n_cmmd = sum(1 for r in all_rows if r["inversion"] and "CMMD" in r["method"])
    n_gram = sum(1 for r in all_rows if r["inversion"] and "Gram" in r["method"])
    if n_cmmd > 0 and n_gram == 0:
        print(f"\n  ★ CMMD inverse mais PAS Gram-MMD ★")
    elif n_gram > 0 and n_cmmd == 0:
        print(f"\n  ★ Gram-MMD inverse mais PAS CMMD ★")
    print("=" * 90)

    # ════════════════════════════════════════════════════════
    #  SAVE
    # ════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "gtav", "coco", "ratio", "gamma", "inversion"])
        for r in all_rows:
            ratio = r["gtav"] / r["coco"] if r["coco"] > 0 else float("inf")
            w.writerow([r["method"],
                        f"{r['gtav']:.6e}", f"{r['coco']:.6e}", f"{ratio:.4f}",
                        f"{r['gamma']:.4e}" if r["gamma"] is not None else "",
                        r["inversion"]])
    logger.info(f"  CSV: {csv_path}")

    # Plot
    methods = [r["method"] for r in all_rows]
    ratios = [r["gtav"] / r["coco"] if r["coco"] > 0 else 0 for r in all_rows]
    colors = ["#F44336" if r < 1 else "#4CAF50" for r in ratios]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(methods, ratios, color=colors, edgecolor="white", zorder=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{r:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Ratio MMD²(GTA V) / MMD²(COCO)")
    ax.set_title(
        f"Anchor = {len(anchor_paths)} KITTI Tracking | GTA V ({len(gtav_paths)}) vs COCO ({len(coco_paths)})\n"
        "Rouge <1 = INVERSION (GTA V vu comme plus proche que COCO)",
        fontsize=11, fontweight="bold",
    )
    ax.set_facecolor("#f9f9f9")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(plots_dir / "gtav_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'gtav_ratios.png'}")
    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
