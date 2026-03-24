"""
compare_kitti_distributions.py
===============================
Compare Gram-MMD and CMMD distances between:
  - Anchor : 1000 KITTI images (real driving scenes)
  - Query 1: 1000 Virtual KITTI images (synthetic driving scenes)
  - Query 2: 1000 KITTI Tracking (data_tracking_image_2) images

Gram-MMD : SD-VAE layer 11, Gram spatial + StandardScaler
  4 gamma values: gamma_med/10, gamma_med, 10*gamma_med, 100*gamma_med

CMMD : CLIP ViT-L/14@336, gamma fixe = 0.005

Usage:
    .venv/bin/python experiments/compare_kitti_distributions.py
"""

import gc
import glob as glob_module
import logging
import os
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

from gram_mmd.features import FeatureExtractor, ImageDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

KITTI_DIR        = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR       = BASE_DIR / "dataset" / "virtual_kitti"
TRACKING_DIR     = BASE_DIR / "dataset" / "data_tracking_image_2"
OUT_DIR          = BASE_DIR / "results" / "kitti_distribution_compare"

N_IMAGES         = 1000
SEED             = 42
BATCH_SIZE       = 16

# Gram-MMD config
BACKBONE         = "sd_vae"
LAYER            = 11

GAMMA_MULTIPLIERS = {
    "gamma_div10":  0.1,
    "gamma_med":    1.0,
    "10x_gamma":   10.0,
    "100x_gamma": 100.0,
}

# CMMD config
CLIP_MODEL_ID    = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA       = 0.005
BATCH_SIZE_CLIP  = 32


# ════════════════════════════════════════════════════════════
#  LOAD IMAGE PATHS
# ════════════════════════════════════════════════════════════

def sample_images(directory, n, seed, recursive=False):
    """Sample n images from a directory."""
    if recursive:
        all_imgs = sorted(
            glob_module.glob(str(directory / "**" / "*.png"), recursive=True) +
            glob_module.glob(str(directory / "**" / "*.jpg"), recursive=True)
        )
    else:
        all_imgs = sorted(
            glob_module.glob(str(directory / "*.png")) +
            glob_module.glob(str(directory / "*.jpg")) +
            glob_module.glob(str(directory / "*.jpeg"))
        )
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {directory}")
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(all_imgs), size=min(n, len(all_imgs)), replace=False).tolist())
    paths = [all_imgs[i] for i in idx]
    logger.info(f"  Sampled {len(paths)} images from {directory}")
    return paths


# ════════════════════════════════════════════════════════════
#  GRAM-MMD : extract + compute
# ════════════════════════════════════════════════════════════

def extract_features(backbone, layer, image_paths, batch_size, device, fit_transform=False, extractor=None):
    """Extract Gram features with StandardScaler."""
    if extractor is None:
        extractor = FeatureExtractor(
            backbone=backbone, layer=layer,
            transform_config={
                "name": "gram_spatial", "use_gram": True,
                "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
            },
            device=device,
        )
    feats = extractor.extract(image_paths, fit_transform=fit_transform,
                               batch_size=batch_size).astype(np.float32)
    return extractor, feats


def compute_median_gamma(features):
    """Compute gamma via median heuristic on pairwise squared L2."""
    f_sq = np.einsum("ij,ij->i", features, features)
    D = f_sq[:, None] + f_sq[None, :] - 2.0 * (features @ features.T)
    triu = D[np.triu_indices(len(features), k=1)]
    median_d2 = float(np.median(triu))
    gamma = 1.0 / (2.0 * median_d2 + 1e-8)
    logger.info(f"  median(d²) = {median_d2:.4e}  ->  gamma_med = {gamma:.4e}")
    return gamma


def mmd2_unbiased(ref, query, gamma):
    """Compute unbiased MMD² between two feature sets with RBF kernel."""
    n_r, n_q = len(ref), len(query)

    ref_sq = np.einsum("ij,ij->i", ref, ref)
    query_sq = np.einsum("ij,ij->i", query, query)

    # K_rr
    D_rr = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
    K_rr = np.exp(-gamma * D_rr)
    term_rr = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))

    # K_qq
    D_qq = query_sq[:, None] + query_sq[None, :] - 2.0 * (query @ query.T)
    K_qq = np.exp(-gamma * D_qq)
    term_qq = (np.sum(K_qq) - np.trace(K_qq)) / (n_q * (n_q - 1))

    # K_rq
    D_rq = ref_sq[:, None] + query_sq[None, :] - 2.0 * (ref @ query.T)
    K_rq = np.exp(-gamma * D_rq)
    term_rq = 2.0 * np.sum(K_rq) / (n_r * n_q)

    return float(term_rr + term_qq - term_rq)


# ════════════════════════════════════════════════════════════
#  CMMD — CLIP ViT-L/14@336
# ════════════════════════════════════════════════════════════

def cmmd_embed(paths, model, proc, device):
    """Embed images with CLIP ViT-L/14@336."""
    embs = []
    for i in range(0, len(paths), BATCH_SIZE_CLIP):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE_CLIP]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def run_cmmd(anchor_paths, query_paths, device):
    """Compute CMMD (unbiased MMD² with CLIP embeddings)."""
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    logger.info(f"  Loading {CLIP_MODEL_ID}...")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    logger.info(f"  Embedding anchor ({len(anchor_paths)} imgs)...")
    ref_emb = cmmd_embed(anchor_paths, model, proc, device)
    logger.info(f"    anchor shape: {ref_emb.shape}")

    logger.info(f"  Embedding query ({len(query_paths)} imgs)...")
    query_emb = cmmd_embed(query_paths, model, proc, device)
    logger.info(f"    query shape: {query_emb.shape}")

    score = mmd2_unbiased(ref_emb, query_emb, CMMD_GAMMA)

    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    return score


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

def plot_results(results, out_path):
    """Bar chart comparing MMD² distances for each method × dataset."""

    methods = list(results.keys())
    vkitti_scores = [results[m]["virtual_kitti"] for m in methods]
    tracking_scores   = [results[m]["tracking"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#f9f9f9")

    bars1 = ax.bar(x - width/2, vkitti_scores, width, label="Virtual KITTI (sim. domain)",
                   color="#2196F3", edgecolor="white", zorder=3)
    bars2 = ax.bar(x + width/2, tracking_scores, width, label="KITTI Tracking",
                   color="#FF9800", edgecolor="white", zorder=3)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{h:.4e}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_ylabel("MMD² (unbiased)", fontsize=11)
    ax.set_title(
        f"Distribution distance from KITTI ({N_IMAGES} imgs)\n"
        f"Gram-MMD (SD-VAE L11 + StandardScaler) vs CMMD",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def plot_ratio(results, out_path):
    """Bar chart of Tracking/VKITTI ratio."""
    methods = list(results.keys())
    ratios = []
    for m in methods:
        vk = results[m]["virtual_kitti"]
        co = results[m]["tracking"]
        ratios.append(co / vk if vk > 0 else float("inf"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#f9f9f9")
    colors = ["#4CAF50" if r > 1 else "#F44336" for r in ratios]
    bars = ax.bar(methods, ratios, color=colors, edgecolor="white", zorder=3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{r:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Ratio Tracking / Virtual KITTI", fontsize=11)
    ax.set_title(
        "Domain separation ratio (higher = better)\n"
        "Tracking vs Virtual KITTI distance from KITTI anchor",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 90)
    logger.info(f"KITTI distribution comparison  |  device={device}")
    logger.info(f"  Anchor:  {N_IMAGES} KITTI images")
    logger.info(f"  Query 1: {N_IMAGES} Virtual KITTI images")
    logger.info(f"  Query 2: {N_IMAGES} KITTI Tracking images")
    logger.info("=" * 90)

    t_total = time.time()

    # ── Sample images ──────────────────────────────────────
    kitti_paths    = sample_images(KITTI_DIR, N_IMAGES, SEED)
    vkitti_paths   = sample_images(VKITTI_DIR, N_IMAGES, SEED)
    tracking_paths = sample_images(TRACKING_DIR, N_IMAGES, SEED, recursive=True)

    results = {}

    # ── Gram-MMD with SD-VAE L11 ───────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info(f"Gram-MMD : SD-VAE layer {LAYER}  +  StandardScaler")
    logger.info("=" * 90)

    # Extract anchor features (fit scaler on anchor)
    logger.info("  Extracting KITTI anchor features...")
    ext, ref_feats = extract_features(BACKBONE, LAYER, kitti_paths, BATCH_SIZE, device,
                                       fit_transform=True)
    logger.info(f"  Anchor features: {ref_feats.shape}")

    # Extract query features (transform with fitted scaler)
    logger.info("  Extracting Virtual KITTI features...")
    _, vkitti_feats = extract_features(BACKBONE, LAYER, vkitti_paths, BATCH_SIZE, device,
                                        extractor=ext)
    logger.info(f"  Virtual KITTI features: {vkitti_feats.shape}")

    logger.info("  Extracting KITTI Tracking features...")
    _, tracking_feats = extract_features(BACKBONE, LAYER, tracking_paths, BATCH_SIZE, device,
                                      extractor=ext)
    logger.info(f"  Tracking features: {tracking_feats.shape}")

    # Compute median gamma on anchor
    gamma_med = compute_median_gamma(ref_feats)

    # Compute MMD² for each gamma
    for tag, mult in GAMMA_MULTIPLIERS.items():
        gamma = mult * gamma_med
        label = f"Gram {tag} (γ={gamma:.2e})"
        logger.info(f"\n  {label}")

        mmd2_vkitti = mmd2_unbiased(ref_feats, vkitti_feats, gamma)
        mmd2_tracking   = mmd2_unbiased(ref_feats, tracking_feats, gamma)

        logger.info(f"    Virtual KITTI : MMD² = {mmd2_vkitti:.6e}")
        logger.info(f"    Tracking      : MMD² = {mmd2_tracking:.6e}")
        if mmd2_vkitti > 0:
            logger.info(f"    Ratio Tracking/VKITTI = {mmd2_tracking / mmd2_vkitti:.2f}x")

        results[label] = {"virtual_kitti": mmd2_vkitti, "tracking": mmd2_tracking}

    # Cleanup Gram extractor
    ext.cleanup()
    del ext, ref_feats, vkitti_feats, tracking_feats
    torch.cuda.empty_cache()
    gc.collect()

    # ── CMMD ───────────────────────────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info(f"CMMD — {CLIP_MODEL_ID}  gamma={CMMD_GAMMA}")
    logger.info("=" * 90)

    mmd2_vkitti_cmmd = run_cmmd(kitti_paths, vkitti_paths, device)
    mmd2_tracking_cmmd   = run_cmmd(kitti_paths, tracking_paths, device)

    label_cmmd = f"CMMD (γ={CMMD_GAMMA})"
    logger.info(f"  Virtual KITTI : MMD² = {mmd2_vkitti_cmmd:.6e}")
    logger.info(f"  Tracking      : MMD² = {mmd2_tracking_cmmd:.6e}")
    if mmd2_vkitti_cmmd > 0:
        logger.info(f"  Ratio Tracking/VKITTI = {mmd2_tracking_cmmd / mmd2_vkitti_cmmd:.2f}x")

    results[label_cmmd] = {"virtual_kitti": mmd2_vkitti_cmmd, "tracking": mmd2_tracking_cmmd}

    # ── Summary ────────────────────────────────────────────
    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")

    print("\n" + "=" * 90)
    print(f"KITTI distribution comparison  |  {N_IMAGES} images per set")
    print(f"  {'Method':<45}  {'VKITTI MMD²':>14}  {'Track MMD²':>14}  {'Ratio':>8}")
    print("  " + "-" * 82)
    for method, scores in results.items():
        vk = scores["virtual_kitti"]
        co = scores["tracking"]
        ratio = co / vk if vk > 0 else float("inf")
        print(f"  {method:<45}  {vk:>14.6e}  {co:>14.6e}  {ratio:>8.2f}x")
    print("=" * 90)

    # ── Save results ───────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Save CSV
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w") as f:
        f.write("method,virtual_kitti_mmd2,tracking_mmd2,ratio_tracking_vkitti\n")
        for method, scores in results.items():
            vk = scores["virtual_kitti"]
            co = scores["tracking"]
            ratio = co / vk if vk > 0 else float("inf")
            f.write(f"{method},{vk:.6e},{co:.6e},{ratio:.4f}\n")
    logger.info(f"  CSV saved: {csv_path}")

    # Plots
    plot_results(results, plots_dir / "mmd2_comparison.png")
    plot_ratio(results, plots_dir / "domain_separation_ratio.png")

    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
