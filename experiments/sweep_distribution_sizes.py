"""
sweep_distribution_sizes.py
============================
Sweep anchor and query distribution sizes to study MMD² stability.

Anchor: KITTI images
Query distributions: Virtual KITTI, KITTI Tracking, COCO

Experiment grid:
  - Anchor sizes:  [1000, 200, 50, 20]
  - Query sizes:   [5, 10, 20, 50, 100, 200, 500, 1000]
  - Methods:       Gram-MMD (SD-VAE L11, gamma_med) + CMMD

For each (anchor_size, query_size, method, dataset) we compute MMD² unbiased.

Usage:
    .venv/bin/python experiments/sweep_distribution_sizes.py
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

KITTI_DIR    = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR   = BASE_DIR / "dataset" / "virtual_kitti"
TRACKING_DIR = BASE_DIR / "dataset" / "data_tracking_image_2"
COCO_DIR     = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR      = BASE_DIR / "results" / "sweep_distribution_sizes"

ANCHOR_SIZES = [1000, 200, 50, 20]
QUERY_SIZES  = [5, 10, 20, 50, 100, 200, 500, 1000]

SEED         = 42
BATCH_SIZE   = 16

# Gram-MMD
BACKBONE = "sd_vae"
LAYER    = 11

# CMMD
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32

DATASETS = {
    "Virtual KITTI":  {"dir": VKITTI_DIR,   "recursive": False},
    "KITTI Tracking": {"dir": TRACKING_DIR, "recursive": True},
    "COCO":           {"dir": COCO_DIR,     "recursive": False},
}


# ════════════════════════════════════════════════════════════
#  UTILS
# ════════════════════════════════════════════════════════════

def list_all_images(directory, recursive=False):
    """List all images in directory, sorted deterministically."""
    if recursive:
        imgs = sorted(
            glob_module.glob(str(directory / "**" / "*.png"), recursive=True) +
            glob_module.glob(str(directory / "**" / "*.jpg"), recursive=True)
        )
    else:
        imgs = sorted(
            glob_module.glob(str(directory / "*.png")) +
            glob_module.glob(str(directory / "*.jpg")) +
            glob_module.glob(str(directory / "*.jpeg"))
        )
    if not imgs:
        raise FileNotFoundError(f"No images in {directory}")
    return imgs


def subsample(paths, n, seed):
    """Deterministic subsample of n paths."""
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(paths), size=min(n, len(paths)), replace=False).tolist())
    return [paths[i] for i in idx]


def compute_median_gamma(features):
    f_sq = np.einsum("ij,ij->i", features, features)
    D = f_sq[:, None] + f_sq[None, :] - 2.0 * (features @ features.T)
    triu = D[np.triu_indices(len(features), k=1)]
    median_d2 = float(np.median(triu))
    return 1.0 / (2.0 * median_d2 + 1e-8)


def mmd2_unbiased(ref, query, gamma):
    n_r, n_q = len(ref), len(query)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    query_sq = np.einsum("ij,ij->i", query, query)

    D_rr = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
    K_rr = np.exp(-gamma * D_rr)
    term_rr = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))

    D_qq = query_sq[:, None] + query_sq[None, :] - 2.0 * (query @ query.T)
    K_qq = np.exp(-gamma * D_qq)
    term_qq = (np.sum(K_qq) - np.trace(K_qq)) / (n_q * (n_q - 1))

    D_rq = ref_sq[:, None] + query_sq[None, :] - 2.0 * (ref @ query.T)
    K_rq = np.exp(-gamma * D_rq)
    term_rq = 2.0 * np.sum(K_rq) / (n_r * n_q)

    return float(term_rr + term_qq - term_rq)


# ════════════════════════════════════════════════════════════
#  CMMD helpers
# ════════════════════════════════════════════════════════════

def load_clip_model(device):
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, proc


def cmmd_embed(paths, model, proc, device):
    embs = []
    for i in range(0, len(paths), BATCH_SIZE_CLIP):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE_CLIP]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

DATASET_COLORS = {
    "Virtual KITTI":  "#2196F3",
    "KITTI Tracking": "#FF9800",
    "COCO":           "#4CAF50",
}
DATASET_MARKERS = {
    "Virtual KITTI":  "o",
    "KITTI Tracking": "s",
    "COCO":           "D",
}


def plot_sweep(all_rows, plots_dir):
    """One figure per (anchor_size, method): MMD² vs query_size, one line per dataset."""

    # Group rows
    groups = {}
    for r in all_rows:
        key = (r["anchor_size"], r["method"])
        groups.setdefault(key, []).append(r)

    for (anchor_sz, method), rows in groups.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#f9f9f9")

        for ds_name in DATASETS:
            ds_rows = sorted([r for r in rows if r["dataset"] == ds_name],
                             key=lambda r: r["query_size"])
            if not ds_rows:
                continue
            xs = [r["query_size"] for r in ds_rows]
            ys = [r["mmd2"] for r in ds_rows]
            ax.plot(xs, ys,
                    marker=DATASET_MARKERS[ds_name],
                    color=DATASET_COLORS[ds_name],
                    label=ds_name, linewidth=2, markersize=7)

        ax.set_xscale("log")
        ax.set_xlabel("Query distribution size (N_query)", fontsize=11)
        ax.set_ylabel("MMD² (unbiased)", fontsize=11)
        ax.set_title(
            f"{method}  |  Anchor = {anchor_sz} KITTI images\n"
            f"MMD² vs query size",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(QUERY_SIZES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

        plt.tight_layout()
        fname = f"sweep_anchor{anchor_sz}_{method.replace(' ', '_').lower()}.png"
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {plots_dir / fname}")

    # Summary figure: all anchor sizes for Gram, one subplot per anchor_size
    gram_rows = [r for r in all_rows if r["method"] == "Gram-MMD"]
    if gram_rows:
        anchor_sizes_found = sorted(set(r["anchor_size"] for r in gram_rows), reverse=True)
        n_panels = len(anchor_sizes_found)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
        if n_panels == 1:
            axes = [axes]

        for ax, anchor_sz in zip(axes, anchor_sizes_found):
            ax.set_facecolor("#f9f9f9")
            for ds_name in DATASETS:
                ds_rows = sorted(
                    [r for r in gram_rows if r["anchor_size"] == anchor_sz and r["dataset"] == ds_name],
                    key=lambda r: r["query_size"])
                if not ds_rows:
                    continue
                xs = [r["query_size"] for r in ds_rows]
                ys = [r["mmd2"] for r in ds_rows]
                ax.plot(xs, ys,
                        marker=DATASET_MARKERS[ds_name],
                        color=DATASET_COLORS[ds_name],
                        label=ds_name, linewidth=2, markersize=6)

            ax.set_xscale("log")
            ax.set_title(f"Anchor = {anchor_sz}", fontsize=11, fontweight="bold")
            ax.set_xlabel("N_query", fontsize=10)
            ax.set_xticks(QUERY_SIZES)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.yaxis.grid(True, alpha=0.3)
            ax.xaxis.grid(True, alpha=0.2)
            ax.set_axisbelow(True)

        axes[0].set_ylabel("MMD² (unbiased)", fontsize=11)
        axes[0].legend(fontsize=9)
        fig.suptitle("Gram-MMD (SD-VAE L11) — MMD² vs query size for different anchor sizes",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(plots_dir / "summary_gram_all_anchors.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {plots_dir / 'summary_gram_all_anchors.png'}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_total = time.time()

    logger.info("=" * 90)
    logger.info("Sweep distribution sizes")
    logger.info(f"  Anchor sizes: {ANCHOR_SIZES}")
    logger.info(f"  Query sizes:  {QUERY_SIZES}")
    logger.info(f"  Datasets:     {list(DATASETS.keys())}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 90)

    # ── Pre-list all images once ───────────────────────────
    all_kitti = list_all_images(KITTI_DIR)
    logger.info(f"  KITTI pool: {len(all_kitti)} images")

    all_query_pools = {}
    for ds_name, ds_cfg in DATASETS.items():
        pool = list_all_images(ds_cfg["dir"], recursive=ds_cfg["recursive"])
        all_query_pools[ds_name] = pool
        logger.info(f"  {ds_name} pool: {len(pool)} images")

    all_rows = []

    # ════════════════════════════════════════════════════════
    #  GRAM-MMD sweeps
    # ════════════════════════════════════════════════════════
    for anchor_sz in ANCHOR_SIZES:
        logger.info(f"\n{'='*90}")
        logger.info(f"GRAM-MMD  |  Anchor = {anchor_sz} KITTI images")
        logger.info("=" * 90)

        anchor_paths = subsample(all_kitti, anchor_sz, SEED)

        # Extract anchor features (fit scaler)
        ext = FeatureExtractor(
            backbone=BACKBONE, layer=LAYER,
            transform_config={
                "name": "gram_spatial", "use_gram": True,
                "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
            },
            device=device,
        )
        ref_feats = ext.extract(anchor_paths, fit_transform=True,
                                batch_size=BATCH_SIZE).astype(np.float32)
        gamma_med = compute_median_gamma(ref_feats)
        logger.info(f"  Anchor features: {ref_feats.shape},  gamma_med = {gamma_med:.4e}")

        # Extract ALL query images at max size, then subsample features
        for ds_name, ds_cfg in DATASETS.items():
            pool = all_query_pools[ds_name]
            max_q = min(max(QUERY_SIZES), len(pool))
            query_paths_max = subsample(pool, max_q, SEED)

            logger.info(f"\n  Extracting {ds_name} features ({max_q} imgs)...")
            query_feats_max = ext.extract(query_paths_max, fit_transform=False,
                                          batch_size=BATCH_SIZE).astype(np.float32)

            for q_sz in QUERY_SIZES:
                if q_sz > max_q:
                    continue
                # Take first q_sz features (already deterministically sampled)
                q_feats = query_feats_max[:q_sz]
                mmd2 = mmd2_unbiased(ref_feats, q_feats, gamma_med)

                row = {
                    "method": "Gram-MMD",
                    "anchor_size": anchor_sz,
                    "query_size": q_sz,
                    "dataset": ds_name,
                    "gamma": gamma_med,
                    "mmd2": mmd2,
                }
                all_rows.append(row)
                logger.info(f"    anchor={anchor_sz}  query={q_sz:>4d}  {ds_name:<16s}  MMD²={mmd2:.6e}")

        ext.cleanup()
        del ext, ref_feats
        torch.cuda.empty_cache()
        gc.collect()

    # ════════════════════════════════════════════════════════
    #  CMMD sweeps
    # ════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info("CMMD sweeps")
    logger.info("=" * 90)

    clip_model, clip_proc = load_clip_model(device)

    for anchor_sz in ANCHOR_SIZES:
        logger.info(f"\n  CMMD  |  Anchor = {anchor_sz} KITTI images")
        anchor_paths = subsample(all_kitti, anchor_sz, SEED)
        ref_emb = cmmd_embed(anchor_paths, clip_model, clip_proc, device)
        logger.info(f"    Anchor embeddings: {ref_emb.shape}")

        for ds_name in DATASETS:
            pool = all_query_pools[ds_name]
            max_q = min(max(QUERY_SIZES), len(pool))
            query_paths_max = subsample(pool, max_q, SEED)
            query_emb_max = cmmd_embed(query_paths_max, clip_model, clip_proc, device)

            for q_sz in QUERY_SIZES:
                if q_sz > max_q:
                    continue
                q_emb = query_emb_max[:q_sz]
                mmd2 = mmd2_unbiased(ref_emb, q_emb, CMMD_GAMMA)

                row = {
                    "method": "CMMD",
                    "anchor_size": anchor_sz,
                    "query_size": q_sz,
                    "dataset": ds_name,
                    "gamma": CMMD_GAMMA,
                    "mmd2": mmd2,
                }
                all_rows.append(row)
                logger.info(f"    anchor={anchor_sz}  query={q_sz:>4d}  {ds_name:<16s}  MMD²={mmd2:.6e}")

    del clip_model, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  SAVE + PLOT
    # ════════════════════════════════════════════════════════
    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "anchor_size", "query_size",
                                          "dataset", "gamma", "mmd2"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})
    logger.info(f"  CSV saved: {csv_path}")

    # Terminal summary
    print("\n" + "=" * 100)
    print(f"  {'Method':<10}  {'Anchor':>6}  {'Query':>5}  {'Dataset':<16}  {'MMD²':>14}")
    print("  " + "-" * 90)
    for r in all_rows:
        print(f"  {r['method']:<10}  {r['anchor_size']:>6}  {r['query_size']:>5}  "
              f"{r['dataset']:<16}  {r['mmd2']:>14.6e}")
    print("=" * 100)

    # Plots
    plot_sweep(all_rows, plots_dir)

    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
