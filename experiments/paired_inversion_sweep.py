"""
paired_inversion_sweep.py
==========================
Sweep taille d'ancrage + StandardScaler ON/OFF pour trouver un régime
où CMMD inverse mais pas Gram-MMD.

Ancrage : KITTI Tracking (variable: 20, 50, 100, 200, 500, 1000)
Query A : VKITTI clone paired (même taille que ancrage)
Query B : COCO (même taille que ancrage)

2 modes Gram-MMD :
  - AVEC StandardScaler (fit sur ancrage)
  - SANS StandardScaler (features Gram brutes)

+ CMMD (CLIP ViT-L/14@336) à gamma=0.005

Usage:
    .venv/bin/python experiments/paired_inversion_sweep.py
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
VKITTI_DIR   = BASE_DIR / "dataset" / "virtual_kitti"
COCO_DIR     = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR      = BASE_DIR / "results" / "paired_inversion_sweep"

ANCHOR_SIZES = [20, 50, 100, 200, 500, 1000]
SEED         = 42
BATCH_SIZE   = 16

SCENE_TO_SEQ = {
    "Scene01": "0001", "Scene02": "0002", "Scene06": "0006",
    "Scene18": "0018", "Scene20": "0020",
}

BACKBONE = "sd_vae"
LAYER    = 11

CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32


# ════════════════════════════════════════════════════════════
#  PAIRING
# ════════════════════════════════════════════════════════════

def build_all_pairs():
    tracking_paths, vkitti_paths = [], []
    for scene, seq in SCENE_TO_SEQ.items():
        seq_dir = TRACKING_DIR / seq
        if not seq_dir.exists():
            continue
        for rp in sorted(glob_module.glob(str(seq_dir / "*.png"))):
            fn = int(Path(rp).stem)
            vp = VKITTI_DIR / f"{scene}_clone_rgb_{fn:05d}.jpg"
            if vp.exists():
                tracking_paths.append(str(rp))
                vkitti_paths.append(str(vp))
    logger.info(f"  Total paired frames: {len(tracking_paths)}")
    return tracking_paths, vkitti_paths


def subsample_paired(tracking, vkitti, n, seed):
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(tracking), size=min(n, len(tracking)), replace=False).tolist())
    return [tracking[i] for i in idx], [vkitti[i] for i in idx]


def sample_coco(n, seed):
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(all_imgs), size=min(n, len(all_imgs)), replace=False).tolist())
    return [all_imgs[i] for i in idx]


# ════════════════════════════════════════════════════════════
#  DISTANCE UTILS
# ════════════════════════════════════════════════════════════

def squared_l2_matrix(X, Y):
    X_sq = np.einsum("ij,ij->i", X, X)
    Y_sq = np.einsum("ij,ij->i", Y, Y)
    return X_sq[:, None] + Y_sq[None, :] - 2.0 * (X @ Y.T)


def mmd2_unbiased_from_feats(ref, query, gamma):
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


def clip_embed(paths, model, proc, device):
    embs = []
    for i in range(0, len(paths), BATCH_SIZE_CLIP):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE_CLIP]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_total = time.time()

    logger.info("=" * 90)
    logger.info("PAIRED INVERSION SWEEP")
    logger.info(f"  Anchor sizes: {ANCHOR_SIZES}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 90)

    # Build all pairs once
    all_tracking, all_vkitti = build_all_pairs()

    all_rows = []

    # ════════════════════════════════════════════════════════
    #  GRAM-MMD — with and without StandardScaler
    # ════════════════════════════════════════════════════════
    for use_scaler in [True, False]:
        scaler_tag = "with_std" if use_scaler else "no_std"
        logger.info(f"\n{'='*90}")
        logger.info(f"GRAM-MMD  StandardScaler={use_scaler}")
        logger.info("=" * 90)

        for anchor_sz in ANCHOR_SIZES:
            logger.info(f"\n  --- Anchor = {anchor_sz} ---")

            track_sub, vkitti_sub = subsample_paired(all_tracking, all_vkitti, anchor_sz, SEED)
            coco_sub = sample_coco(anchor_sz, SEED)

            ext = FeatureExtractor(
                backbone=BACKBONE, layer=LAYER,
                transform_config={
                    "name": "gram_spatial", "use_gram": use_scaler,
                    "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
                },
                device=device,
            )

            if use_scaler:
                ref_feats = ext.extract(track_sub, fit_transform=True,
                                        batch_size=BATCH_SIZE).astype(np.float32)
                vk_feats = ext.extract(vkitti_sub, fit_transform=False,
                                       batch_size=BATCH_SIZE).astype(np.float32)
                co_feats = ext.extract(coco_sub, fit_transform=False,
                                       batch_size=BATCH_SIZE).astype(np.float32)
            else:
                # Extract raw Gram features without scaler
                # use_gram=False in transform_config means no scaler in transform_features
                # but we still want Gram computation in extract_batch
                # So we set use_gram=True but skip the scaler manually
                ext_gram = FeatureExtractor(
                    backbone=BACKBONE, layer=LAYER,
                    transform_config={
                        "name": "gram_spatial", "use_gram": True,
                        "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
                    },
                    device=device,
                )
                ref_feats = ext_gram.extract(track_sub, fit_transform=False,
                                              batch_size=BATCH_SIZE).astype(np.float32)
                vk_feats = ext_gram.extract(vkitti_sub, fit_transform=False,
                                             batch_size=BATCH_SIZE).astype(np.float32)
                co_feats = ext_gram.extract(coco_sub, fit_transform=False,
                                             batch_size=BATCH_SIZE).astype(np.float32)
                ext_gram.cleanup()
                del ext_gram

            gamma = median_gamma(ref_feats)

            mmd_vk = mmd2_unbiased_from_feats(ref_feats, vk_feats, gamma)
            mmd_co = mmd2_unbiased_from_feats(ref_feats, co_feats, gamma)
            inv = mmd_vk < mmd_co

            row = {
                "method": f"Gram-MMD ({scaler_tag})",
                "anchor_size": anchor_sz,
                "vkitti_mmd2": mmd_vk,
                "coco_mmd2": mmd_co,
                "ratio": mmd_vk / mmd_co if mmd_co > 0 else float("inf"),
                "gamma": gamma,
                "inversion": inv,
            }
            all_rows.append(row)
            logger.info(f"    {scaler_tag}  n={anchor_sz}  VKITTI={mmd_vk:.6e}  COCO={mmd_co:.6e}  "
                         f"ratio={row['ratio']:.3f}  {'INV!' if inv else 'ok'}")

            ext.cleanup()
            del ext, ref_feats, vk_feats, co_feats
            torch.cuda.empty_cache()
            gc.collect()

    # ════════════════════════════════════════════════════════
    #  CMMD
    # ════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info("CMMD (CLIP ViT-L/14@336)")
    logger.info("=" * 90)

    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    for anchor_sz in ANCHOR_SIZES:
        logger.info(f"\n  --- Anchor = {anchor_sz} ---")

        track_sub, vkitti_sub = subsample_paired(all_tracking, all_vkitti, anchor_sz, SEED)
        coco_sub = sample_coco(anchor_sz, SEED)

        ref_emb = clip_embed(track_sub, clip_model, clip_proc, device)
        vk_emb  = clip_embed(vkitti_sub, clip_model, clip_proc, device)
        co_emb  = clip_embed(coco_sub, clip_model, clip_proc, device)

        mmd_vk = mmd2_unbiased_from_feats(ref_emb, vk_emb, CMMD_GAMMA)
        mmd_co = mmd2_unbiased_from_feats(ref_emb, co_emb, CMMD_GAMMA)
        inv = mmd_vk < mmd_co

        row = {
            "method": "CMMD",
            "anchor_size": anchor_sz,
            "vkitti_mmd2": mmd_vk,
            "coco_mmd2": mmd_co,
            "ratio": mmd_vk / mmd_co if mmd_co > 0 else float("inf"),
            "gamma": CMMD_GAMMA,
            "inversion": inv,
        }
        all_rows.append(row)
        logger.info(f"    CMMD  n={anchor_sz}  VKITTI={mmd_vk:.6e}  COCO={mmd_co:.6e}  "
                     f"ratio={row['ratio']:.3f}  {'INV!' if inv else 'ok'}")

    del clip_model, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  SUMMARY + PLOTS
    # ════════════════════════════════════════════════════════
    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")

    # Terminal summary
    print("\n" + "=" * 100)
    print("PAIRED INVERSION SWEEP — KITTI Tracking ↔ VKITTI clone vs COCO")
    print("=" * 100)
    print(f"  {'Méthode':<25s}  {'N':>5s}  {'VKITTI':>12s}  {'COCO':>12s}  {'Ratio':>7s}  {'INV?':>5s}")
    print("  " + "-" * 75)
    for r in all_rows:
        print(f"  {r['method']:<25s}  {r['anchor_size']:>5d}  {r['vkitti_mmd2']:>12.4e}  "
              f"{r['coco_mmd2']:>12.4e}  {r['ratio']:>7.3f}  {'OUI' if r['inversion'] else 'non':>5s}")
    print("=" * 100)

    # Save CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "anchor_size", "vkitti_mmd2",
                                          "coco_mmd2", "ratio", "gamma", "inversion"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})
    logger.info(f"  CSV: {csv_path}")

    # ── Plot: ratio vs anchor size ────────────────────────
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    methods = sorted(set(r["method"] for r in all_rows))
    method_colors = {
        "Gram-MMD (with_std)": "#2196F3",
        "Gram-MMD (no_std)":   "#F44336",
        "CMMD":                "#4CAF50",
    }
    method_markers = {
        "Gram-MMD (with_std)": "o",
        "Gram-MMD (no_std)":   "s",
        "CMMD":                "D",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Ratio VKITTI/COCO vs anchor size
    ax1.set_facecolor("#f9f9f9")
    for m in methods:
        rows_m = sorted([r for r in all_rows if r["method"] == m], key=lambda r: r["anchor_size"])
        xs = [r["anchor_size"] for r in rows_m]
        ys = [r["ratio"] for r in rows_m]
        ax1.plot(xs, ys, marker=method_markers.get(m, "o"), color=method_colors.get(m, "gray"),
                 label=m, linewidth=2, markersize=8)

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Seuil (ratio=1)")
    ax1.fill_between([min(ANCHOR_SIZES)*0.8, max(ANCHOR_SIZES)*1.2], 0, 1,
                     alpha=0.1, color="red", label="Zone inversion")
    ax1.set_xscale("log")
    ax1.set_xlabel("Taille ancrage (N)", fontsize=11)
    ax1.set_ylabel("Ratio MMD²(VKITTI) / MMD²(COCO)", fontsize=11)
    ax1.set_title("Ratio VKITTI/COCO vs taille ancrage\n<1 = inversion (VKITTI plus proche)",
                  fontsize=11, fontweight="bold")
    ax1.set_xticks(ANCHOR_SIZES)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Panel 2: MMD² values vs anchor size
    ax2.set_facecolor("#f9f9f9")
    for m in methods:
        rows_m = sorted([r for r in all_rows if r["method"] == m], key=lambda r: r["anchor_size"])
        xs = [r["anchor_size"] for r in rows_m]
        vk = [r["vkitti_mmd2"] for r in rows_m]
        co = [r["coco_mmd2"] for r in rows_m]
        color = method_colors.get(m, "gray")
        ax2.plot(xs, vk, marker=method_markers.get(m, "o"), color=color,
                 linewidth=2, markersize=7, linestyle="-", label=f"{m} — VKITTI")
        ax2.plot(xs, co, marker=method_markers.get(m, "o"), color=color,
                 linewidth=2, markersize=7, linestyle="--", label=f"{m} — COCO")

    ax2.set_xscale("log")
    ax2.set_xlabel("Taille ancrage (N)", fontsize=11)
    ax2.set_ylabel("MMD² (unbiased)", fontsize=11)
    ax2.set_title("MMD² absolu vs taille ancrage\nTrait plein = VKITTI, tirets = COCO",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(ANCHOR_SIZES)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.legend(fontsize=8, ncol=2)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(plots_dir / "sweep_anchor_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'sweep_anchor_sizes.png'}")

    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
