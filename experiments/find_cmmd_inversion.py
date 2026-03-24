"""
find_cmmd_inversion.py
=======================
Cherche un régime où CMMD inverse le classement VKITTI/COCO
mais pas Gram-MMD.

Hypothèse : CLIP encode la sémantique → Virtual KITTI (même contenu
que KITTI) pourrait être vu comme plus proche que COCO (contenu différent).
Les features Gram capturent la texture → VKITTI (synthétique) reste loin.

Expérience :
  - Anchor : 1000 KITTI
  - Query A : 500 Virtual KITTI  (même sémantique, texture synthétique)
  - Query B : 500 COCO           (sémantique différente, texture réelle)

  1) Gram-MMD (SD-VAE L11) : sweep gamma sur grille log
  2) CMMD (CLIP ViT-L/14@336) : sweep gamma sur grille log
  3) Cosine distance brute sur embeddings CLIP (pas de kernel)
  4) FID sur embeddings CLIP

Usage:
    .venv/bin/python experiments/find_cmmd_inversion.py
"""

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

KITTI_DIR  = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR = BASE_DIR / "dataset" / "virtual_kitti"
COCO_DIR   = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR    = BASE_DIR / "results" / "cmmd_inversion"

N_ANCHOR = 1000
N_QUERY  = 500
SEED     = 42
BATCH_SIZE = 16

# Gram-MMD
BACKBONE = "sd_vae"
LAYER    = 11

# CLIP
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
BATCH_SIZE_CLIP = 32

# Gamma sweep (log grid)
GAMMA_GRID_GRAM = np.logspace(-8, -3, 60)
GAMMA_GRID_CLIP = np.logspace(-5, 2, 60)


# ════════════════════════════════════════════════════════════
#  UTILS
# ════════════════════════════════════════════════════════════

def list_images(directory, recursive=False):
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
    return imgs


def subsample(paths, n, seed):
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(paths), size=min(n, len(paths)), replace=False).tolist())
    return [paths[i] for i in idx]


def squared_l2_matrix(X, Y):
    """Pairwise squared L2 distances."""
    X_sq = np.einsum("ij,ij->i", X, X)
    Y_sq = np.einsum("ij,ij->i", Y, Y)
    return X_sq[:, None] + Y_sq[None, :] - 2.0 * (X @ Y.T)


def mmd2_from_distances(D_rr, D_qq, D_rq, gamma):
    """Compute unbiased MMD² from precomputed squared distance matrices."""
    n_r = D_rr.shape[0]
    n_q = D_qq.shape[0]

    K_rr = np.exp(-gamma * D_rr)
    K_qq = np.exp(-gamma * D_qq)
    K_rq = np.exp(-gamma * D_rq)

    term_rr = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))
    term_qq = (np.sum(K_qq) - np.trace(K_qq)) / (n_q * (n_q - 1))
    term_rq = 2.0 * np.sum(K_rq) / (n_r * n_q)

    return float(term_rr + term_qq - term_rq)


def compute_fid(feat_ref, feat_query):
    """Simple FID between two feature sets."""
    mu_r = np.mean(feat_ref, axis=0)
    mu_q = np.mean(feat_query, axis=0)

    diff = mu_r - mu_q
    mean_term = np.sum(diff ** 2)

    n_r = feat_ref.shape[0]
    n_q = feat_query.shape[0]
    cov_r = np.cov(feat_ref, rowvar=False) + np.eye(feat_ref.shape[1]) * 1e-6
    cov_q = np.cov(feat_query, rowvar=False) + np.eye(feat_query.shape[1]) * 1e-6

    from scipy import linalg
    covmean, _ = linalg.sqrtm(cov_r @ cov_q, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(mean_term + np.trace(cov_r + cov_q - 2 * covmean))


def cosine_distance_distributions(feat_ref, feat_query):
    """Mean pairwise cosine distance between two sets."""
    # Normalize
    ref_n = feat_ref / (np.linalg.norm(feat_ref, axis=1, keepdims=True) + 1e-8)
    q_n = feat_query / (np.linalg.norm(feat_query, axis=1, keepdims=True) + 1e-8)
    # Mean cosine similarity
    cos_sim = np.mean(ref_n @ q_n.T)
    return float(1.0 - cos_sim)


# ════════════════════════════════════════════════════════════
#  CLIP embedding
# ════════════════════════════════════════════════════════════

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
    logger.info("Recherche d'inversion CMMD vs Gram-MMD")
    logger.info(f"  Anchor: {N_ANCHOR} KITTI  |  Query: {N_QUERY} VKITTI + {N_QUERY} COCO")
    logger.info(f"  Device: {device}")
    logger.info("=" * 90)

    # ── Sample images ──────────────────────────────────────
    kitti_paths  = subsample(list_images(KITTI_DIR), N_ANCHOR, SEED)
    vkitti_paths = subsample(list_images(VKITTI_DIR), N_QUERY, SEED)
    coco_paths   = subsample(list_images(COCO_DIR), N_QUERY, SEED)

    logger.info(f"  KITTI:  {len(kitti_paths)} images")
    logger.info(f"  VKITTI: {len(vkitti_paths)} images")
    logger.info(f"  COCO:   {len(coco_paths)} images")

    # ════════════════════════════════════════════════════════
    #  PART 1: Gram-MMD gamma sweep
    # ════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info("PART 1: Gram-MMD (SD-VAE L11) — gamma sweep")
    logger.info("=" * 90)

    ext = FeatureExtractor(
        backbone=BACKBONE, layer=LAYER,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )

    gram_ref    = ext.extract(kitti_paths, fit_transform=True, batch_size=BATCH_SIZE).astype(np.float32)
    gram_vkitti = ext.extract(vkitti_paths, fit_transform=False, batch_size=BATCH_SIZE).astype(np.float32)
    gram_coco   = ext.extract(coco_paths, fit_transform=False, batch_size=BATCH_SIZE).astype(np.float32)
    ext.cleanup()
    del ext
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"  Gram features: ref={gram_ref.shape}, vkitti={gram_vkitti.shape}, coco={gram_coco.shape}")

    # Precompute distance matrices
    D_rr_gram = squared_l2_matrix(gram_ref, gram_ref)
    D_vk_vk   = squared_l2_matrix(gram_vkitti, gram_vkitti)
    D_co_co   = squared_l2_matrix(gram_coco, gram_coco)
    D_r_vk    = squared_l2_matrix(gram_ref, gram_vkitti)
    D_r_co    = squared_l2_matrix(gram_ref, gram_coco)

    # Median heuristic
    triu_gram = D_rr_gram[np.triu_indices(len(gram_ref), k=1)]
    median_d2_gram = float(np.median(triu_gram))
    gamma_med_gram = 1.0 / (2.0 * median_d2_gram + 1e-8)
    logger.info(f"  Gram median heuristic: gamma_med = {gamma_med_gram:.4e}")

    gram_mmd_vkitti = []
    gram_mmd_coco   = []
    for g in GAMMA_GRID_GRAM:
        gram_mmd_vkitti.append(mmd2_from_distances(D_rr_gram, D_vk_vk, D_r_vk, g))
        gram_mmd_coco.append(mmd2_from_distances(D_rr_gram, D_co_co, D_r_co, g))

    del D_rr_gram, D_vk_vk, D_co_co, D_r_vk, D_r_co
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  PART 2: CLIP embedding + gamma sweep
    # ════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info("PART 2: CLIP ViT-L/14@336 — gamma sweep + cosine + FID")
    logger.info("=" * 90)

    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    clip_ref    = clip_embed(kitti_paths, clip_model, clip_proc, device)
    clip_vkitti = clip_embed(vkitti_paths, clip_model, clip_proc, device)
    clip_coco   = clip_embed(coco_paths, clip_model, clip_proc, device)
    del clip_model, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"  CLIP features: ref={clip_ref.shape}, vkitti={clip_vkitti.shape}, coco={clip_coco.shape}")

    # Precompute distance matrices
    D_rr_clip = squared_l2_matrix(clip_ref, clip_ref)
    D_vk_vk_c = squared_l2_matrix(clip_vkitti, clip_vkitti)
    D_co_co_c = squared_l2_matrix(clip_coco, clip_coco)
    D_r_vk_c  = squared_l2_matrix(clip_ref, clip_vkitti)
    D_r_co_c  = squared_l2_matrix(clip_ref, clip_coco)

    # Median heuristic for CLIP
    triu_clip = D_rr_clip[np.triu_indices(len(clip_ref), k=1)]
    median_d2_clip = float(np.median(triu_clip))
    gamma_med_clip = 1.0 / (2.0 * median_d2_clip + 1e-8)
    logger.info(f"  CLIP median heuristic: gamma_med = {gamma_med_clip:.4e}")

    clip_mmd_vkitti = []
    clip_mmd_coco   = []
    for g in GAMMA_GRID_CLIP:
        clip_mmd_vkitti.append(mmd2_from_distances(D_rr_clip, D_vk_vk_c, D_r_vk_c, g))
        clip_mmd_coco.append(mmd2_from_distances(D_rr_clip, D_co_co_c, D_r_co_c, g))

    # Extra metrics on CLIP embeddings
    cosine_vkitti = cosine_distance_distributions(clip_ref, clip_vkitti)
    cosine_coco   = cosine_distance_distributions(clip_ref, clip_coco)
    logger.info(f"  Cosine distance: VKITTI={cosine_vkitti:.6f}  COCO={cosine_coco:.6f}")

    fid_vkitti = compute_fid(clip_ref, clip_vkitti)
    fid_coco   = compute_fid(clip_ref, clip_coco)
    logger.info(f"  FID (CLIP):      VKITTI={fid_vkitti:.4f}  COCO={fid_coco:.4f}")

    del D_rr_clip, D_vk_vk_c, D_co_co_c, D_r_vk_c, D_r_co_c
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  ANALYSIS: find crossover
    # ════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info("ANALYSIS")
    logger.info("=" * 90)

    # Gram: find gammas where VKITTI > COCO
    gram_diff = np.array(gram_mmd_vkitti) - np.array(gram_mmd_coco)
    gram_vkitti_farther = gram_diff > 0

    # CLIP: find gammas where VKITTI < COCO (inversion!)
    clip_diff = np.array(clip_mmd_vkitti) - np.array(clip_mmd_coco)
    clip_vkitti_closer = clip_diff < 0  # inversion

    logger.info(f"  Gram: VKITTI farther than COCO for {np.sum(gram_vkitti_farther)}/{len(GAMMA_GRID_GRAM)} gammas")
    logger.info(f"  CLIP: VKITTI CLOSER than COCO (inversion) for {np.sum(clip_vkitti_closer)}/{len(GAMMA_GRID_CLIP)} gammas")

    # Find best inversion point for CLIP
    if np.any(clip_vkitti_closer):
        best_idx = np.argmin(clip_diff)
        logger.info(f"  Best CLIP inversion at gamma={GAMMA_GRID_CLIP[best_idx]:.4e}: "
                     f"VKITTI={clip_mmd_vkitti[best_idx]:.6e}  COCO={clip_mmd_coco[best_idx]:.6e}")

    # Summary
    print("\n" + "=" * 90)
    print("RÉSUMÉ — Distances KITTI → VKITTI vs KITTI → COCO")
    print("=" * 90)
    print(f"\n  Cosine distance (CLIP):")
    print(f"    VKITTI: {cosine_vkitti:.6f}")
    print(f"    COCO:   {cosine_coco:.6f}")
    print(f"    → {'VKITTI plus proche (INVERSION!)' if cosine_vkitti < cosine_coco else 'COCO plus proche'}")
    print(f"\n  FID (CLIP embeddings):")
    print(f"    VKITTI: {fid_vkitti:.4f}")
    print(f"    COCO:   {fid_coco:.4f}")
    print(f"    → {'VKITTI plus proche (INVERSION!)' if fid_vkitti < fid_coco else 'COCO plus proche'}")
    print(f"\n  CMMD (γ=0.005, original):")
    # Find closest gamma to 0.005
    idx_005 = np.argmin(np.abs(GAMMA_GRID_CLIP - 0.005))
    print(f"    VKITTI: {clip_mmd_vkitti[idx_005]:.6e}")
    print(f"    COCO:   {clip_mmd_coco[idx_005]:.6e}")
    print(f"\n  Gram-MMD (γ=median):")
    idx_med = np.argmin(np.abs(GAMMA_GRID_GRAM - gamma_med_gram))
    print(f"    VKITTI: {gram_mmd_vkitti[idx_med]:.6e}")
    print(f"    COCO:   {gram_mmd_coco[idx_med]:.6e}")
    print("=" * 90)

    # ════════════════════════════════════════════════════════
    #  PLOTS
    # ════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── Plot 1: Gram-MMD gamma sweep ──────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.set_facecolor("#f9f9f9")
    ax1.plot(GAMMA_GRID_GRAM, gram_mmd_vkitti, "-o", color="#2196F3",
             label="Virtual KITTI", markersize=3, linewidth=1.5)
    ax1.plot(GAMMA_GRID_GRAM, gram_mmd_coco, "-s", color="#4CAF50",
             label="COCO", markersize=3, linewidth=1.5)
    ax1.axvline(gamma_med_gram, color="red", linestyle="--", alpha=0.7,
                label=f"γ_median = {gamma_med_gram:.2e}")
    ax1.set_xscale("log")
    ax1.set_xlabel("γ (gamma)", fontsize=11)
    ax1.set_ylabel("MMD² (unbiased)", fontsize=11)
    ax1.set_title("Gram-MMD (SD-VAE L11) — gamma sweep\n"
                  "VKITTI devrait rester AU-DESSUS de COCO (pas d'inversion)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Shade inversion zone if any
    gram_inv = np.array(gram_mmd_vkitti) < np.array(gram_mmd_coco)
    if np.any(gram_inv):
        for i in range(len(gram_inv)):
            if gram_inv[i]:
                ax1.axvspan(GAMMA_GRID_GRAM[max(0, i-1)], GAMMA_GRID_GRAM[min(len(gram_inv)-1, i+1)],
                           alpha=0.2, color="red")

    # ── Plot 2: CLIP gamma sweep ─────────────────────────
    ax2.set_facecolor("#f9f9f9")
    ax2.plot(GAMMA_GRID_CLIP, clip_mmd_vkitti, "-o", color="#2196F3",
             label="Virtual KITTI", markersize=3, linewidth=1.5)
    ax2.plot(GAMMA_GRID_CLIP, clip_mmd_coco, "-s", color="#4CAF50",
             label="COCO", markersize=3, linewidth=1.5)
    ax2.axvline(gamma_med_clip, color="red", linestyle="--", alpha=0.7,
                label=f"γ_median = {gamma_med_clip:.2e}")
    ax2.axvline(0.005, color="purple", linestyle=":", alpha=0.7,
                label="γ=0.005 (CMMD original)")
    ax2.set_xscale("log")
    ax2.set_xlabel("γ (gamma)", fontsize=11)
    ax2.set_ylabel("MMD² (unbiased)", fontsize=11)
    ax2.set_title("CMMD (CLIP ViT-L/14@336) — gamma sweep\n"
                  "Cherche une zone où VKITTI PASSE EN-DESSOUS de COCO (inversion)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # Shade inversion zone
    clip_inv = np.array(clip_mmd_vkitti) < np.array(clip_mmd_coco)
    if np.any(clip_inv):
        starts = []
        in_zone = False
        for i in range(len(clip_inv)):
            if clip_inv[i] and not in_zone:
                starts.append(i)
                in_zone = True
            elif not clip_inv[i] and in_zone:
                ax2.axvspan(GAMMA_GRID_CLIP[starts[-1]], GAMMA_GRID_CLIP[i-1],
                           alpha=0.25, color="red", label="INVERSION" if len(starts) == 1 else "")
                in_zone = False
        if in_zone:
            ax2.axvspan(GAMMA_GRID_CLIP[starts[-1]], GAMMA_GRID_CLIP[-1],
                       alpha=0.25, color="red", label="INVERSION" if len(starts) == 1 else "")
        ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(plots_dir / "gamma_sweep_gram_vs_clip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'gamma_sweep_gram_vs_clip.png'}")

    # ── Plot 3: Difference curves ─────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.set_facecolor("#f9f9f9")
    ax1.plot(GAMMA_GRID_GRAM, gram_diff, "-", color="#2196F3", linewidth=2)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.axvline(gamma_med_gram, color="red", linestyle="--", alpha=0.7)
    ax1.fill_between(GAMMA_GRID_GRAM, gram_diff, 0,
                     where=np.array(gram_diff) < 0, alpha=0.3, color="red",
                     label="INVERSION (VKITTI < COCO)")
    ax1.fill_between(GAMMA_GRID_GRAM, gram_diff, 0,
                     where=np.array(gram_diff) > 0, alpha=0.3, color="green",
                     label="Normal (VKITTI > COCO)")
    ax1.set_xscale("log")
    ax1.set_xlabel("γ", fontsize=11)
    ax1.set_ylabel("MMD²(VKITTI) − MMD²(COCO)", fontsize=11)
    ax1.set_title("Gram-MMD : différence", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    ax2.set_facecolor("#f9f9f9")
    ax2.plot(GAMMA_GRID_CLIP, clip_diff, "-", color="#FF9800", linewidth=2)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.axvline(gamma_med_clip, color="red", linestyle="--", alpha=0.7)
    ax2.axvline(0.005, color="purple", linestyle=":", alpha=0.7)
    ax2.fill_between(GAMMA_GRID_CLIP, clip_diff, 0,
                     where=np.array(clip_diff) < 0, alpha=0.3, color="red",
                     label="INVERSION (VKITTI < COCO)")
    ax2.fill_between(GAMMA_GRID_CLIP, clip_diff, 0,
                     where=np.array(clip_diff) > 0, alpha=0.3, color="green",
                     label="Normal (VKITTI > COCO)")
    ax2.set_xscale("log")
    ax2.set_xlabel("γ", fontsize=11)
    ax2.set_ylabel("MMD²(VKITTI) − MMD²(COCO)", fontsize=11)
    ax2.set_title("CMMD (CLIP) : différence", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(plots_dir / "difference_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'difference_curves.png'}")

    # ── Plot 4: Bar chart — scalar metrics ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("Cosine dist (CLIP)", cosine_vkitti, cosine_coco),
        ("FID (CLIP)", fid_vkitti, fid_coco),
        (f"Gram-MMD (γ_med)", gram_mmd_vkitti[idx_med], gram_mmd_coco[idx_med]),
    ]

    for ax, (name, val_vk, val_co) in zip(axes, metrics):
        ax.set_facecolor("#f9f9f9")
        x = [0, 1]
        bars = ax.bar(x, [val_vk, val_co],
                      color=["#2196F3", "#4CAF50"], edgecolor="white", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["Virtual KITTI", "COCO"], fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{h:.4f}" if h > 0.01 else f"{h:.2e}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        if val_vk < val_co:
            ax.set_title(name + "\nINVERSION!", fontsize=11, fontweight="bold", color="red")

    fig.suptitle("Distance KITTI → VKITTI vs KITTI → COCO\n"
                 "(VKITTI devrait être plus loin si la métrique capture la texture)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plots_dir / "scalar_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'scalar_metrics_comparison.png'}")

    # ── Save CSV ──────────────────────────────────────────
    csv_path = OUT_DIR / "gamma_sweep.csv"
    with open(csv_path, "w") as f:
        f.write("method,gamma,mmd2_vkitti,mmd2_coco,diff,inversion\n")
        for i, g in enumerate(GAMMA_GRID_GRAM):
            inv = gram_mmd_vkitti[i] < gram_mmd_coco[i]
            f.write(f"Gram-MMD,{g:.6e},{gram_mmd_vkitti[i]:.6e},{gram_mmd_coco[i]:.6e},"
                    f"{gram_diff[i]:.6e},{inv}\n")
        for i, g in enumerate(GAMMA_GRID_CLIP):
            inv = clip_mmd_vkitti[i] < clip_mmd_coco[i]
            f.write(f"CMMD,{g:.6e},{clip_mmd_vkitti[i]:.6e},{clip_mmd_coco[i]:.6e},"
                    f"{clip_diff[i]:.6e},{inv}\n")
    logger.info(f"  CSV saved: {csv_path}")

    scalar_path = OUT_DIR / "scalar_metrics.csv"
    with open(scalar_path, "w") as f:
        f.write("metric,vkitti,coco,inversion\n")
        f.write(f"cosine_clip,{cosine_vkitti:.6f},{cosine_coco:.6f},{cosine_vkitti < cosine_coco}\n")
        f.write(f"fid_clip,{fid_vkitti:.6f},{fid_coco:.6f},{fid_vkitti < fid_coco}\n")
        f.write(f"gram_mmd_median,{gram_mmd_vkitti[idx_med]:.6e},{gram_mmd_coco[idx_med]:.6e},"
                f"{gram_mmd_vkitti[idx_med] < gram_mmd_coco[idx_med]}\n")

    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")
    logger.info(f"Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
