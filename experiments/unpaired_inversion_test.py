"""
unpaired_inversion_test.py
===========================
Test d'inversion CMMD vs Gram-MMD avec des distributions NON-appairées.

  Ancrage : 1000 frames KITTI Tracking — séquences HORS VKITTI
            (seq 0000,0003-0005,0007-0017,0019 = 5882 frames dispo)
  Query A : 1000 VKITTI clone_rgb (même domaine conduite, synthétique)
  Query B : 1000 COCO (domaine différent, réel)

L'ancrage et le query A viennent de séquences DIFFÉRENTES →
pas de composition spatiale partagée.

Hypothèse :
  - CMMD (CLIP) : VKITTI "conduite" ≈ ancrage "conduite" → proche → INVERSION
  - GMMD (Gram) : texture synthétique ≠ réel → loin → PAS d'inversion

Usage:
    .venv/bin/python experiments/unpaired_inversion_test.py
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
import matplotlib.patches as mpatches
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
OUT_DIR      = BASE_DIR / "results" / "unpaired_inversion"

# Séquences tracking associées à VKITTI (exclues de l'ancrage)
VKITTI_SEQS = {"0001", "0002", "0006", "0018", "0020"}

N_ANCHOR = 1000
N_QUERY  = 1000
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
    """Load tracking frames from sequences NOT in VKITTI."""
    paths = []
    for seq_dir in sorted(TRACKING_DIR.iterdir()):
        if seq_dir.name in VKITTI_SEQS:
            continue
        paths.extend(sorted(glob_module.glob(str(seq_dir / "*.png"))))
    logger.info(f"  Anchor pool (non-VKITTI seqs): {len(paths)} frames")
    return paths


VKITTI_GROUPS = {
    "clone":       ["clone_rgb"],
    "fog":         ["fog_rgb"],
    "rain_sunset": ["rain_rgb", "sunset_rgb"],
    "all_mix":     ["clone_rgb", "fog_rgb", "rain_rgb", "sunset_rgb", "morning_rgb"],
}


def load_vkitti_group_paths(group_name):
    """Load VKITTI images for a given variant group."""
    variants = VKITTI_GROUPS[group_name]
    paths = []
    for variant in variants:
        paths.extend(sorted(glob_module.glob(str(VKITTI_DIR / f"*_{variant}_*.jpg"))))
    logger.info(f"  VKITTI [{group_name}] pool: {len(paths)} images ({variants})")
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
#  EVALUATION CORE
# ════════════════════════════════════════════════════════════

def evaluate_gram(anchor_paths, vkitti_paths, coco_paths, device, use_scaler=True):
    """Run Gram-MMD for one VKITTI group, return dict of results."""
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
    gram_vk  = ext.extract(vkitti_paths, fit_transform=False,
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
        m_vk = mmd2_unbiased(gram_ref, gram_vk, gamma)
        m_co = mmd2_unbiased(gram_ref, gram_co, gamma)
        rows.append({
            "method": f"Gram{'_std' if use_scaler else '_raw'} {tag}",
            "vkitti": m_vk, "coco": m_co, "gamma": gamma,
            "inversion": m_vk < m_co,
        })

    del gram_ref, gram_vk, gram_co
    gc.collect()
    return rows


def evaluate_clip(anchor_paths, vkitti_paths, coco_paths, device,
                  clip_model, clip_proc):
    """Run CMMD + cosine + FID for one VKITTI group."""
    clip_ref = clip_embed(anchor_paths, clip_model, clip_proc, device)
    clip_vk  = clip_embed(vkitti_paths, clip_model, clip_proc, device)
    clip_co  = clip_embed(coco_paths, clip_model, clip_proc, device)

    rows = []

    # CMMD γ=0.005
    m_vk = mmd2_unbiased(clip_ref, clip_vk, CMMD_GAMMA)
    m_co = mmd2_unbiased(clip_ref, clip_co, CMMD_GAMMA)
    rows.append({"method": "CMMD γ=0.005", "vkitti": m_vk, "coco": m_co,
                 "gamma": CMMD_GAMMA, "inversion": m_vk < m_co})

    # CMMD γ_median
    g_med = median_gamma(clip_ref)
    m_vk = mmd2_unbiased(clip_ref, clip_vk, g_med)
    m_co = mmd2_unbiased(clip_ref, clip_co, g_med)
    rows.append({"method": "CMMD γ_med", "vkitti": m_vk, "coco": m_co,
                 "gamma": g_med, "inversion": m_vk < m_co})

    # Cosine
    cos_vk = cosine_distance(clip_ref, clip_vk)
    cos_co = cosine_distance(clip_ref, clip_co)
    rows.append({"method": "Cosine CLIP", "vkitti": cos_vk, "coco": cos_co,
                 "gamma": None, "inversion": cos_vk < cos_co})

    # FID
    fid_vk = compute_fid(clip_ref, clip_vk)
    fid_co = compute_fid(clip_ref, clip_co)
    rows.append({"method": "FID CLIP", "vkitti": fid_vk, "coco": fid_co,
                 "gamma": None, "inversion": fid_vk < fid_co})

    del clip_ref, clip_vk, clip_co
    gc.collect()
    return rows


# ════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════

def plot_group_results(all_data, plots_dir):
    """One summary figure: ratio per method, one subplot per VKITTI group."""
    groups = list(all_data.keys())
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 7), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        rows = all_data[grp]
        methods = [r["method"] for r in rows]
        ratios = [r["vkitti"] / r["coco"] if r["coco"] > 0 else 0 for r in rows]
        colors = ["#F44336" if r < 1 else "#4CAF50" for r in ratios]

        bars = ax.barh(methods, ratios, color=colors, edgecolor="white", zorder=3)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)

        for bar, r in zip(bars, ratios):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{r:.2f}", va="center", fontsize=8, fontweight="bold")

        ax.set_title(f"VKITTI: {grp}", fontsize=11, fontweight="bold")
        ax.set_facecolor("#f9f9f9")
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_xlabel("Ratio MMD²(VKITTI) / MMD²(COCO)", fontsize=10)
    fig.suptitle(
        "Anchor = KITTI Tracking (séq. hors VKITTI) vs COCO\n"
        "Rouge <1 = INVERSION (VKITTI vu comme plus proche que COCO)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(plots_dir / "all_groups_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'all_groups_ratios.png'}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_total = time.time()

    logger.info("=" * 90)
    logger.info("UNPAIRED INVERSION TEST — 4 VKITTI groups")
    logger.info(f"  Anchor:  {N_ANCHOR} KITTI Tracking (séquences hors VKITTI)")
    logger.info(f"  Query A: {N_QUERY} VKITTI (clone / fog / rain+sunset / all_mix)")
    logger.info(f"  Query B: {N_QUERY} COCO")
    logger.info(f"  Device: {device}")
    logger.info("=" * 90)

    # ── Load pools ────────────────────────────────────────
    anchor_pool = load_anchor_paths()
    coco_pool   = load_coco_paths()
    anchor_paths = subsample(anchor_pool, N_ANCHOR, SEED)
    coco_paths   = subsample(coco_pool, N_QUERY, SEED)

    all_data = {}   # group_name → list of result rows
    all_rows = []   # flat list for CSV

    # ── Load CLIP model once ──────────────────────────────
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    for grp_name in VKITTI_GROUPS:
        logger.info(f"\n{'='*90}")
        logger.info(f"VKITTI GROUP: {grp_name}  ({VKITTI_GROUPS[grp_name]})")
        logger.info("=" * 90)

        vkitti_pool = load_vkitti_group_paths(grp_name)
        vkitti_paths = subsample(vkitti_pool, N_QUERY, SEED)
        logger.info(f"  Selected {len(vkitti_paths)} VKITTI [{grp_name}] images")

        group_rows = []

        # Gram-MMD with scaler
        logger.info(f"\n  Gram-MMD + StandardScaler")
        rows = evaluate_gram(anchor_paths, vkitti_paths, coco_paths, device, use_scaler=True)
        for r in rows:
            r["group"] = grp_name
            logger.info(f"    {r['method']:<18s}  VK={r['vkitti']:.4e}  CO={r['coco']:.4e}  "
                         f"ratio={r['vkitti']/r['coco']:.3f}  {'INV!' if r['inversion'] else 'ok'}")
        group_rows.extend(rows)

        # Gram-MMD raw
        logger.info(f"\n  Gram-MMD raw (no scaler)")
        rows = evaluate_gram(anchor_paths, vkitti_paths, coco_paths, device, use_scaler=False)
        for r in rows:
            r["group"] = grp_name
            logger.info(f"    {r['method']:<18s}  VK={r['vkitti']:.4e}  CO={r['coco']:.4e}  "
                         f"ratio={r['vkitti']/r['coco']:.3f}  {'INV!' if r['inversion'] else 'ok'}")
        group_rows.extend(rows)

        # CLIP metrics
        logger.info(f"\n  CMMD / CLIP metrics")
        rows = evaluate_clip(anchor_paths, vkitti_paths, coco_paths, device,
                             clip_model, clip_proc)
        for r in rows:
            r["group"] = grp_name
            logger.info(f"    {r['method']:<18s}  VK={r['vkitti']:.4e}  CO={r['coco']:.4e}  "
                         f"ratio={r['vkitti']/r['coco']:.3f}  {'INV!' if r['inversion'] else 'ok'}")
        group_rows.extend(rows)

        all_data[grp_name] = group_rows
        all_rows.extend(group_rows)

    del clip_model, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════
    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")

    print("\n" + "=" * 110)
    print("UNPAIRED INVERSION TEST — 4 VKITTI groups")
    print("Anchor = KITTI Tracking (séq. hors VKITTI) | Query B = COCO")
    print("=" * 110)

    for grp_name, rows in all_data.items():
        print(f"\n  ── VKITTI group: {grp_name} ({VKITTI_GROUPS[grp_name]}) ──")
        print(f"  {'Méthode':<20s}  {'VKITTI':>14s}  {'COCO':>14s}  {'Ratio':>8s}  {'INV?':>5s}")
        print("  " + "-" * 65)
        for r in rows:
            ratio = r["vkitti"] / r["coco"] if r["coco"] > 0 else float("inf")
            print(f"  {r['method']:<20s}  {r['vkitti']:>14.4e}  {r['coco']:>14.4e}  "
                  f"{ratio:>8.3f}  {'OUI' if r['inversion'] else 'non':>5s}")

        n_clip = sum(1 for r in rows if r["inversion"] and
                     any(k in r["method"] for k in ["CMMD", "CLIP", "Cosine", "FID"]))
        n_gram = sum(1 for r in rows if r["inversion"] and "Gram" in r["method"])
        if n_clip > 0 and n_gram == 0:
            print(f"  ★ CMMD inverse mais PAS Gram-MMD ★")
        elif n_gram > 0 and n_clip == 0:
            print(f"  ★ Gram-MMD inverse mais PAS CMMD ★")

    print("\n" + "=" * 110)

    # ════════════════════════════════════════════════════════
    #  SAVE
    # ════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "vkitti", "coco", "ratio", "gamma", "inversion"])
        for r in all_rows:
            ratio = r["vkitti"] / r["coco"] if r["coco"] > 0 else float("inf")
            w.writerow([r["group"], r["method"],
                        f"{r['vkitti']:.6e}", f"{r['coco']:.6e}", f"{ratio:.4f}",
                        f"{r['gamma']:.4e}" if r["gamma"] is not None else "",
                        r["inversion"]])
    logger.info(f"  CSV: {csv_path}")

    # Plot
    plot_group_results(all_data, plots_dir)

    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
