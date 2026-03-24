"""
paired_inversion_test.py
=========================
Test d'inversion CMMD vs Gram-MMD avec des paires exactes.

Virtual KITTI "clone" = rendu synthétique frame-à-frame des séquences
KITTI Tracking. Même pose caméra, même composition, même sémantique,
mais texture synthétique.

  Ancrage (réf) : 1000 frames KITTI Tracking (réelles)
  Query A       : 1000 frames Virtual KITTI clone_rgb appairées 1:1
                  (même sémantique, texture synthétique)
  Query B       : 1000 images COCO clean
                  (sémantique différente, texture réelle)

Hypothèse :
  - CMMD (CLIP, sémantique) → VKITTI plus proche que COCO  (inversion!)
  - GMMD (Gram texture)     → VKITTI plus loin que COCO    (pas d'inversion)

Métriques :
  - Gram-MMD : SD-VAE L11, 4 gammas (0.1×, 1×, 10×, 100× médiane)
  - CMMD     : CLIP ViT-L/14@336, gamma fixe 0.005 + sweep
  - Cosine distance CLIP
  - FID CLIP

Usage:
    .venv/bin/python experiments/paired_inversion_test.py
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
OUT_DIR      = BASE_DIR / "results" / "paired_inversion"

N_PAIRS = 1000
N_COCO  = 1000
SEED    = 42
BATCH_SIZE = 16

# Scene mapping: VKITTI Scene → KITTI Tracking sequence
SCENE_TO_SEQ = {
    "Scene01": "0001",
    "Scene02": "0002",
    "Scene06": "0006",
    "Scene18": "0018",
    "Scene20": "0020",
}

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
GAMMA_GRID_CLIP = np.logspace(-4, 2, 50)


# ════════════════════════════════════════════════════════════
#  BUILD PAIRED LISTS
# ════════════════════════════════════════════════════════════

def build_paired_paths():
    """Build 1:1 paired lists: tracking real frame ↔ VKITTI clone_rgb frame."""
    tracking_paths = []
    vkitti_paths = []

    for scene, seq in SCENE_TO_SEQ.items():
        seq_dir = TRACKING_DIR / seq
        if not seq_dir.exists():
            logger.warning(f"  Tracking seq {seq} not found, skipping")
            continue

        real_frames = sorted(glob_module.glob(str(seq_dir / "*.png")))

        for real_path in real_frames:
            frame_num = int(Path(real_path).stem)  # e.g. 42
            vkitti_name = f"{scene}_clone_rgb_{frame_num:05d}.jpg"
            vkitti_path = VKITTI_DIR / vkitti_name

            if vkitti_path.exists():
                tracking_paths.append(str(real_path))
                vkitti_paths.append(str(vkitti_path))

    logger.info(f"  Found {len(tracking_paths)} paired frames across {len(SCENE_TO_SEQ)} scenes")
    return tracking_paths, vkitti_paths


def subsample_paired(tracking_paths, vkitti_paths, n, seed):
    """Subsample n pairs, keeping pairing intact."""
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(tracking_paths), size=min(n, len(tracking_paths)),
                            replace=False).tolist())
    return [tracking_paths[i] for i in idx], [vkitti_paths[i] for i in idx]


def sample_coco(n, seed):
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(all_imgs), size=min(n, len(all_imgs)), replace=False).tolist())
    return [all_imgs[i] for i in idx]


# ════════════════════════════════════════════════════════════
#  UTILS
# ════════════════════════════════════════════════════════════

def squared_l2_matrix(X, Y):
    X_sq = np.einsum("ij,ij->i", X, X)
    Y_sq = np.einsum("ij,ij->i", Y, Y)
    return X_sq[:, None] + Y_sq[None, :] - 2.0 * (X @ Y.T)


def mmd2_unbiased(D_rr, D_qq, D_rq, gamma):
    n_r, n_q = D_rr.shape[0], D_qq.shape[0]
    K_rr = np.exp(-gamma * D_rr)
    K_qq = np.exp(-gamma * D_qq)
    K_rq = np.exp(-gamma * D_rq)
    term_rr = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))
    term_qq = (np.sum(K_qq) - np.trace(K_qq)) / (n_q * (n_q - 1))
    term_rq = 2.0 * np.sum(K_rq) / (n_r * n_q)
    return float(term_rr + term_qq - term_rq)


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


def cosine_distance(feat_ref, feat_query):
    ref_n = feat_ref / (np.linalg.norm(feat_ref, axis=1, keepdims=True) + 1e-8)
    q_n = feat_query / (np.linalg.norm(feat_query, axis=1, keepdims=True) + 1e-8)
    return float(1.0 - np.mean(ref_n @ q_n.T))


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
    logger.info("PAIRED INVERSION TEST")
    logger.info("  Anchor:  KITTI Tracking (real)")
    logger.info("  Query A: Virtual KITTI clone (paired synthetic)")
    logger.info("  Query B: COCO (out-of-domain)")
    logger.info(f"  N_pairs={N_PAIRS}  N_coco={N_COCO}  device={device}")
    logger.info("=" * 90)

    # ── Build paired paths ─────────────────────────────────
    all_tracking, all_vkitti = build_paired_paths()
    tracking_paths, vkitti_paths = subsample_paired(all_tracking, all_vkitti, N_PAIRS, SEED)
    coco_paths = sample_coco(N_COCO, SEED)

    logger.info(f"  Anchor (tracking): {len(tracking_paths)}")
    logger.info(f"  Query A (VKITTI):  {len(vkitti_paths)}")
    logger.info(f"  Query B (COCO):    {len(coco_paths)}")

    results = {}

    # ════════════════════════════════════════════════════════
    #  GRAM-MMD
    # ════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info(f"GRAM-MMD : SD-VAE layer {LAYER} + StandardScaler")
    logger.info("=" * 90)

    ext = FeatureExtractor(
        backbone=BACKBONE, layer=LAYER,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )

    gram_ref    = ext.extract(tracking_paths, fit_transform=True, batch_size=BATCH_SIZE).astype(np.float32)
    gram_vkitti = ext.extract(vkitti_paths, fit_transform=False, batch_size=BATCH_SIZE).astype(np.float32)
    gram_coco   = ext.extract(coco_paths, fit_transform=False, batch_size=BATCH_SIZE).astype(np.float32)
    ext.cleanup()
    del ext
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"  Features: ref={gram_ref.shape}, vkitti={gram_vkitti.shape}, coco={gram_coco.shape}")

    # Distance matrices
    D_rr = squared_l2_matrix(gram_ref, gram_ref)
    D_vv = squared_l2_matrix(gram_vkitti, gram_vkitti)
    D_cc = squared_l2_matrix(gram_coco, gram_coco)
    D_rv = squared_l2_matrix(gram_ref, gram_vkitti)
    D_rc = squared_l2_matrix(gram_ref, gram_coco)

    # Median heuristic
    triu = D_rr[np.triu_indices(len(gram_ref), k=1)]
    gamma_med = 1.0 / (2.0 * float(np.median(triu)) + 1e-8)
    logger.info(f"  gamma_med = {gamma_med:.4e}")

    for tag, mult in GAMMA_MULTIPLIERS.items():
        g = mult * gamma_med
        m_vk = mmd2_unbiased(D_rr, D_vv, D_rv, g)
        m_co = mmd2_unbiased(D_rr, D_cc, D_rc, g)
        inv = m_vk < m_co
        key = f"Gram {tag}"
        results[key] = {"vkitti": m_vk, "coco": m_co, "gamma": g, "inversion": inv}
        logger.info(f"  {key:<15s}  γ={g:.2e}  VKITTI={m_vk:.6e}  COCO={m_co:.6e}  "
                     f"{'INVERSION!' if inv else 'ok'}")

    del D_rr, D_vv, D_cc, D_rv, D_rc, gram_ref, gram_vkitti, gram_coco
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  CLIP / CMMD
    # ════════════════════════════════════════════════════════
    logger.info(f"\n{'='*90}")
    logger.info(f"CMMD : CLIP ViT-L/14@336")
    logger.info("=" * 90)

    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    clip_ref    = clip_embed(tracking_paths, clip_model, clip_proc, device)
    clip_vkitti = clip_embed(vkitti_paths, clip_model, clip_proc, device)
    clip_coco   = clip_embed(coco_paths, clip_model, clip_proc, device)
    del clip_model, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"  CLIP features: ref={clip_ref.shape}, vkitti={clip_vkitti.shape}, coco={clip_coco.shape}")

    # Cosine distance
    cos_vk = cosine_distance(clip_ref, clip_vkitti)
    cos_co = cosine_distance(clip_ref, clip_coco)
    inv_cos = cos_vk < cos_co
    results["Cosine CLIP"] = {"vkitti": cos_vk, "coco": cos_co, "gamma": None, "inversion": inv_cos}
    logger.info(f"  Cosine CLIP    VKITTI={cos_vk:.6f}  COCO={cos_co:.6f}  "
                f"{'INVERSION!' if inv_cos else 'ok'}")

    # FID on CLIP embeddings
    fid_vk = compute_fid(clip_ref, clip_vkitti)
    fid_co = compute_fid(clip_ref, clip_coco)
    inv_fid = fid_vk < fid_co
    results["FID CLIP"] = {"vkitti": fid_vk, "coco": fid_co, "gamma": None, "inversion": inv_fid}
    logger.info(f"  FID CLIP       VKITTI={fid_vk:.4f}  COCO={fid_co:.4f}  "
                f"{'INVERSION!' if inv_fid else 'ok'}")

    # CMMD at fixed gamma
    Dc_rr = squared_l2_matrix(clip_ref, clip_ref)
    Dc_vv = squared_l2_matrix(clip_vkitti, clip_vkitti)
    Dc_cc = squared_l2_matrix(clip_coco, clip_coco)
    Dc_rv = squared_l2_matrix(clip_ref, clip_vkitti)
    Dc_rc = squared_l2_matrix(clip_ref, clip_coco)

    m_vk_cmmd = mmd2_unbiased(Dc_rr, Dc_vv, Dc_rv, CMMD_GAMMA)
    m_co_cmmd = mmd2_unbiased(Dc_rr, Dc_cc, Dc_rc, CMMD_GAMMA)
    inv_cmmd = m_vk_cmmd < m_co_cmmd
    results[f"CMMD γ={CMMD_GAMMA}"] = {"vkitti": m_vk_cmmd, "coco": m_co_cmmd,
                                         "gamma": CMMD_GAMMA, "inversion": inv_cmmd}
    logger.info(f"  CMMD γ=0.005   VKITTI={m_vk_cmmd:.6e}  COCO={m_co_cmmd:.6e}  "
                f"{'INVERSION!' if inv_cmmd else 'ok'}")

    # CMMD median heuristic
    triu_clip = Dc_rr[np.triu_indices(len(clip_ref), k=1)]
    gamma_med_clip = 1.0 / (2.0 * float(np.median(triu_clip)) + 1e-8)
    m_vk_med = mmd2_unbiased(Dc_rr, Dc_vv, Dc_rv, gamma_med_clip)
    m_co_med = mmd2_unbiased(Dc_rr, Dc_cc, Dc_rc, gamma_med_clip)
    inv_med = m_vk_med < m_co_med
    results[f"CMMD γ_med"] = {"vkitti": m_vk_med, "coco": m_co_med,
                               "gamma": gamma_med_clip, "inversion": inv_med}
    logger.info(f"  CMMD γ_med={gamma_med_clip:.2e}  VKITTI={m_vk_med:.6e}  COCO={m_co_med:.6e}  "
                f"{'INVERSION!' if inv_med else 'ok'}")

    # Gamma sweep CLIP
    clip_sweep_vk = []
    clip_sweep_co = []
    for g in GAMMA_GRID_CLIP:
        clip_sweep_vk.append(mmd2_unbiased(Dc_rr, Dc_vv, Dc_rv, g))
        clip_sweep_co.append(mmd2_unbiased(Dc_rr, Dc_cc, Dc_rc, g))

    del Dc_rr, Dc_vv, Dc_cc, Dc_rv, Dc_rc
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════
    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")

    print("\n" + "=" * 95)
    print("PAIRED INVERSION TEST — KITTI Tracking (anchor) vs VKITTI clone (paired) vs COCO")
    print("=" * 95)
    print(f"  {'Méthode':<20s}  {'VKITTI (paired)':>15s}  {'COCO':>15s}  {'INVERSION?':>12s}")
    print("  " + "-" * 65)
    for name, r in results.items():
        vk_str = f"{r['vkitti']:.6e}" if r['vkitti'] > 0.001 else f"{r['vkitti']:.6f}"
        co_str = f"{r['coco']:.6e}" if r['coco'] > 0.001 else f"{r['coco']:.6f}"
        inv_str = "OUI ←" if r['inversion'] else "non"
        print(f"  {name:<20s}  {vk_str:>15s}  {co_str:>15s}  {inv_str:>12s}")
    print("=" * 95)

    n_inv = sum(1 for r in results.values() if r["inversion"])
    n_clip_inv = sum(1 for k, r in results.items()
                     if r["inversion"] and ("CLIP" in k or "CMMD" in k or "Cosine" in k or "FID" in k))
    n_gram_inv = sum(1 for k, r in results.items()
                     if r["inversion"] and "Gram" in k)

    print(f"\n  Inversions totales: {n_inv}/{len(results)}")
    print(f"  Inversions CLIP/CMMD: {n_clip_inv}")
    print(f"  Inversions Gram-MMD:  {n_gram_inv}")
    if n_clip_inv > 0 and n_gram_inv == 0:
        print("\n  ★ CMMD inverse le classement mais PAS Gram-MMD ★")
    print()

    # ════════════════════════════════════════════════════════
    #  PLOTS
    # ════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── Plot 1: Bar chart all metrics ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f9f9f9")

    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.35

    vk_vals = [results[m]["vkitti"] for m in methods]
    co_vals = [results[m]["coco"] for m in methods]

    bars1 = ax.bar(x - width/2, vk_vals, width,
                   label="Virtual KITTI (paired, synthetic)", color="#2196F3",
                   edgecolor="white", zorder=3)
    bars2 = ax.bar(x + width/2, co_vals, width,
                   label="COCO (out-of-domain, real)", color="#4CAF50",
                   edgecolor="white", zorder=3)

    # Highlight inversions
    for i, m in enumerate(methods):
        if results[m]["inversion"]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red", zorder=1)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f"{h:.3e}" if h < 0.01 else f"{h:.4f}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Distance (MMD² / Cosine / FID)", fontsize=11)
    ax.set_title(
        "Paired inversion test — Anchor = KITTI Tracking (real)\n"
        "VKITTI clone (même composition) vs COCO (sémantique différente)\n"
        "Zone rouge = INVERSION (VKITTI plus proche que COCO)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    inv_patch = mpatches.Patch(color="red", alpha=0.15, label="Inversion zone")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(inv_patch)
    ax.legend(handles=handles, fontsize=9)

    plt.tight_layout()
    fig.savefig(plots_dir / "all_metrics_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'all_metrics_bars.png'}")

    # ── Plot 2: CLIP gamma sweep ─────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#f9f9f9")

    ax.plot(GAMMA_GRID_CLIP, clip_sweep_vk, "-o", color="#2196F3",
            label="Virtual KITTI (paired)", markersize=3, linewidth=1.5)
    ax.plot(GAMMA_GRID_CLIP, clip_sweep_co, "-s", color="#4CAF50",
            label="COCO", markersize=3, linewidth=1.5)

    ax.axvline(CMMD_GAMMA, color="purple", linestyle=":", alpha=0.7,
               label=f"γ=0.005 (CMMD)")
    ax.axvline(gamma_med_clip, color="red", linestyle="--", alpha=0.7,
               label=f"γ_med={gamma_med_clip:.2e}")

    # Shade inversion zones
    sweep_vk = np.array(clip_sweep_vk)
    sweep_co = np.array(clip_sweep_co)
    inv_mask = sweep_vk < sweep_co
    if np.any(inv_mask):
        for i in range(len(inv_mask)):
            if inv_mask[i]:
                lo = GAMMA_GRID_CLIP[max(0, i-1)]
                hi = GAMMA_GRID_CLIP[min(len(inv_mask)-1, i)]
                ax.axvspan(lo, hi, alpha=0.2, color="red")
        ax.axvspan(0, 0, alpha=0.2, color="red", label="INVERSION")

    ax.set_xscale("log")
    ax.set_xlabel("γ (gamma)", fontsize=11)
    ax.set_ylabel("MMD² (unbiased)", fontsize=11)
    ax.set_title("CMMD (CLIP) gamma sweep — paired VKITTI vs COCO\n"
                 "Zone rouge = VKITTI plus proche que COCO (inversion sémantique)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(plots_dir / "clip_gamma_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'clip_gamma_sweep.png'}")

    # ── Plot 3: Normalized comparison ─────────────────────
    # Normalize each metric by max(vkitti, coco) to compare on same scale
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#f9f9f9")

    ratios = []
    for m in methods:
        vk = results[m]["vkitti"]
        co = results[m]["coco"]
        # Ratio < 1 means VKITTI closer = inversion
        ratios.append(vk / co if co > 0 else float("inf"))

    colors = ["#F44336" if r < 1 else "#4CAF50" for r in ratios]
    bars = ax.bar(methods, ratios, color=colors, edgecolor="white", zorder=3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Seuil (ratio=1)")

    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{r:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Ratio MMD²(VKITTI) / MMD²(COCO)", fontsize=11)
    ax.set_title("Ratio VKITTI/COCO — <1 = INVERSION (rouge), >1 = correct (vert)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(plots_dir / "ratio_vkitti_over_coco.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'ratio_vkitti_over_coco.png'}")

    # ── Save CSV ──────────────────────────────────────────
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "vkitti_paired", "coco", "ratio_vk_co", "gamma", "inversion"])
        for m in methods:
            r = results[m]
            ratio = r["vkitti"] / r["coco"] if r["coco"] > 0 else float("inf")
            w.writerow([m, f"{r['vkitti']:.6e}", f"{r['coco']:.6e}",
                        f"{ratio:.4f}",
                        f"{r['gamma']:.4e}" if r["gamma"] is not None else "",
                        r["inversion"]])
    logger.info(f"  CSV saved: {csv_path}")

    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
