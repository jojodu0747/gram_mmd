"""
kitti_fog_inversion_test.py
===========================
Test d'inversion CMMD vs Gram-MMD.

  Ancrage : 1000 images KITTI (dataset principal)
  Query A : 1000 VKITTI fog_rgb (synthétique conduite + brouillard)
  Query B : 1000 COCO (domaine différent, réel)
  Query C : 1000 KITTI Tracking (même domaine, réel)

Usage:
    .venv/bin/python experiments/kitti_fog_inversion_test.py
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

KITTI_IMG_DIR  = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR     = BASE_DIR / "dataset" / "virtual_kitti"
COCO_DIR       = BASE_DIR / "dataset" / "coco_5000"
TRACKING_DIR   = BASE_DIR / "dataset" / "data_tracking_image_2" / "training" / "image_02"
OUT_DIR        = BASE_DIR / "results" / "kitti_fog_inversion"

N_ANCHOR = 1000
N_QUERY  = 1000
SEED     = 42
BATCH_SIZE = 16

BACKBONE = "sd_vae"
LAYER    = 11
GAMMA_MULTIPLIERS = {
    "γ/10":   0.1,
    "γ_med":  1.0,
    "10×γ":  10.0,
    "100×γ": 100.0,
}

CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32


# ════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ════════════════════════════════════════════════════════════

def load_kitti_paths():
    paths = sorted(glob_module.glob(str(KITTI_IMG_DIR / "*.png")))
    logger.info(f"  KITTI pool: {len(paths)} images")
    return paths

def load_vkitti_fog_paths():
    paths = sorted(glob_module.glob(str(VKITTI_DIR / "*_fog_rgb_*.jpg")))
    logger.info(f"  VKITTI fog pool: {len(paths)} images")
    return paths

def load_coco_paths():
    paths = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    logger.info(f"  COCO pool: {len(paths)} images")
    return paths

def load_tracking_paths():
    paths = []
    for seq_dir in sorted(TRACKING_DIR.iterdir()):
        paths.extend(sorted(glob_module.glob(str(seq_dir / "*.png"))))
    logger.info(f"  Tracking pool: {len(paths)} images")
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

QUERIES = {}  # filled in main

def evaluate_gram(anchor_paths, queries, device, use_scaler=True):
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

    query_feats = {}
    for qname, qpaths in queries.items():
        query_feats[qname] = ext.extract(qpaths, fit_transform=False,
                                         batch_size=BATCH_SIZE).astype(np.float32)

    ext.cleanup()
    del ext
    torch.cuda.empty_cache()
    gc.collect()

    g_med = median_gamma(gram_ref)
    rows = []
    for tag, mult in GAMMA_MULTIPLIERS.items():
        gamma = mult * g_med
        dists = {}
        for qname, qf in query_feats.items():
            dists[qname] = mmd2_unbiased(gram_ref, qf, gamma)
        rows.append({
            "method": f"Gram{'_std' if use_scaler else '_raw'} {tag}",
            "distances": dists, "gamma": gamma,
        })

    del gram_ref, query_feats
    gc.collect()
    return rows


def evaluate_clip(anchor_paths, queries, device, clip_model, clip_proc):
    clip_ref = clip_embed(anchor_paths, clip_model, clip_proc, device)

    query_feats = {}
    for qname, qpaths in queries.items():
        query_feats[qname] = clip_embed(qpaths, clip_model, clip_proc, device)

    rows = []

    # CMMD γ=0.005
    dists = {qn: mmd2_unbiased(clip_ref, qf, CMMD_GAMMA) for qn, qf in query_feats.items()}
    rows.append({"method": "CMMD γ=0.005", "distances": dists, "gamma": CMMD_GAMMA})

    # CMMD γ_med
    g_med = median_gamma(clip_ref)
    dists = {qn: mmd2_unbiased(clip_ref, qf, g_med) for qn, qf in query_feats.items()}
    rows.append({"method": "CMMD γ_med", "distances": dists, "gamma": g_med})

    # Cosine
    dists = {qn: cosine_distance(clip_ref, qf) for qn, qf in query_feats.items()}
    rows.append({"method": "Cosine CLIP", "distances": dists, "gamma": None})

    # FID
    dists = {qn: compute_fid(clip_ref, qf) for qn, qf in query_feats.items()}
    rows.append({"method": "FID CLIP", "distances": dists, "gamma": None})

    del clip_ref, query_feats
    gc.collect()
    return rows


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_total = time.time()

    # ── Load pools ────────────────────────────────────────
    kitti_pool    = load_kitti_paths()
    fog_pool      = load_vkitti_fog_paths()
    coco_pool     = load_coco_paths()
    tracking_pool = load_tracking_paths()

    anchor_paths   = subsample(kitti_pool, N_ANCHOR, SEED)
    queries = {
        "VKITTI_fog": subsample(fog_pool, N_QUERY, SEED),
        "COCO":       subsample(coco_pool, N_QUERY, SEED),
        "Tracking":   subsample(tracking_pool, N_QUERY, SEED),
    }

    logger.info("=" * 90)
    logger.info("KITTI FOG INVERSION TEST")
    logger.info(f"  Anchor:    {len(anchor_paths)} KITTI")
    for qn, qp in queries.items():
        logger.info(f"  Query {qn}: {len(qp)} images")
    logger.info(f"  Device: {device}")
    logger.info("=" * 90)

    all_rows = []

    # Gram-MMD with scaler
    logger.info("\n  Gram-MMD + StandardScaler")
    rows = evaluate_gram(anchor_paths, queries, device, use_scaler=True)
    for r in rows:
        parts = "  ".join(f"{qn}={r['distances'][qn]:.4e}" for qn in queries)
        logger.info(f"    {r['method']:<18s}  {parts}")
    all_rows.extend(rows)

    # Gram-MMD raw
    logger.info("\n  Gram-MMD raw (no scaler)")
    rows = evaluate_gram(anchor_paths, queries, device, use_scaler=False)
    for r in rows:
        parts = "  ".join(f"{qn}={r['distances'][qn]:.4e}" for qn in queries)
        logger.info(f"    {r['method']:<18s}  {parts}")
    all_rows.extend(rows)

    # CLIP metrics
    logger.info("\n  CMMD / CLIP metrics")
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    rows = evaluate_clip(anchor_paths, queries, device, clip_model, clip_proc)
    for r in rows:
        parts = "  ".join(f"{qn}={r['distances'][qn]:.4e}" for qn in queries)
        logger.info(f"    {r['method']:<18s}  {parts}")
    all_rows.extend(rows)

    del clip_model, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    # ════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════
    logger.info(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min")

    qnames = list(queries.keys())
    print("\n" + "=" * 110)
    print("KITTI FOG INVERSION TEST")
    print(f"Anchor = {len(anchor_paths)} KITTI")
    for qn, qp in queries.items():
        print(f"  {qn}: {len(qp)} images")
    print("=" * 110)

    header = f"  {'Méthode':<20s}" + "".join(f"  {qn:>14s}" for qn in qnames) + "  Ranking (plus proche → plus loin)"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in all_rows:
        d = r["distances"]
        line = f"  {r['method']:<20s}"
        for qn in qnames:
            line += f"  {d[qn]:>14.4e}"
        # Ranking
        ranking = sorted(qnames, key=lambda qn: d[qn])
        line += f"  {' < '.join(ranking)}"
        print(line)

    # Check: fog closer than COCO for CMMD but not for Gram?
    print()
    for r in all_rows:
        d = r["distances"]
        fog_closer_than_coco = d["VKITTI_fog"] < d["COCO"]
        label = "FOG<COCO" if fog_closer_than_coco else "FOG>COCO"
        print(f"  {r['method']:<20s}  {label}")

    print("=" * 110)

    # ════════════════════════════════════════════════════════
    #  SAVE
    # ════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + qnames + ["gamma"])
        for r in all_rows:
            row = [r["method"]] + [f"{r['distances'][qn]:.6e}" for qn in qnames]
            row.append(f"{r['gamma']:.4e}" if r["gamma"] is not None else "")
            w.writerow(row)
    logger.info(f"  CSV: {csv_path}")

    # Plot: grouped bar chart
    methods = [r["method"] for r in all_rows]
    x = np.arange(len(methods))
    width = 0.25
    n_q = len(qnames)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2196F3", "#F44336", "#4CAF50"]
    for i, qn in enumerate(qnames):
        vals = [r["distances"][qn] for r in all_rows]
        ax.barh(x + i * width, vals, width, label=qn, color=colors[i % len(colors)],
                edgecolor="white", zorder=3)

    ax.set_yticks(x + width)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("MMD² / Distance")
    ax.legend(fontsize=10)
    ax.set_title(
        f"Anchor = {len(anchor_paths)} KITTI | Queries: VKITTI fog, COCO, Tracking",
        fontsize=12, fontweight="bold",
    )
    ax.set_facecolor("#f9f9f9")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(plots_dir / "kitti_fog_distances.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'kitti_fog_distances.png'}")
    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
