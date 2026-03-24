"""
paired_inversion_v2.py
======================
Test d'inversion CMMD vs Gram-MMD avec paires exactes.

  Ancrage : 1000 KITTI Tracking (séquences VKITTI : 0001,0002,0006,0018,0020)
  Query A : 1000 VKITTI clone_rgb appairées 1:1 (même frame, texture synthétique)
  Query B : 1000 KITTI Tracking (toutes séquences, réel)

Hypothèse :
  - CMMD (CLIP sémantique) : VKITTI clone ≈ ancrage → plus proche → INVERSION
  - GMMD (Gram texture)    : synthétique ≠ réel → plus loin → PAS d'inversion

Usage:
    .venv/bin/python experiments/paired_inversion_v2.py
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
OUT_DIR      = BASE_DIR / "results" / "paired_inversion_v2"

N_PAIRS = 1000
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
VKITTI_SEQS = set(SCENE_TO_SEQ.values())

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
            frame_num = int(Path(real_path).stem)
            vkitti_name = f"{scene}_clone_rgb_{frame_num:05d}.jpg"
            vkitti_path = VKITTI_DIR / vkitti_name
            if vkitti_path.exists():
                tracking_paths.append(str(real_path))
                vkitti_paths.append(str(vkitti_path))

    logger.info(f"  Found {len(tracking_paths)} paired frames across {len(SCENE_TO_SEQ)} scenes")
    return tracking_paths, vkitti_paths


def subsample_paired(tracking_paths, vkitti_paths, n, seed):
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(tracking_paths), size=min(n, len(tracking_paths)),
                            replace=False).tolist())
    return [tracking_paths[i] for i in idx], [vkitti_paths[i] for i in idx]


def load_query_b(n, seed):
    """Query B = tracking toutes séquences."""
    tracking_all = []
    for seq_dir in sorted(TRACKING_DIR.iterdir()):
        tracking_all.extend(sorted(glob_module.glob(str(seq_dir / "*.png"))))
    logger.info(f"  Tracking pool (all seqs): {len(tracking_all)} frames")

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(tracking_all), size=min(n, len(tracking_all)),
                            replace=False).tolist())
    paths = [tracking_all[i] for i in idx]
    logger.info(f"  Query B: {len(paths)} tracking images")
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
        dists = {qn: mmd2_unbiased(gram_ref, qf, gamma) for qn, qf in query_feats.items()}
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

    # ── Build paired lists ────────────────────────────────
    all_tracking, all_vkitti = build_paired_paths()
    anchor_paths, vkitti_paths = subsample_paired(all_tracking, all_vkitti, N_PAIRS, SEED)
    queryb_paths = load_query_b(N_PAIRS, SEED)

    queries = {
        "VKITTI_clone": vkitti_paths,
        "Tracking":   queryb_paths,
    }

    logger.info("=" * 90)
    logger.info("PAIRED INVERSION TEST v2")
    logger.info(f"  Anchor:         {len(anchor_paths)} KITTI Tracking (paired seqs)")
    logger.info(f"  Query VKITTI:   {len(vkitti_paths)} clone_rgb (appairées 1:1)")
    logger.info(f"  Query Tracking: {len(queryb_paths)} (toutes séquences)")
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
    print("PAIRED INVERSION TEST v2")
    print(f"Anchor = {len(anchor_paths)} KITTI Tracking (paired)")
    print(f"  VKITTI_clone: {len(vkitti_paths)} (appairées 1:1, synthétique)")
    print(f"  Tracking:     {len(queryb_paths)} (toutes séquences)")
    print("=" * 110)

    header = f"  {'Méthode':<20s}" + "".join(f"  {qn:>14s}" for qn in qnames) + "  {'Ratio':>8s}  INV?"
    print(header)
    print("  " + "-" * 90)

    for r in all_rows:
        d = r["distances"]
        ratio = d["VKITTI_clone"] / d["Tracking"] if d["Tracking"] > 0 else float("inf")
        inv = d["VKITTI_clone"] < d["Tracking"]
        line = f"  {r['method']:<20s}"
        for qn in qnames:
            line += f"  {d[qn]:>14.4e}"
        line += f"  {ratio:>8.3f}  {'OUI' if inv else 'non'}"
        print(line)

    # Check pattern
    print()
    n_cmmd_inv = sum(1 for r in all_rows if "CMMD" in r["method"]
                     and r["distances"]["VKITTI_clone"] < r["distances"]["Tracking"])
    n_gram_inv = sum(1 for r in all_rows if "Gram" in r["method"]
                     and r["distances"]["VKITTI_clone"] < r["distances"]["Tracking"])
    if n_cmmd_inv > 0 and n_gram_inv == 0:
        print("  ★ CMMD inverse mais PAS Gram-MMD ★")
    elif n_gram_inv > 0 and n_cmmd_inv == 0:
        print("  ★ Gram-MMD inverse mais PAS CMMD ★")
    else:
        print(f"  CMMD inversions: {n_cmmd_inv}, Gram inversions: {n_gram_inv}")
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
        w.writerow(["method"] + qnames + ["ratio", "gamma", "inversion"])
        for r in all_rows:
            d = r["distances"]
            ratio = d["VKITTI_clone"] / d["Tracking"] if d["Tracking"] > 0 else float("inf")
            inv = d["VKITTI_clone"] < d["Tracking"]
            row = [r["method"]] + [f"{d[qn]:.6e}" for qn in qnames]
            row += [f"{ratio:.4f}",
                    f"{r['gamma']:.4e}" if r["gamma"] is not None else "",
                    inv]
            w.writerow(row)
    logger.info(f"  CSV: {csv_path}")

    # Plot
    methods = [r["method"] for r in all_rows]
    ratios = [r["distances"]["VKITTI_clone"] / r["distances"]["Tracking"]
              if r["distances"]["Tracking"] > 0 else 0 for r in all_rows]
    colors = ["#F44336" if rat < 1 else "#4CAF50" for rat in ratios]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(methods, ratios, color=colors, edgecolor="white", zorder=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    for bar, rat in zip(bars, ratios):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{rat:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Ratio dist(VKITTI_clone) / dist(Tracking)")
    ax.set_title(
        "Paired: Anchor=KITTI Tracking | VKITTI clone (paired) vs Tracking+COCO\n"
        "Rouge <1 = INVERSION (VKITTI vu comme plus proche)",
        fontsize=11, fontweight="bold",
    )
    ax.set_facecolor("#f9f9f9")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(plots_dir / "paired_v2_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'paired_v2_ratios.png'}")
    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
