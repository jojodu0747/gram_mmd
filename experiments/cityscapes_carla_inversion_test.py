"""
cityscapes_carla_inversion_test.py
===================================
Test d'inversion CMMD vs Gram-MMD.

  Ancrage : 1000 Cityscapes (leftImg8bit_trainextra) — conduite urbaine réelle
  Query A : 1000 CARLA Town01 (synthétique, conduite urbaine — même sémantique)
  Query B : 1000 COCO (domaine différent, réel)

Hypothèse :
  - CMMD (CLIP sémantique) : CARLA "conduite urbaine" ≈ Cityscapes → INVERSION
  - GMMD (Gram texture)    : texture synthétique ≠ réel → PAS d'inversion

Usage:
    .venv/bin/python experiments/cityscapes_carla_inversion_test.py
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

CITYSCAPES_DIR = BASE_DIR / "dataset" / "leftImg8bit_trainextra" / "leftImg8bit" / "train_extra"
CARLA_DIR      = BASE_DIR / "dataset" / "Town01" / "generated" / "images_rgb"
COCO_DIR       = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR        = BASE_DIR / "results" / "cityscapes_carla_inversion"

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

def load_cityscapes_paths():
    paths = sorted(glob_module.glob(str(CITYSCAPES_DIR / "**" / "*.png"), recursive=True))
    logger.info(f"  Cityscapes pool: {len(paths)} images")
    return paths

def load_carla_paths():
    paths = sorted(
        glob_module.glob(str(CARLA_DIR / "*.png")) +
        glob_module.glob(str(CARLA_DIR / "*.jpg"))
    )
    logger.info(f"  CARLA Town01 pool: {len(paths)} images")
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
    query_feats = {qn: ext.extract(qp, fit_transform=False, batch_size=BATCH_SIZE).astype(np.float32)
                   for qn, qp in queries.items()}
    ext.cleanup()
    del ext
    torch.cuda.empty_cache()
    gc.collect()

    g_med = median_gamma(gram_ref)
    rows = []
    for tag, mult in GAMMA_MULTIPLIERS.items():
        gamma = mult * g_med
        dists = {qn: mmd2_unbiased(gram_ref, qf, gamma) for qn, qf in query_feats.items()}
        rows.append({"method": f"Gram{'_std' if use_scaler else '_raw'} {tag}",
                     "distances": dists, "gamma": gamma})
    del gram_ref, query_feats
    gc.collect()
    return rows


def evaluate_clip(anchor_paths, queries, device, clip_model, clip_proc):
    clip_ref = clip_embed(anchor_paths, clip_model, clip_proc, device)
    query_feats = {qn: clip_embed(qp, clip_model, clip_proc, device)
                   for qn, qp in queries.items()}

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

    cityscapes_pool = load_cityscapes_paths()
    carla_pool      = load_carla_paths()
    coco_pool       = load_coco_paths()

    anchor_paths = subsample(cityscapes_pool, N_ANCHOR, SEED)
    queries = {
        "CARLA":  subsample(carla_pool, N_QUERY, SEED),
        "COCO":   subsample(coco_pool, N_QUERY, SEED),
    }

    logger.info("=" * 90)
    logger.info("CITYSCAPES / CARLA INVERSION TEST")
    logger.info(f"  Anchor: {len(anchor_paths)} Cityscapes")
    for qn, qp in queries.items():
        logger.info(f"  Query {qn}: {len(qp)} images")
    logger.info(f"  Device: {device}")
    logger.info("=" * 90)

    all_rows = []

    logger.info("\n  Gram-MMD + StandardScaler")
    rows = evaluate_gram(anchor_paths, queries, device, use_scaler=True)
    for r in rows:
        parts = "  ".join(f"{qn}={r['distances'][qn]:.4e}" for qn in queries)
        logger.info(f"    {r['method']:<18s}  {parts}")
    all_rows.extend(rows)

    logger.info("\n  Gram-MMD raw (no scaler)")
    rows = evaluate_gram(anchor_paths, queries, device, use_scaler=False)
    for r in rows:
        parts = "  ".join(f"{qn}={r['distances'][qn]:.4e}" for qn in queries)
        logger.info(f"    {r['method']:<18s}  {parts}")
    all_rows.extend(rows)

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
    print("\n" + "=" * 100)
    print("CITYSCAPES / CARLA INVERSION TEST")
    print(f"Anchor = {len(anchor_paths)} Cityscapes | CARLA ({len(queries['CARLA'])}) vs COCO ({len(queries['COCO'])})")
    print("=" * 100)
    header = f"  {'Méthode':<20s}" + "".join(f"  {qn:>14s}" for qn in qnames) + "  {'Ratio':>8s}  INV?"
    print(header)
    print("  " + "-" * 70)

    for r in all_rows:
        d = r["distances"]
        ratio = d["CARLA"] / d["COCO"] if d["COCO"] > 0 else float("inf")
        inv = d["CARLA"] < d["COCO"]
        line = f"  {r['method']:<20s}"
        for qn in qnames:
            line += f"  {d[qn]:>14.4e}"
        line += f"  {ratio:>8.3f}  {'OUI' if inv else 'non'}"
        print(line)

    print()
    n_cmmd_inv = sum(1 for r in all_rows if "CMMD" in r["method"] and r["distances"]["CARLA"] < r["distances"]["COCO"])
    n_gram_inv = sum(1 for r in all_rows if "Gram" in r["method"] and r["distances"]["CARLA"] < r["distances"]["COCO"])
    if n_cmmd_inv > 0 and n_gram_inv == 0:
        print("  ★ CMMD inverse mais PAS Gram-MMD ★")
    elif n_gram_inv > 0 and n_cmmd_inv == 0:
        print("  ★ Gram-MMD inverse mais PAS CMMD ★")
    else:
        print(f"  CMMD inversions: {n_cmmd_inv}, Gram inversions: {n_gram_inv}")
    print("=" * 100)

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
            ratio = d["CARLA"] / d["COCO"] if d["COCO"] > 0 else float("inf")
            inv = d["CARLA"] < d["COCO"]
            w.writerow([r["method"]] + [f"{d[qn]:.6e}" for qn in qnames] +
                       [f"{ratio:.4f}", f"{r['gamma']:.4e}" if r["gamma"] is not None else "", inv])
    logger.info(f"  CSV: {csv_path}")

    methods = [r["method"] for r in all_rows]
    ratios = [r["distances"]["CARLA"] / r["distances"]["COCO"] if r["distances"]["COCO"] > 0 else 0
              for r in all_rows]
    colors = ["#F44336" if rat < 1 else "#4CAF50" for rat in ratios]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(methods, ratios, color=colors, edgecolor="white", zorder=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
    for bar, rat in zip(bars, ratios):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{rat:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Ratio dist(CARLA) / dist(COCO)")
    ax.set_title(
        f"Anchor = {len(anchor_paths)} Cityscapes | CARLA Town01 vs COCO\n"
        "Rouge <1 = INVERSION (CARLA vu comme plus proche que COCO)",
        fontsize=11, fontweight="bold",
    )
    ax.set_facecolor("#f9f9f9")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(plots_dir / "cityscapes_carla_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {plots_dir / 'cityscapes_carla_ratios.png'}")
    logger.info(f"\nResults in: {OUT_DIR}")


if __name__ == "__main__":
    main()
