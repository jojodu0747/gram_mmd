"""
evaluate_kadid_gamma_comparison.py
===================================
Comparaison 9 configs (3 backbones × 3 gammas heuristiques) + CMMD sur KADID-10k.

Backbones :
  - SD-VAE L11
  - DC-AE L8
  - DINOv2 ViT-B/14 L3

Gammas (calculés sur les 81 ancres KADID) :
  - γ_median   = 1 / (2 · median(d²))
  - 10×γ_median
  - 50×γ_median

Scoring N=1 biaisé (streaming) :
  MMD²(y, X) = mean(K_rr) + k(y,y) - 2/n · Σ k(x_i, y)

Usage :
    .venv/bin/python experiments/evaluate_kadid_gamma_comparison.py
    .venv/bin/python experiments/evaluate_kadid_gamma_comparison.py --skip-cmmd
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor, ImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════

KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
OUT_DIR    = BASE_DIR / "results" / "kadid_gamma_comparison"

# (backbone, layer, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",        11, "SD-VAE L11",        16),
    ("dc_ae",          8, "DC-AE L8",            4),
    ("dinov2_vitb14",  3, "DINOv2 L3",           8),
]

GAMMA_MULTIPLIERS = {
    "median":     1.0,
    "10x_median": 10.0,
    "50x_median": 50.0,
}

ENABLE_CMMD   = True
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005
BATCH_SIZE_CLIP = 32
SCORE_CHUNK   = 64

# ════════════════════════════════════════════════════════════
#  CHARGEMENT KADID
# ════════════════════════════════════════════════════════════

def load_kadid_csv():
    rows = []
    with open(DMOS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dist_img = r["dist_img"]
            parts = dist_img.replace(".png", "").split("_")
            rows.append({
                "dist_img":  dist_img,
                "ref_img":   r["ref_img"],
                "dmos":      float(r["dmos"]),
                "var":       float(r["var"]),
                "ref_id":    int(parts[0][1:]),
                "dist_type": int(parts[1]),
                "level":     int(parts[2]),
            })
    logger.info(f"KADID CSV : {len(rows)} images dégradées")
    return rows


# ════════════════════════════════════════════════════════════
#  EXTRACTION + SCORING GRAM MMD
# ════════════════════════════════════════════════════════════

def extract_anchor_features(backbone, layer, batch_size, anchor_paths, device):
    ext = FeatureExtractor(
        backbone=backbone, layer=layer,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )
    t0 = time.time()
    feats = ext.extract(anchor_paths, fit_transform=True, batch_size=batch_size)
    logger.info(f"  Ancrage : {feats.shape}  ({time.time()-t0:.1f}s)")
    return ext, feats.astype(np.float32)


def compute_gammas(ref: np.ndarray) -> dict:
    """Calcule les 3 gammas heuristiques à partir de l'ancrage."""
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    D = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
    triu = D[np.triu_indices(len(ref), k=1)]
    median_d2 = float(np.median(triu))
    gamma_median = 1.0 / (2.0 * median_d2 + 1e-8)
    logger.info(f"  median(d²) = {median_d2:.4e}  → γ_median = {gamma_median:.4e}")
    gammas = {tag: mult * gamma_median for tag, mult in GAMMA_MULTIPLIERS.items()}
    for tag, g in gammas.items():
        logger.info(f"    {tag:15s}: γ = {g:.4e}")
    return gammas


def precompute_kernel_rr(ref: np.ndarray, gamma: float):
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    K_rr = np.exp(-gamma * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)))
    mean_Krr = float(np.mean(K_rr))
    return ref_sq, mean_Krr


def score_streaming(ext, ref, ref_sq, gamma, mean_Krr, dist_paths, device, batch_size):
    n = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    chunk = min(batch_size, SCORE_CHUNK)

    dataset = ImageDataset(dist_paths, transform=ext.image_transform)
    loader = DataLoader(dataset, batch_size=chunk, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"),
                        persistent_workers=True)
    t0 = time.time()
    idx = 0
    for batch_imgs, _ in loader:
        feats = ext.extract_batch(batch_imgs)
        feats = ext.transform_features(feats, fit_pca=False).astype(np.float32)
        t_sq = np.einsum("ij,ij->i", feats, feats)
        K_rt = np.exp(-gamma * (ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ feats.T)))
        s = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        scores[idx:idx + len(s)] = s
        idx += len(s)
        if idx % 2000 < chunk or idx == n:
            logger.info(f"    Scoring : {idx}/{n} ({idx/n:.0%})")
        del feats, K_rt
        gc.collect()

    logger.info(f"  Scoring terminé : {time.time()-t0:.1f}s")
    return scores


# ════════════════════════════════════════════════════════════
#  CMMD
# ════════════════════════════════════════════════════════════

def run_cmmd(anchor_paths, dist_paths, device):
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    logger.info(f"  Chargement {CLIP_MODEL_ID}...")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    embeds = []
    for i in range(0, len(anchor_paths), BATCH_SIZE_CLIP):
        batch = [Image.open(p).convert("RGB") for p in anchor_paths[i:i + BATCH_SIZE_CLIP]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            embeds.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
    ref = np.concatenate(embeds, axis=0).astype(np.float32)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    K_rr = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)))
    mean_Krr = float(np.mean(K_rr))
    logger.info(f"  Ancrage CLIP : {ref.shape}  mean_Krr={mean_Krr:.6f}")

    n = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    chunk = BATCH_SIZE_CLIP * 4
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        batch = [Image.open(p).convert("RGB") for p in dist_paths[s:e]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            emb = model(pixel_values=pv).image_embeds.float().cpu().numpy().astype(np.float32)
        t_sq = np.einsum("ij,ij->i", emb, emb)
        K_rt = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ emb.T)))
        scores[s:e] = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        if s % 2000 < chunk or e == n:
            logger.info(f"    CLIP scoring : {e}/{n} ({e/n:.0%})")
        del emb, K_rt
    logger.info(f"  CMMD scoring terminé : {time.time()-t0:.1f}s")

    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    return scores


# ════════════════════════════════════════════════════════════
#  STATISTIQUES
# ════════════════════════════════════════════════════════════

def compute_stats(rows, scores):
    degradation = np.array([5.0 - r["dmos"] for r in rows])
    levels = np.array([r["level"] for r in rows])
    sp, sp_p = spearmanr(scores, degradation)
    kt, kt_p = kendalltau(scores, degradation)
    sp_lv, _ = spearmanr(scores, levels)

    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        groups[(row["ref_id"], row["dist_type"])][row["level"]] = scores[i]
    n_total = n_mono = n_partial = 0
    for lvl_scores in groups.values():
        if len(lvl_scores) < 5:
            continue
        seq = [lvl_scores[lv] for lv in sorted(lvl_scores)]
        n_total += 1
        if all(seq[i] < seq[i + 1] for i in range(4)):
            n_mono += 1
        if sum(1 for i in range(4) if seq[i] < seq[i + 1]) >= 3:
            n_partial += 1

    by_level = {}
    for r, s in zip(rows, scores):
        by_level.setdefault(r["level"], []).append(s)
    by_level = {lv: float(np.mean(v)) for lv, v in by_level.items()}

    return {
        "spearman":      float(sp),
        "spearman_p":    float(sp_p),
        "kendall":       float(kt),
        "kendall_p":     float(kt_p),
        "spearman_lv":   float(sp_lv),
        "mono_strict":   n_mono / n_total if n_total else 0.0,
        "mono_partial":  n_partial / n_total if n_total else 0.0,
        "n_total":       n_total,
        "by_level":      by_level,
    }


# ════════════════════════════════════════════════════════════
#  SAUVEGARDE
# ════════════════════════════════════════════════════════════

def save_summary(all_results, ts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"summary_{ts}.csv"
    fields = ["key", "backbone", "layer", "gamma_tag", "gamma_val", "label",
              "spearman", "spearman_p", "kendall", "kendall_p", "spearman_lv",
              "mono_strict", "mono_partial", "n_total",
              "score_lv1", "score_lv2", "score_lv3", "score_lv4", "score_lv5"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for entry in all_results:
            st = entry["stats"]
            w.writerow({
                "key": entry["key"], "backbone": entry["backbone"],
                "layer": entry["layer"], "gamma_tag": entry["gamma_tag"],
                "gamma_val": f"{entry['gamma_val']:.4e}", "label": entry["label"],
                "spearman":    f"{st['spearman']:.6f}",
                "spearman_p":  f"{st['spearman_p']:.4e}",
                "kendall":     f"{st['kendall']:.6f}",
                "kendall_p":   f"{st['kendall_p']:.4e}",
                "spearman_lv": f"{st['spearman_lv']:.6f}",
                "mono_strict":  f"{st['mono_strict']:.6f}",
                "mono_partial": f"{st['mono_partial']:.6f}",
                "n_total": st["n_total"],
                "score_lv1": f"{st['by_level'].get(1, ''):.6e}" if st['by_level'].get(1) else "",
                "score_lv2": f"{st['by_level'].get(2, ''):.6e}" if st['by_level'].get(2) else "",
                "score_lv3": f"{st['by_level'].get(3, ''):.6e}" if st['by_level'].get(3) else "",
                "score_lv4": f"{st['by_level'].get(4, ''):.6e}" if st['by_level'].get(4) else "",
                "score_lv5": f"{st['by_level'].get(5, ''):.6e}" if st['by_level'].get(5) else "",
            })
    logger.info(f"CSV résumé : {path}")

    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ — KADID-10k gamma comparison")
    logger.info(f"  {'Label':<45}  {'Spearman':>9}  {'Kendall':>8}  {'Mono.str':>9}  {'Mono.par':>9}")
    logger.info("-" * 80)
    for e in sorted(all_results, key=lambda x: -x["stats"]["spearman"]):
        st = e["stats"]
        logger.info(
            f"  {e['label']:<45}  {st['spearman']:>+9.4f}  {st['kendall']:>+8.4f}  "
            f"{st['mono_strict']:>9.1%}  {st['mono_partial']:>9.1%}"
        )
    logger.info("=" * 80)
    logger.info(f"\nRésultats dans : {OUT_DIR}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cmmd", action="store_true")
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 80)
    logger.info("KADID-10k — Comparaison gamma heuristique (9 configs + CMMD)")
    logger.info(f"  Device : {args.device}")
    logger.info("=" * 80)

    rows = load_kadid_csv()
    ref_names = sorted(set(r["ref_img"] for r in rows))
    anchor_paths = [str(IMAGES_DIR / n) for n in ref_names]
    dist_paths   = [str(IMAGES_DIR / r["dist_img"]) for r in rows]
    logger.info(f"  {len(ref_names)} références  |  {len(dist_paths)} dégradées")

    all_results = []

    # ── Backbones MMD ────────────────────────────────────────
    for backbone, layer, display, batch_size in BACKBONE_CONFIGS:
        logger.info(f"\n{'='*80}")
        logger.info(f"{display} — extraction ancrage + calcul gammas")
        logger.info("=" * 80)

        try:
            ext, ref = extract_anchor_features(
                backbone, layer, batch_size, anchor_paths, args.device
            )
            gammas = compute_gammas(ref)

            for gamma_tag, gamma_val in gammas.items():
                logger.info(f"\n  ── {display}  γ={gamma_val:.4e} ({gamma_tag}) ──")
                ref_sq, mean_Krr = precompute_kernel_rr(ref, gamma_val)
                logger.info(f"     mean_Krr = {mean_Krr:.6f}")

                scores = score_streaming(
                    ext, ref, ref_sq, gamma_val, mean_Krr,
                    dist_paths, args.device, batch_size
                )
                stats = compute_stats(rows, scores)

                key   = f"{backbone}_l{layer}_{gamma_tag}"
                label = f"{display}  γ={gamma_val:.2e} ({gamma_tag})"
                all_results.append({
                    "key": key, "backbone": backbone, "layer": layer,
                    "gamma_tag": gamma_tag, "gamma_val": gamma_val,
                    "label": label, "stats": stats,
                })
                logger.info(
                    f"  Spearman={stats['spearman']:+.4f}  Kendall={stats['kendall']:+.4f}  "
                    f"Mono.strict={stats['mono_strict']:.1%}"
                )
                del scores, ref_sq
                gc.collect()

            ext.cleanup()
            del ext, ref
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as exc:
            logger.error(f"  ERREUR {display} : {exc}", exc_info=True)
            torch.cuda.empty_cache()
            gc.collect()

    # ── CMMD ────────────────────────────────────────────────
    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info(f"\n{'='*80}")
        logger.info(f"CMMD — CLIP ViT-L/14@336  γ={CMMD_GAMMA}")
        logger.info("=" * 80)
        try:
            scores_clip = run_cmmd(anchor_paths, dist_paths, args.device)
            stats = compute_stats(rows, scores_clip)
            all_results.append({
                "key": "CMMD", "backbone": "clip_vit_large", "layer": 0,
                "gamma_tag": "fixed", "gamma_val": CMMD_GAMMA,
                "label": f"CMMD CLIP-L/14  γ={CMMD_GAMMA:.3e}",
                "stats": stats,
            })
            logger.info(
                f"  Spearman={stats['spearman']:+.4f}  Kendall={stats['kendall']:+.4f}  "
                f"Mono.strict={stats['mono_strict']:.1%}"
            )
            del scores_clip
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)

    if all_results:
        save_summary(all_results, ts)


if __name__ == "__main__":
    main()
