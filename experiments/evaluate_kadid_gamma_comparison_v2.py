"""
evaluate_kadid_gamma_comparison_v2.py
======================================
Version optimisée : single-pass extraction par backbone.

Pour chaque backbone/layer :
  1. Extraction ancrage (81 images) → calcul des 3 gammas
  2. UNE SEULE passe sur les 10125 images dégradées
     → kernel calculé pour les 3 gammas simultanément dans chaque batch
  3. Scoring final = mean_Krr + 1 - 2 * cross_term  (par gamma)

Gain : 3× moins de forward passes GPU vs v1.

Backbones :
  - SD-VAE L11    (batch 16, input 256)
  - DC-AE L8      (batch  4, input 512)
  - DINOv2 L3     (batch  8, input 518)
  + CMMD (CLIP ViT-L/14@336, γ fixe = 0.005)

Usage :
    .venv/bin/python experiments/evaluate_kadid_gamma_comparison_v2.py
    .venv/bin/python experiments/evaluate_kadid_gamma_comparison_v2.py --skip-cmmd
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
    ("sd_vae",        11, "SD-VAE L11",  16),
    ("dc_ae",          8, "DC-AE L8",     4),
    ("dinov2_vitb14",  3, "DINOv2 L3",    8),
]

GAMMA_MULTIPLIERS = {
    "median":     1.0,
    "10x_median": 10.0,
    "50x_median": 50.0,
}

ENABLE_CMMD     = True
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32

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
#  EXTRACTION ANCRAGE
# ════════════════════════════════════════════════════════════

def extract_anchor(backbone, layer, batch_size, anchor_paths, device):
    ext = FeatureExtractor(
        backbone=backbone, layer=layer,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )
    t0 = time.time()
    ref = ext.extract(anchor_paths, fit_transform=True, batch_size=batch_size).astype(np.float32)
    logger.info(f"  Ancrage : {ref.shape}  ({time.time()-t0:.1f}s)")
    return ext, ref


def compute_gammas(ref: np.ndarray) -> dict:
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    D = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
    triu = D[np.triu_indices(len(ref), k=1)]
    median_d2 = float(np.median(triu))
    g_med = 1.0 / (2.0 * median_d2 + 1e-8)
    logger.info(f"  median(d²) = {median_d2:.4e}  → γ_median = {g_med:.4e}")
    gammas = {tag: mult * g_med for tag, mult in GAMMA_MULTIPLIERS.items()}
    for tag, g in gammas.items():
        logger.info(f"    {tag:15s}: γ = {g:.4e}")
    return gammas


# ════════════════════════════════════════════════════════════
#  SINGLE-PASS SCORING (tous gammas simultanément)
# ════════════════════════════════════════════════════════════

def score_all_gammas_single_pass(ext, ref, gammas: dict, dist_paths, device, batch_size):
    """
    Une seule passe sur les 10125 images.
    Pour chaque batch : 1 forward pass GPU, N kernels (un par gamma).

    Retourne : {gamma_tag: np.ndarray shape (N,)}
    """
    n = len(dist_paths)
    ref_sq = np.einsum("ij,ij->i", ref, ref)  # (n_ref,)

    # Pré-calcul mean_Krr et accumulateurs cross-term
    mean_Krr = {}
    cross_sum = {}   # accumule Σ_i K(x_i, y_j) pour chaque j
    for tag, g in gammas.items():
        K_rr = np.exp(-g * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)))
        mean_Krr[tag] = float(np.mean(K_rr))
        cross_sum[tag] = np.zeros(n, dtype=np.float64)
        logger.info(f"  [{tag}] mean_Krr = {mean_Krr[tag]:.6f}")

    dataset = ImageDataset(dist_paths, transform=ext.image_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"),
                        persistent_workers=True)

    t0 = time.time()
    idx = 0
    for batch_imgs, _ in loader:
        # ─ 1 seul forward pass GPU ─
        feats = ext.extract_batch(batch_imgs)
        feats = ext.transform_features(feats, fit_pca=False).astype(np.float32)
        B = feats.shape[0]

        t_sq = np.einsum("ij,ij->i", feats, feats)   # (B,)
        # D_rt[i, j] = ||ref_i - feat_j||²   shape (n_ref, B)
        D_rt = ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ feats.T)

        # ─ N kernel evaluations (très rapide, pas de GPU) ─
        for tag, g in gammas.items():
            K_rt = np.exp(-g * D_rt)                  # (n_ref, B)
            cross_sum[tag][idx:idx + B] = np.sum(K_rt, axis=0)  # Σ_i K(x_i, y_j)

        idx += B
        if idx % 2000 < batch_size or idx >= n:
            logger.info(f"    {idx}/{n} ({idx/n:.0%})  {time.time()-t0:.0f}s")

        del feats, D_rt
        gc.collect()

    logger.info(f"  Passe unique terminée : {time.time()-t0:.1f}s")

    # Calcul final des scores
    n_ref = len(ref)
    scores = {}
    for tag, g in gammas.items():
        # k(y,y) = exp(0) = 1
        scores[tag] = mean_Krr[tag] + 1.0 - 2.0 / n_ref * cross_sum[tag]

    return scores


# ════════════════════════════════════════════════════════════
#  CMMD (CLIP ViT-L/14@336) — single pass aussi
# ════════════════════════════════════════════════════════════

def run_cmmd(anchor_paths, dist_paths, device):
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    logger.info(f"  Chargement {CLIP_MODEL_ID}...")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    # Ancrage
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

    # Scoring
    n = len(dist_paths)
    cross_sum = np.zeros(n, dtype=np.float64)
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
        cross_sum[s:e] = np.sum(K_rt, axis=0)
        if s % 2000 < chunk or e == n:
            logger.info(f"    CLIP : {e}/{n} ({e/n:.0%})  {time.time()-t0:.0f}s")
        del emb, K_rt

    scores = mean_Krr + 1.0 - 2.0 / len(ref) * cross_sum
    logger.info(f"  CMMD terminé : {time.time()-t0:.1f}s")

    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    return scores


# ════════════════════════════════════════════════════════════
#  STATISTIQUES
# ════════════════════════════════════════════════════════════

def compute_stats(rows, scores):
    degradation = np.array([5.0 - r["dmos"] for r in rows])
    sp, sp_p = spearmanr(scores, degradation)
    kt, kt_p = kendalltau(scores, degradation)
    sp_lv, _ = spearmanr(scores, [r["level"] for r in rows])

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

    by_level = defaultdict(list)
    for r, s in zip(rows, scores):
        by_level[r["level"]].append(s)
    by_level = {lv: float(np.mean(v)) for lv, v in by_level.items()}

    return {
        "spearman":     float(sp),
        "spearman_p":   float(sp_p),
        "kendall":      float(kt),
        "kendall_p":    float(kt_p),
        "spearman_lv":  float(sp_lv),
        "mono_strict":  n_mono / n_total if n_total else 0.0,
        "mono_partial": n_partial / n_total if n_total else 0.0,
        "n_total":      n_total,
        "by_level":     by_level,
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
        for e in all_results:
            st = e["stats"]
            lv = st["by_level"]
            w.writerow({
                "key": e["key"], "backbone": e["backbone"],
                "layer": e["layer"], "gamma_tag": e["gamma_tag"],
                "gamma_val": f"{e['gamma_val']:.4e}", "label": e["label"],
                "spearman":    f"{st['spearman']:.6f}",
                "spearman_p":  f"{st['spearman_p']:.4e}",
                "kendall":     f"{st['kendall']:.6f}",
                "kendall_p":   f"{st['kendall_p']:.4e}",
                "spearman_lv": f"{st['spearman_lv']:.6f}",
                "mono_strict":  f"{st['mono_strict']:.6f}",
                "mono_partial": f"{st['mono_partial']:.6f}",
                "n_total": st["n_total"],
                **{f"score_lv{lv_k}": f"{lv.get(lv_k, ''):.6e}" if lv.get(lv_k) is not None else ""
                   for lv_k in [1, 2, 3, 4, 5]},
            })
    logger.info(f"CSV résumé : {path}")

    logger.info("\n" + "=" * 85)
    logger.info("RÉSUMÉ — KADID-10k gamma comparison v2")
    logger.info(f"  {'Label':<50}  {'Spearman':>9}  {'Kendall':>8}  {'Mono.str':>9}")
    logger.info("-" * 85)
    for e in sorted(all_results, key=lambda x: -x["stats"]["spearman"]):
        st = e["stats"]
        logger.info(
            f"  {e['label']:<50}  {st['spearman']:>+9.4f}  "
            f"{st['kendall']:>+8.4f}  {st['mono_strict']:>9.1%}"
        )
    logger.info("=" * 85)
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

    t_total = time.time()
    logger.info("=" * 85)
    logger.info("KADID-10k — Gamma comparison v2 (single-pass par backbone)")
    logger.info(f"  Device : {args.device}")
    logger.info("=" * 85)

    rows = load_kadid_csv()
    ref_names    = sorted(set(r["ref_img"] for r in rows))
    anchor_paths = [str(IMAGES_DIR / n) for n in ref_names]
    dist_paths   = [str(IMAGES_DIR / r["dist_img"]) for r in rows]
    logger.info(f"  {len(ref_names)} références  |  {len(dist_paths)} dégradées")

    all_results = []

    # ── Backbones Gram MMD ──────────────────────────────────
    for backbone, layer, display, batch_size in BACKBONE_CONFIGS:
        logger.info(f"\n{'='*85}")
        logger.info(f"{display}  —  extraction ancrage + single-pass scoring (3 gammas)")
        logger.info("=" * 85)

        try:
            ext, ref = extract_anchor(backbone, layer, batch_size, anchor_paths, args.device)
            gammas   = compute_gammas(ref)

            # ─ Une seule passe sur les 10125 images ─
            t_bb = time.time()
            all_scores = score_all_gammas_single_pass(
                ext, ref, gammas, dist_paths, args.device, batch_size
            )
            logger.info(f"  Total {display} : {time.time()-t_bb:.0f}s (3 gammas)")

            ext.cleanup()
            del ext, ref
            torch.cuda.empty_cache()
            gc.collect()

            for gamma_tag, scores in all_scores.items():
                gamma_val = gammas[gamma_tag]
                stats = compute_stats(rows, scores)
                key   = f"{backbone}_l{layer}_{gamma_tag}"
                label = f"{display}  {gamma_tag}  γ={gamma_val:.2e}"
                all_results.append({
                    "key": key, "backbone": backbone, "layer": layer,
                    "gamma_tag": gamma_tag, "gamma_val": gamma_val,
                    "label": label, "stats": stats,
                })
                logger.info(
                    f"  [{gamma_tag:15s}] Spearman={stats['spearman']:+.4f}  "
                    f"Kendall={stats['kendall']:+.4f}  Mono.strict={stats['mono_strict']:.1%}"
                )

        except Exception as exc:
            logger.error(f"  ERREUR {display} : {exc}", exc_info=True)
            torch.cuda.empty_cache()
            gc.collect()

    # ── CMMD ───────────────────────────────────────────────
    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info(f"\n{'='*85}")
        logger.info(f"CMMD — CLIP ViT-L/14@336  γ={CMMD_GAMMA}")
        logger.info("=" * 85)
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
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)

    logger.info(f"\nTemps total : {(time.time()-t_total)/60:.1f} min")

    if all_results:
        save_summary(all_results, ts)
        logger.info("\nPour les graphiques :")
        logger.info("  .venv/bin/python experiments/analyze_kadid_gamma_comparison.py")


if __name__ == "__main__":
    main()
