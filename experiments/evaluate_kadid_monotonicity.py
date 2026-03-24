"""
evaluate_kadid_monotonicity.py
==============================
Évaluation de monotonie sur KADID-10k — multi-backbone.

Protocole :
  Ancrage  = 81 images de référence KADID (originales, non-dégradées)
  Éval     = 10 125 images dégradées, scoring N=1 en streaming
             → jamais plus de SCORE_CHUNK vecteurs en mémoire

Pour chaque image dégradée on calcule MMD²(y, X) biaisé :
  MMD²(y, X) = mean(K_rr) + k(y,y) - 2/n · Σ k(x_i, y)

Métriques testées (configurables) :
  - SD-VAE layer 7, Gram spatial
  - DC-AE layer 10, Gram spatial
  - DinoV2 ViT-B/14 layer 5, Gram spatial
  - CMMD (CLIP ViT-L/14@336, γ=0.005)

Tests :
  - Spearman(score, 5 - dmos) et Kendall tau
  - Taux de monotonie stricte et partielle par (ref, dist_type)

Usage :
    python experiments/evaluate_kadid_monotonicity.py
    python experiments/evaluate_kadid_monotonicity.py --skip-cmmd
    python experiments/evaluate_kadid_monotonicity.py --device cpu

Résultats -> results/kadid_monotonicity/
"""

import argparse
import csv
import gc
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import DataLoader

from gram_mmd.features import FeatureExtractor, ImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION EXPÉRIENCE                       ║
# ║  Modifier cette section pour adapter l'expérience facilement      ║
# ╚════════════════════════════════════════════════════════════════════╝

# --- Chemins ---
BASE_DIR   = Path(__file__).resolve().parent.parent
KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
OUT_DIR    = BASE_DIR / "results" / "kadid_monotonicity"

# --- Backbones MMD Gram (backbone, layer_index, display_name, gamma) ---
# Chaque entrée génère un score MMD² N=1 indépendant
# gamma = None → heuristique médiane
MMD_CONFIGS = [
    ("dc_ae",         14,  "DC-AE l14",         2.8e-5),
    # ("dc_ae",          7,  "DC-AE l7",          2.8e-5),
    # ("sd_vae",         7,  "SD-VAE l7",         2.8e-5),
    # ("dinov2_vitb14",  5,  "DinoV2 l5",         2.8e-5),
]

# --- CMMD ---
ENABLE_CMMD   = True
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005    # sigma=10 → gamma = 1/(2·σ²)

# --- Batch sizes ---
BATCH_SIZE_DEFAULT = 16    # Extraction features
BATCH_SIZE_LARGE   = 4     # DC-AE / Flux VAE (input 512x512 + gram large)
BATCH_SIZE_CLIP    = 32    # CMMD
SCORE_CHUNK        = 64    # Images traitées à la fois pour scoring (contrôle RAM)

# ╔════════════════════════════════════════════════════════════════════╗
# ║                  FIN DE LA CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════╝


def get_batch_size(backbone: str) -> int:
    if backbone in ("dc_ae", "flux_vae"):
        return BATCH_SIZE_LARGE
    return BATCH_SIZE_DEFAULT


# ============================================================
# Chargement CSV KADID
# ============================================================

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


# ============================================================
# Ancrage + scoring Gram MMD N=1
# ============================================================

def build_gram_anchor(backbone: str, layer_idx: int, gamma: float,
                      anchor_paths: list, device: str):
    """Extrait features gram des 81 refs et pré-calcule K_rr."""
    bs = get_batch_size(backbone)
    logger.info(f"  Extraction ancrage ({len(anchor_paths)} refs)  batch_size={bs}")

    ext = FeatureExtractor(
        backbone=backbone,
        layer=layer_idx,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )
    t0 = time.time()
    feats = ext.extract(anchor_paths, fit_transform=True, batch_size=bs)
    logger.info(f"  Ancrage : {feats.shape} ({time.time()-t0:.1f}s)")

    ref = feats.astype(np.float32)
    ref_sq = np.einsum("ij,ij->i", ref, ref)

    # gamma heuristique si None
    if gamma is None:
        D_rr_triu = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
        triu = D_rr_triu[np.triu_indices(len(ref), k=1)]
        gamma = 1.0 / (2.0 * float(np.median(triu)) + 1e-8)
        logger.info(f"  Gamma heuristique médiane : {gamma:.6e}")

    K_rr = np.exp(-gamma * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)))
    mean_Krr = float(np.mean(K_rr))
    logger.info(f"  gamma={gamma:.4e}  mean_Krr={mean_Krr:.6f}")

    return ext, ref, ref_sq, gamma, mean_Krr


def score_gram_streaming(ext, ref, ref_sq, gamma, mean_Krr,
                         dist_paths, device):
    """Score MMD² N=1 en streaming par chunks."""
    n = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    t0 = time.time()

    bs = get_batch_size(ext.backbone_name)
    chunk = min(bs, SCORE_CHUNK)

    dataset = ImageDataset(dist_paths, transform=ext.image_transform)
    loader = DataLoader(dataset, batch_size=chunk, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"),
                        persistent_workers=True)

    idx = 0
    for batch_imgs, _ in loader:
        feats = ext.extract_batch(batch_imgs)
        feats = ext.transform_features(feats, fit_pca=False)
        feats = feats.astype(np.float32)

        t_sq = np.einsum("ij,ij->i", feats, feats)
        K_rt = np.exp(-gamma * (ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ feats.T)))
        s = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        scores[idx:idx + len(s)] = s
        idx += len(s)

        if idx % 1000 < chunk or idx == n:
            logger.info(f"    Scoring : {idx}/{n} ({idx/n:.0%})")

        del feats, K_rt
        gc.collect()

    logger.info(f"  Scoring terminé : {time.time()-t0:.1f}s")
    return scores


# ============================================================
# CMMD (CLIP ViT-L/14@336)
# ============================================================

def build_clip_anchor(anchor_paths: list, device: str):
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
    ref_emb = np.concatenate(embeds, axis=0).astype(np.float32)
    logger.info(f"  Embeddings ancrage CLIP : {ref_emb.shape}")

    ref_sq = np.einsum("ij,ij->i", ref_emb, ref_emb)
    K_rr = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref_emb @ ref_emb.T)))
    mean_Krr = float(np.mean(K_rr))
    logger.info(f"  gamma={CMMD_GAMMA:.4e} (fixe)  mean_Krr={mean_Krr:.4f}")

    return model, proc, ref_emb, ref_sq, mean_Krr


def score_clip_streaming(model, proc, ref_emb, ref_sq, mean_Krr,
                         dist_paths, device):
    n = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    t0 = time.time()
    chunk = BATCH_SIZE_CLIP * 4

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        batch = [Image.open(p).convert("RGB") for p in dist_paths[s:e]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            emb = model(pixel_values=pv).image_embeds.float().cpu().numpy().astype(np.float32)

        t_sq = np.einsum("ij,ij->i", emb, emb)
        K_rt = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref_emb @ emb.T)))
        scores[s:e] = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)

        if s % 1000 < chunk or e == n:
            logger.info(f"    CLIP scoring : {e}/{n} ({e/n:.0%})")
        del emb, K_rt

    logger.info(f"  CLIP scoring terminé : {time.time()-t0:.1f}s")
    return scores


# ============================================================
# Analyses statistiques
# ============================================================

def spearman_kendall(rows, scores):
    degradation = np.array([5.0 - r["dmos"] for r in rows])
    levels = np.array([r["level"] for r in rows])
    sp, sp_p = spearmanr(scores, degradation)
    kt, kt_p = kendalltau(scores, degradation)
    sp_lv, _ = spearmanr(scores, levels)
    return {
        "spearman_vs_degradation": float(sp),
        "spearman_p":              float(sp_p),
        "kendall_vs_degradation":  float(kt),
        "kendall_p":               float(kt_p),
        "spearman_vs_level":       float(sp_lv),
    }


def monotonicity_rate(rows, scores):
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

    return {
        "n_total":        n_total,
        "n_mono_strict":  n_mono,
        "rate_strict":    n_mono / n_total if n_total else 0.0,
        "n_mono_partial": n_partial,
        "rate_partial":   n_partial / n_total if n_total else 0.0,
    }


def mean_score_by_level(rows, scores):
    by_lv = defaultdict(list)
    for i, row in enumerate(rows):
        by_lv[row["level"]].append(scores[i])
    return {lv: float(np.mean(v)) for lv, v in sorted(by_lv.items())}


# ============================================================
# Sauvegarde CSV
# ============================================================

def save_results(rows, results, ts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV détaillé (10125 lignes)
    metric_keys = list(results.keys())
    detail_path = OUT_DIR / f"scores_detailed_{ts}.csv"
    fieldnames = ["dist_img", "ref_img", "ref_id", "dist_type",
                  "level", "dmos", "var"] + [f"score_{k}" for k in metric_keys]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            rec = {k: row[k] for k in ("dist_img", "ref_img", "ref_id",
                                        "dist_type", "level", "dmos", "var")}
            for k in metric_keys:
                rec[f"score_{k}"] = float(results[k]["scores"][i])
            writer.writerow(rec)
    logger.info(f"CSV détaillé : {detail_path}")

    # CSV résumé
    summary_path = OUT_DIR / f"summary_{ts}.csv"
    fields = ["metric", "label",
              "spearman_vs_degradation", "spearman_p",
              "kendall_vs_degradation", "kendall_p",
              "spearman_vs_level",
              "mono_strict_n", "mono_strict_total", "mono_strict_rate",
              "mono_partial_n", "mono_partial_total", "mono_partial_rate",
              "score_lv1", "score_lv2", "score_lv3", "score_lv4", "score_lv5"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, res in results.items():
            c = res["corr"]
            m = res["mono"]
            lv = res["by_level"]
            writer.writerow({
                "metric": key, "label": res["label"],
                "spearman_vs_degradation": c["spearman_vs_degradation"],
                "spearman_p":              c["spearman_p"],
                "kendall_vs_degradation":  c["kendall_vs_degradation"],
                "kendall_p":               c["kendall_p"],
                "spearman_vs_level":       c["spearman_vs_level"],
                "mono_strict_n":     m["n_mono_strict"],
                "mono_strict_total": m["n_total"],
                "mono_strict_rate":  m["rate_strict"],
                "mono_partial_n":    m["n_mono_partial"],
                "mono_partial_total": m["n_total"],
                "mono_partial_rate": m["rate_partial"],
                "score_lv1": lv.get(1, ""), "score_lv2": lv.get(2, ""),
                "score_lv3": lv.get(3, ""), "score_lv4": lv.get(4, ""),
                "score_lv5": lv.get(5, ""),
            })
    logger.info(f"CSV résumé   : {summary_path}")

    # Résumé console
    SEP = "=" * 100
    logger.info("\n" + SEP)
    logger.info("RÉSUMÉ COMPARATIF — KADID-10k")
    logger.info(SEP)
    logger.info(f"  {'Métrique':<45}  {'Spearman':>9}  {'Kendall':>8}  "
                f"{'Mono. strict':>13}  {'Mono. >=3':>10}")
    logger.info("-" * 100)
    for key, res in results.items():
        c = res["corr"]
        m = res["mono"]
        logger.info(
            f"  {res['label']:<45}  "
            f"{c['spearman_vs_degradation']:>+9.4f}  "
            f"{c['kendall_vs_degradation']:>+8.4f}  "
            f"{m['n_mono_strict']:>5}/{m['n_total']:<4} ({m['rate_strict']:>5.1%})  "
            f"{m['n_mono_partial']:>5}/{m['n_total']:<4} ({m['rate_partial']:>5.1%})"
        )
    logger.info(SEP)

    logger.info("\nScore moyen par niveau (1=léger → 5=sévère) :")
    header = f"  {'Niveau':<8}"
    for k in metric_keys:
        header += f"  {k:>20}"
    logger.info(header)
    for lv in range(1, 6):
        row_str = f"  {lv:<8}"
        for res in results.values():
            row_str += f"  {res['by_level'].get(lv, float('nan')):>20.6f}"
        logger.info(row_str)

    logger.info(f"\nRésultats dans : {OUT_DIR}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Monotonie KADID-10k — multi-backbone")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cmmd", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_cmmd = ENABLE_CMMD and not args.skip_cmmd

    logger.info("=" * 80)
    logger.info("ÉVALUATION MONOTONIE — KADID-10k (streaming, multi-backbone)")
    logger.info(f"  Backbones : {[(b, l, d) for b, l, d, _ in MMD_CONFIGS]}")
    if run_cmmd:
        logger.info(f"  CMMD : {CLIP_MODEL_ID} (gamma={CMMD_GAMMA})")
    logger.info(f"  Device : {args.device}")
    logger.info("=" * 80)

    # Chargement
    rows = load_kadid_csv()
    ref_names = sorted(set(r["ref_img"] for r in rows))
    anchor_paths = [str(IMAGES_DIR / n) for n in ref_names]
    dist_paths = [str(IMAGES_DIR / r["dist_img"]) for r in rows]
    logger.info(f"  {len(ref_names)} références  |  {len(dist_paths)} dégradées")

    results = {}

    # ---- Backbones Gram MMD ----
    for backbone, layer_idx, display, gamma in MMD_CONFIGS:
        key = f"MMD_{backbone}_l{layer_idx}"
        logger.info("\n" + "=" * 80)
        logger.info(f"{key} — {display}  |  gamma={gamma}")
        logger.info("=" * 80)

        try:
            ext, ref, ref_sq, actual_gamma, mean_Krr = build_gram_anchor(
                backbone, layer_idx, gamma, anchor_paths, args.device
            )
            scores = score_gram_streaming(
                ext, ref, ref_sq, actual_gamma, mean_Krr, dist_paths, args.device
            )

            ext.cleanup()
            del ext, ref, ref_sq
            torch.cuda.empty_cache()
            gc.collect()

            corr = spearman_kendall(rows, scores)
            mono = monotonicity_rate(rows, scores)
            lv = mean_score_by_level(rows, scores)

            results[key] = {
                "scores": scores, "corr": corr, "mono": mono, "by_level": lv,
                "label": f"{display} gamma={actual_gamma:.2e}",
            }
            logger.info(f"  Spearman(score, 5-dmos) = {corr['spearman_vs_degradation']:+.4f}")
            logger.info(f"  Kendall tau             = {corr['kendall_vs_degradation']:+.4f}")
            logger.info(f"  Mono. stricte           = {mono['n_mono_strict']}/{mono['n_total']} "
                        f"({mono['rate_strict']:.1%})")
            logger.info(f"  Mono. partielle >=3     = {mono['n_mono_partial']}/{mono['n_total']} "
                        f"({mono['rate_partial']:.1%})")

        except Exception as exc:
            logger.error(f"  ERREUR {key} : {exc}", exc_info=True)
            torch.cuda.empty_cache()
            gc.collect()

    # ---- CMMD ----
    if run_cmmd:
        logger.info("\n" + "=" * 80)
        logger.info(f"CMMD — CLIP ViT-L/14@336  |  gamma={CMMD_GAMMA}")
        logger.info("=" * 80)

        try:
            clip_model, clip_proc, clip_ref, clip_ref_sq, clip_mean_Krr = \
                build_clip_anchor(anchor_paths, args.device)
            scores_clip = score_clip_streaming(
                clip_model, clip_proc, clip_ref, clip_ref_sq,
                clip_mean_Krr, dist_paths, args.device
            )

            del clip_model, clip_proc, clip_ref, clip_ref_sq
            torch.cuda.empty_cache()
            gc.collect()

            corr = spearman_kendall(rows, scores_clip)
            mono = monotonicity_rate(rows, scores_clip)
            lv = mean_score_by_level(rows, scores_clip)

            results["CMMD"] = {
                "scores": scores_clip, "corr": corr, "mono": mono, "by_level": lv,
                "label": f"CMMD CLIP ViT-L/14 gamma={CMMD_GAMMA:.3e}",
            }
            logger.info(f"  Spearman(score, 5-dmos) = {corr['spearman_vs_degradation']:+.4f}")
            logger.info(f"  Kendall tau             = {corr['kendall_vs_degradation']:+.4f}")
            logger.info(f"  Mono. stricte           = {mono['n_mono_strict']}/{mono['n_total']} "
                        f"({mono['rate_strict']:.1%})")

        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)
    else:
        logger.info("\nCMMD ignoré")

    # Sauvegarde
    if results:
        save_results(rows, results, ts)
    else:
        logger.error("Aucun résultat à sauvegarder !")


if __name__ == "__main__":
    main()
