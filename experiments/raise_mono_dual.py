"""
raise_mono_dual.py
==================
Test de monotonicité MOS ↔ MMD sur RAISE, deux granularités en un seul run :
  • Phase 1 : 48 groupes × 10 images  (n=48, barplot −ρ)
  • Phase 2 : 12 groupes × 40 images  (n=12, scatter)

Ancrage : 1000 images COCO (seed=42).
Comparaison avec CMMD (CLIP ViT-L/14@336).

Features extraites UNE SEULE FOIS par backbone, puis reagroupées.

Sorties dans results/raise_mono_dual/ :
  raise_mono_g48x10_detailed_{ts}.csv
  raise_mono_g48x10_summary_{ts}.csv
  raise_mono_g12x40_detailed_{ts}.csv
  raise_mono_g12x40_summary_{ts}.csv

Usage :
    .venv/bin/python experiments/raise_mono_dual.py
    .venv/bin/python experiments/raise_mono_dual.py --skip-cmmd
"""

import argparse
import csv
import gc
import glob as glob_module
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, pearsonr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

RAISE_ROOT          = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_TRAIN_IMAGES  = RAISE_ROOT / "images" / "train_images"
RAISE_TEST_IMAGES   = RAISE_ROOT / "images" / "test_images"
RAISE_TRAIN_RATINGS = RAISE_ROOT / "ratings" / "train.csv"
RAISE_TEST_RATINGS  = RAISE_ROOT / "ratings" / "test.csv"
COCO_DIR            = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR             = BASE_DIR / "results" / "raise_mono_dual"

N_ANCHOR    = 1000
ANCHOR_SEED = 42

# Deux granularités : (N_GROUPS, GROUP_SIZE)
GROUP_CONFIGS = [
    (48, 10),   # 48 × 10 = 480  → barplot −ρ Spearman
    (12, 40),   # 12 × 40 = 480  → scatter
]

# Backbones sélectionnés
SELECTED_CONFIGS = [
    {"backbone": "sd_vae",        "layer": 11, "gamma": 1e-4,   "batch_size": 16},
    {"backbone": "sd_vae",        "layer": 11, "gamma": 2.8e-5, "batch_size": 16},
    {"backbone": "dinov2_vitb14", "layer": 11, "gamma": 1e-6,   "batch_size": 8},
    {"backbone": "dc_ae",         "layer": 9,  "gamma": 5e-6,   "batch_size": 8},
]

ENABLE_CMMD   = True
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005   # sigma=10 → gamma = 1/(2σ²)
BATCH_CLIP    = 32

# ═══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════

def load_coco_anchor() -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if not all_imgs:
        raise FileNotFoundError(f"Aucune image dans {COCO_DIR}")
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = sorted(rng.choice(len(all_imgs), size=N_ANCHOR, replace=False).tolist())
    paths = [all_imgs[i] for i in idx]
    logger.info(f"Ancrage COCO : {len(paths)} images (seed={ANCHOR_SEED})")
    return paths


def load_mos_dict() -> dict:
    mos_dict = {}
    for csv_path in [RAISE_TRAIN_RATINGS, RAISE_TEST_RATINGS]:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = float(row["MOS_Rating"])
    return mos_dict


def load_raise_fake(mos_dict: dict) -> list:
    """480 images IA (f*.png), triées par MOS_Rating croissant."""
    all_imgs = {}
    for folder in [RAISE_TRAIN_IMAGES, RAISE_TEST_IMAGES]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[os.path.basename(p)] = p
    items = [
        (path, mos_dict[fname])
        for fname, path in all_imgs.items()
        if fname.startswith("f") and fname in mos_dict
    ]
    items.sort(key=lambda x: x[1])
    logger.info(f"Images IA RAISE : {len(items)}  MOS [{items[0][1]:.2f} … {items[-1][1]:.2f}]")
    return items


def make_groups(items: list, n_groups: int, group_size: int) -> list:
    assert len(items) >= n_groups * group_size, \
        f"Besoin de {n_groups * group_size} images, trouvé {len(items)}"
    groups = []
    for i in range(n_groups):
        g = items[i * group_size: (i + 1) * group_size]
        groups.append({
            "paths":   [x[0] for x in g],
            "mos":     float(np.mean([x[1] for x in g])),
            "mos_min": g[0][1],
            "mos_max": g[-1][1],
        })
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  MMD² (non-biaisé)
# ═══════════════════════════════════════════════════════════════════════

def sq_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True)
    return np.maximum(sq_A + sq_B.T - 2.0 * (A @ B.T), 0.0)


def mmd2_unbiased(anchor_feat: np.ndarray, group_feat: np.ndarray,
                  gamma: float, term1_rr: float) -> float:
    n_r = anchor_feat.shape[0]
    n_g = group_feat.shape[0]
    D_rg = sq_dist_matrix(anchor_feat, group_feat)
    D_gg = sq_dist_matrix(group_feat, group_feat)
    K_rg = np.exp(-gamma * D_rg)
    K_gg = np.exp(-gamma * D_gg)
    term2 = (np.sum(K_gg) - np.trace(K_gg)) / (n_g * (n_g - 1)) if n_g > 1 else 1.0
    term3 = 2.0 * np.sum(K_rg) / (n_r * n_g)
    return float(term1_rr + term2 - term3)


def compute_term1_rr(anchor_feat: np.ndarray, gamma: float) -> float:
    D_rr = sq_dist_matrix(anchor_feat, anchor_feat)
    K_rr = np.exp(-gamma * D_rr)
    n = anchor_feat.shape[0]
    return float((np.sum(K_rr) - np.trace(K_rr)) / (n * (n - 1)))


# ═══════════════════════════════════════════════════════════════════════
#  STATISTIQUES
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(scores: list, groups: list, label: str) -> dict:
    s  = np.array(scores)
    mos = np.array([g["mos"] for g in groups])
    rho, p_rho = spearmanr(s, mos)
    r,   p_r   = pearsonr(s, mos)
    logger.info(
        f"  {label:55s}  ρ={rho:+.4f}  r={r:+.4f}  p_ρ={p_rho:.3e}  p_r={p_r:.3e}"
    )
    return {
        "label":       label,
        "n_groups":    len(scores),
        "spearman_rho": float(rho),
        "p_spearman":   float(p_rho),
        "pearson_r":    float(r),
        "p_pearson":    float(p_r),
    }


# ═══════════════════════════════════════════════════════════════════════
#  SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════

DETAIL_FIELDS  = ["label", "group_idx", "mos_mean", "mos_min", "mos_max", "mmd_score"]
SUMMARY_FIELDS = ["label", "n_groups", "spearman_rho", "p_spearman", "pearson_r", "p_pearson"]


def save_csv(rows: list, path: Path, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  Sauvegardé : {path}")


def make_detail_rows(scores: list, groups: list, label: str) -> list:
    return [
        {
            "label":     label,
            "group_idx": i + 1,
            "mos_mean":  g["mos"],
            "mos_min":   g["mos_min"],
            "mos_max":   g["mos_max"],
            "mmd_score": score,
        }
        for i, (score, g) in enumerate(zip(scores, groups))
    ]


# ═══════════════════════════════════════════════════════════════════════
#  ÉVALUATION D'UN BACKBONE GRAM-MMD
# ═══════════════════════════════════════════════════════════════════════

def compute_median_gamma(anchor_feat: np.ndarray) -> float:
    """gamma = 1 / (2 · median(d²))  sur les paires de l'ancrage."""
    D_rr  = sq_dist_matrix(anchor_feat, anchor_feat)
    triu  = D_rr[np.triu_indices(D_rr.shape[0], k=1)]
    return float(1.0 / (2.0 * float(np.median(triu)) + 1e-8))


def evaluate_gram_backbone(cfg: dict, anchor_paths: list, items: list,
                            all_group_configs: list,
                            all_detail: dict, all_summary: dict):
    backbone    = cfg["backbone"]
    layer       = cfg["layer"]
    gamma_cfg   = cfg["gamma"]   # None → médiane adaptive
    bs          = cfg["batch_size"]

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKBONE : {backbone} / layer {layer}")
    logger.info(f"{'='*60}")

    extractor = FeatureExtractor(
        backbone=backbone,
        layer=layer,
        transform_config={
            "name":           "gram_spatial",
            "use_gram":       True,
            "use_pca":        False,
            "gram_patches":   False,
            "gram_averaging": "spatial",
        },
    )

    # Extraction ancrage
    logger.info(f"  Extraction ancrage ({N_ANCHOR} images)...")
    anchor_feat = extractor.extract(anchor_paths, fit_transform=True, batch_size=bs)
    logger.info(f"  anchor_feat : {anchor_feat.shape}")

    # Gamma
    if gamma_cfg is None:
        gamma = compute_median_gamma(anchor_feat)
        gamma_tag = "median"
        logger.info(f"  gamma médiane = {gamma:.4e}")
    else:
        gamma = gamma_cfg
        gamma_tag = f"{gamma:.1e}"

    label = f"MMD_{backbone}_L{layer}_g{gamma_tag}"

    # term1_rr pour ce gamma
    term1_rr = compute_term1_rr(anchor_feat, gamma)
    logger.info(f"  gamma={gamma:.4e}  term1_rr={term1_rr:.6f}")

    # Extraction features RAISE (toutes les 480 images en une passe)
    all_paths = [item[0] for item in items]
    logger.info(f"  Extraction 480 images RAISE...")
    all_feat = extractor.extract(
        [Image.open(p).convert("RGB") for p in all_paths],
        fit_transform=False, batch_size=bs
    )
    logger.info(f"  all_feat : {all_feat.shape}")

    # Pour chaque granularité
    for n_groups, group_size in all_group_configs:
        key = f"g{n_groups}x{group_size}"
        groups = make_groups(items, n_groups, group_size)

        logger.info(f"\n  -- Granularité {n_groups}×{group_size} --")
        scores = []
        for i, grp in enumerate(groups):
            start = i * group_size
            end   = start + group_size
            gf    = all_feat[start:end]
            score = mmd2_unbiased(anchor_feat, gf, gamma, term1_rr)
            scores.append(score)
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(f"    Groupe {i+1}/{n_groups} MMD²={score:.5f}  MOS={grp['mos']:.2f}")

        stats = compute_stats(scores, groups, label)
        if key not in all_detail:
            all_detail[key] = []
            all_summary[key] = []
        all_detail[key]  += make_detail_rows(scores, groups, label)
        all_summary[key].append(stats)

    extractor.cleanup()
    del extractor, anchor_feat, all_feat
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  ÉVALUATION CMMD
# ═══════════════════════════════════════════════════════════════════════

def evaluate_cmmd(anchor_paths: list, items: list,
                  all_group_configs: list, device: str,
                  all_detail: dict, all_summary: dict):
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    label = "CMMD_CLIP-ViT-L14"
    logger.info(f"\n{'='*60}")
    logger.info(f"CMMD — {CLIP_MODEL_ID}")
    logger.info(f"{'='*60}")

    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def embed_paths(paths: list) -> np.ndarray:
        embs = []
        for i in range(0, len(paths), BATCH_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
        return np.concatenate(embs, axis=0)

    # Ancrage
    logger.info(f"  Embeddings ancrage ({N_ANCHOR} images)...")
    anc_emb = embed_paths(anchor_paths)
    D_rr_c  = sq_dist_matrix(anc_emb, anc_emb)
    K_rr_c  = np.exp(-CMMD_GAMMA * D_rr_c)
    n_r     = anc_emb.shape[0]
    term1   = float((np.sum(K_rr_c) - np.trace(K_rr_c)) / (n_r * (n_r - 1)))
    logger.info(f"  CMMD gamma={CMMD_GAMMA:.4e}  term1_rr={term1:.6f}")

    # Embeddings RAISE
    all_paths = [item[0] for item in items]
    logger.info(f"  Embeddings 480 images RAISE...")
    all_emb = embed_paths(all_paths)
    logger.info(f"  all_emb : {all_emb.shape}")

    for n_groups, group_size in all_group_configs:
        key    = f"g{n_groups}x{group_size}"
        groups = make_groups(items, n_groups, group_size)
        logger.info(f"\n  -- Granularité {n_groups}×{group_size} --")
        scores = []
        for i, grp in enumerate(groups):
            start = i * group_size
            end   = start + group_size
            gf    = all_emb[start:end]
            score = mmd2_unbiased(anc_emb, gf, CMMD_GAMMA, term1)
            scores.append(score)
        stats = compute_stats(scores, groups, label)
        if key not in all_detail:
            all_detail[key] = []
            all_summary[key] = []
        all_detail[key]  += make_detail_rows(scores, groups, label)
        all_summary[key].append(stats)

    del model, proc, anc_emb, all_emb
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cmmd", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("RAISE Monotonicity — Dual granularity")
    logger.info(f"  Granularités : {GROUP_CONFIGS}")
    logger.info(f"  Ancrage      : {N_ANCHOR} COCO (seed={ANCHOR_SEED})")
    logger.info(f"  Backbones    : {[c['backbone'] for c in SELECTED_CONFIGS]}")
    logger.info(f"  CMMD         : {ENABLE_CMMD and not args.skip_cmmd}")
    logger.info(f"  Device       : {args.device}")
    logger.info("=" * 60)

    anchor_paths = load_coco_anchor()
    mos_dict     = load_mos_dict()
    items        = load_raise_fake(mos_dict)

    all_detail  = {}  # key: "g96x5" / "g12x40"  → list of detail rows
    all_summary = {}  # key: same               → list of summary dicts

    # Backbones Gram-MMD
    for cfg in SELECTED_CONFIGS:
        try:
            evaluate_gram_backbone(
                cfg, anchor_paths, items, GROUP_CONFIGS, all_detail, all_summary
            )
        except Exception as exc:
            logger.error(f"ERREUR {cfg['backbone']} : {exc}", exc_info=True)

    # CMMD
    if ENABLE_CMMD and not args.skip_cmmd:
        try:
            evaluate_cmmd(anchor_paths, items, GROUP_CONFIGS, args.device,
                          all_detail, all_summary)
        except Exception as exc:
            logger.error(f"ERREUR CMMD : {exc}", exc_info=True)

    # Sauvegarde
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in all_detail:
        n_g, g_s = key[1:].split("x")
        tag = f"g{n_g}x{g_s}"
        save_csv(all_detail[key],  OUT_DIR / f"raise_mono_{tag}_detailed_{ts}.csv",  DETAIL_FIELDS)
        save_csv(all_summary[key], OUT_DIR / f"raise_mono_{tag}_summary_{ts}.csv",  SUMMARY_FIELDS)

    # Résumé terminal
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ FINAL")
    logger.info("=" * 80)
    for key, rows in all_summary.items():
        logger.info(f"\n  [{key}]")
        for r in sorted(rows, key=lambda x: -x["spearman_rho"]):
            logger.info(
                f"    {r['label']:55s}  ρ={r['spearman_rho']:+.4f}  r={r['pearson_r']:+.4f}"
            )

    logger.info(f"\nRésultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
