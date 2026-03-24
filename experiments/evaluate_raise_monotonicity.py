"""
evaluate_raise_monotonicity.py
==============================
Test de monotonicité MOS <-> distance sur le dataset RAISE.

Deux modes :
  - MODE "distribution" : N_GROUPS groupes de GROUP_SIZE images, MMD² non-biaisé
  - MODE "point"        : 480 groupes de 1 image, MMD² biaisé (1 point vs distribution)

Convention signe :
  MOS élevé = image réaliste -> proche des réels -> MMD faible
  -> rho(MMD, MOS) < 0  ->  on rapporte -rho (positif si métrique cohérente)

Usage :
    python experiments/evaluate_raise_monotonicity.py
    python experiments/evaluate_raise_monotonicity.py --skip-cmmd
    python experiments/evaluate_raise_monotonicity.py --device cpu

Résultats -> results/raise_monotonicity/
"""

import argparse
import csv
import gc
import glob as glob_module
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, t as t_dist

from gram_mmd.config import BACKBONE_CONFIGS
from gram_mmd.distances import compute_mmd
from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION EXPÉRIENCE                       ║
# ║  Modifier cette section pour adapter l'expérience facilement      ║
# ╚════════════════════════════════════════════════════════════════════╝

# --- Chemins dataset ---
BASE_DIR            = Path(__file__).resolve().parent.parent
RAISE_ROOT          = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_TRAIN_IMAGES  = RAISE_ROOT / "images" / "train_images"
RAISE_TEST_IMAGES   = RAISE_ROOT / "images" / "test_images"
RAISE_TRAIN_RATINGS = RAISE_ROOT / "ratings" / "train.csv"
RAISE_TEST_RATINGS  = RAISE_ROOT / "ratings" / "test.csv"
COCO_DIR            = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR             = BASE_DIR / "results" / "raise_monotonicity"

# --- Ancrage ---
N_ANCHOR = 1000        # Nombre d'images COCO pour l'ancrage
ANCHOR_SEED = 42       # Seed de sélection aléatoire

# --- Groupes (mode distribution) ---
GROUP_SIZE = 5        # Images par groupe
N_GROUPS   = 96      # 480 / 12 = 40 groupes

# --- Mode point (N=1) ---
# Si True, ajoute une passe avec 480 groupes de 1 image (MMD biaisé)
ENABLE_POINT_MODE = False

# --- Gammas à tester (None = heuristique médiane) ---
GAMMAS = [1e-6, 1e-5, 2.8e-5, 1e-4, 1e-3, None]

# --- Backbones MMD (backbone, layer_index, display_name) ---
# Ajouter/supprimer des lignes pour tester d'autres backbones/couches
MMD_BACKBONE_CONFIGS = [
    ("dc_ae",          10, "down_blocks.3.0"),
    ("dinov2_vitb14",   5, "blocks.4"),
]

# --- CMMD ---
ENABLE_CMMD    = True    # Mettre à False pour désactiver CMMD
CLIP_MODEL_ID  = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA     = 0.005   # sigma=10 -> gamma = 1/(2*sigma²)

# --- Batch sizes ---
BATCH_SIZE_DEFAULT = 64   # Pour VGG, DinoV2, etc.
BATCH_SIZE_LARGE   = 8    # Pour DC-AE, Flux VAE (input 512x512)
BATCH_SIZE_CLIP    = 32   # Pour CMMD (CLIP)

# ╔════════════════════════════════════════════════════════════════════╗
# ║                  FIN DE LA CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════╝


def gamma_label(gamma):
    return "median" if gamma is None else f"{gamma:.2e}"


def get_batch_size(backbone: str) -> int:
    if backbone in ("dc_ae", "flux_vae"):
        return BATCH_SIZE_LARGE
    return BATCH_SIZE_DEFAULT


# ============================================================
# Chargement des données
# ============================================================

def load_coco_anchor(n: int = N_ANCHOR, seed: int = ANCHOR_SEED) -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if not all_imgs:
        raise FileNotFoundError(f"Aucune image trouvée dans {COCO_DIR}")
    if n < len(all_imgs):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(all_imgs), size=n, replace=False).tolist())
        all_imgs = [all_imgs[i] for i in idx]
    logger.info(f"Ancrage COCO-5000 : {len(all_imgs)} images (seed={seed})")
    return all_imgs


def load_mos_dict() -> dict:
    mos_dict = {}
    for csv_path in [RAISE_TRAIN_RATINGS, RAISE_TEST_RATINGS]:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = {
                    "MOS_Rating": float(row["MOS_Rating"]),
                    "MOS_ZScore": float(row["MOS_ZScore"]),
                }
    return mos_dict


def load_raise_fake(mos_dict: dict) -> list:
    """480 images IA RAISE (f*.png), triées par MOS_Rating croissant."""
    all_imgs = {}
    for folder in [RAISE_TRAIN_IMAGES, RAISE_TEST_IMAGES]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[os.path.basename(p)] = p

    fake_items = [
        (path, mos_dict[fname]["MOS_Rating"], mos_dict[fname]["MOS_ZScore"])
        for fname, path in all_imgs.items()
        if fname.startswith("f") and fname in mos_dict
    ]
    fake_items.sort(key=lambda x: x[1])
    logger.info(f"Images IA RAISE : {len(fake_items)}  "
                f"MOS [{fake_items[0][1]:.1f} ... {fake_items[-1][1]:.1f}]")
    return fake_items


def form_groups(fake_items: list, group_size: int, n_groups: int) -> list:
    assert len(fake_items) >= group_size * n_groups, \
        f"Attendu {group_size * n_groups} images, trouvé {len(fake_items)}"
    groups = []
    for i in range(n_groups):
        g = fake_items[i * group_size : (i + 1) * group_size]
        groups.append({
            "paths":    [x[0] for x in g],
            "mos_r":    float(np.mean([x[1] for x in g])),
            "mos_z":    float(np.mean([x[2] for x in g])),
            "mos_r_min": g[0][1],
            "mos_r_max": g[-1][1],
        })
    return groups


def form_point_groups(fake_items: list) -> list:
    """480 groupes de 1 image (mode point vs distribution)."""
    return [
        {
            "paths":    [item[0]],
            "mos_r":    item[1],
            "mos_z":    item[2],
            "mos_r_min": item[1],
            "mos_r_max": item[1],
        }
        for item in fake_items
    ]


# ============================================================
# Setup — gram_spatial
# ============================================================

def setup_gram_extractor(backbone: str, layer_idx: int, layer_name: str,
                         anchor_paths: list):
    batch_size = get_batch_size(backbone)
    logger.info(f"\n[MMD_RBF] {backbone}/layer{layer_idx} ({layer_name})  batch_size={batch_size}")
    extractor = FeatureExtractor(
        backbone=backbone,
        layer=layer_idx,
        transform_config={
            "name":           "gram_spatial",
            "use_gram":       True,
            "use_pca":        False,
            "gram_patches":   False,
            "gram_averaging": "spatial",
        },
    )
    logger.info(f"  Extraction ancrage ({len(anchor_paths)} images)...")
    anchor_feat = extractor.extract(anchor_paths, fit_transform=True, batch_size=batch_size)
    logger.info(f"  Features ancrage : {anchor_feat.shape}")
    return extractor, anchor_feat, batch_size


# ============================================================
# Distances pré-calculées
# ============================================================

def precompute_sq_distances(anchor_feat: np.ndarray,
                             all_group_feat: list) -> tuple:
    r    = anchor_feat.astype(np.float32)
    r_sq = np.sum(r ** 2, axis=1, keepdims=True)
    D_rr = np.maximum(r_sq + r_sq.T - 2.0 * (r @ r.T), 0.0)

    D_rgs, D_ggs = [], []
    for gf in all_group_feat:
        g    = gf.astype(np.float32)
        g_sq = np.sum(g ** 2, axis=1, keepdims=True)
        D_rgs.append(np.maximum(r_sq + g_sq.T - 2.0 * (r @ g.T), 0.0))
        D_ggs.append(np.maximum(g_sq + g_sq.T - 2.0 * (g @ g.T), 0.0))
    return D_rr, D_rgs, D_ggs


def mmd2_unbiased(term1_rr: float,
                  D_rg: np.ndarray, D_gg: np.ndarray,
                  gamma: float) -> float:
    """MMD² non-biaisé (distribution vs distribution, N_g > 1)."""
    n_r = D_rg.shape[0]
    n_g = D_gg.shape[0]
    K_rg = np.exp(-gamma * D_rg)
    K_gg = np.exp(-gamma * D_gg)
    term2 = (np.sum(K_gg) - np.trace(K_gg)) / (n_g * (n_g - 1))
    term3 = 2.0 * np.sum(K_rg) / (n_r * n_g)
    return float(term1_rr + term2 - term3)


def mmd2_point(term1_rr: float,
               D_ry: np.ndarray,
               gamma: float) -> float:
    """MMD² biaisé pour 1 point y vs distribution X (N_y = 1).

    MMD²(y, X) = term1_rr + k(y,y) - 2/n · Σ k(x_i, y)
    k(y,y) = exp(0) = 1
    """
    n_r = D_ry.shape[0]
    K_ry = np.exp(-gamma * D_ry)    # (n_r, 1)
    term2 = 1.0                     # k(y, y) = exp(0) = 1
    term3 = 2.0 * np.sum(K_ry) / n_r
    return float(term1_rr + term2 - term3)


# ============================================================
# Setup — CMMD (CLIP ViT-L/14@336)
# ============================================================

def setup_cmmd(anchor_paths: list, device: str, batch_size: int = BATCH_SIZE_CLIP):
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    logger.info(f"\n[CMMD] Chargement {CLIP_MODEL_ID}...")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    logger.info(f"  Extraction embeddings ancrage ({len(anchor_paths)} images)...")
    all_emb = []
    for i in range(0, len(anchor_paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in anchor_paths[i:i + batch_size]]
        pv    = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            all_emb.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())

    ref_embeds = np.concatenate(all_emb, axis=0)

    X_sq  = np.sum(ref_embeds ** 2, axis=1, keepdims=True)
    K_rr  = np.exp(-CMMD_GAMMA * (X_sq + X_sq.T - 2.0 * ref_embeds @ ref_embeds.T))
    n_r   = ref_embeds.shape[0]
    term1 = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))

    logger.info(f"  CMMD gamma={CMMD_GAMMA:.4e} (fixe)  term1_rr={term1:.6f}")
    return model, proc, ref_embeds, CMMD_GAMMA, term1


def extract_group_clip(model, proc, paths: list, device: str,
                       batch_size: int = BATCH_SIZE_CLIP) -> np.ndarray:
    all_emb = []
    for i in range(0, len(paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        pv    = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            all_emb.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ============================================================
# Spearman
# ============================================================

def compute_spearman_summary(scores: list, groups: list,
                              metric_label: str, backbone: str,
                              layer_name: str, gamma_str: str,
                              gamma_eff: float, group_size: int) -> dict:
    s   = np.array(scores)
    mr  = np.array([g["mos_r"] for g in groups])
    mz  = np.array([g["mos_z"] for g in groups])

    rho_r, p_r = spearmanr(s, mr)
    rho_z, p_z = spearmanr(s, mz)

    def verdict(neg_rho):
        if   neg_rho >  0.5: return "bon"
        elif neg_rho >  0.2: return "partiel"
        elif neg_rho > -0.2: return "neutre"
        else:                return "inverse"

    logger.info(
        f"  {metric_label:55s}  "
        f"MOS_R: rho={rho_r:+.4f} -rho={-rho_r:+.4f} p={p_r:.3e}  |  "
        f"MOS_Z: rho={rho_z:+.4f} -rho={-rho_z:+.4f} p={p_z:.3e}  (n={len(scores)})"
    )
    return {
        "metric_label":        metric_label,
        "backbone":            backbone,
        "layer_name":          layer_name,
        "gamma_str":           gamma_str,
        "gamma_eff":           f"{gamma_eff:.6e}",
        "n_groups":            len(scores),
        "group_size":          group_size,
        "rho_mos_rating":      float(rho_r),
        "neg_rho_mos_rating":  float(-rho_r),
        "p_mos_rating":        float(p_r),
        "verdict_mos_rating":  verdict(-rho_r),
        "rho_mos_zscore":      float(rho_z),
        "neg_rho_mos_zscore":  float(-rho_z),
        "p_mos_zscore":        float(p_z),
        "verdict_mos_zscore":  verdict(-rho_z),
    }


# ============================================================
# Sauvegarde
# ============================================================

SUMMARY_FIELDS = [
    "metric_label", "backbone", "layer_name", "gamma_str", "gamma_eff",
    "n_groups", "group_size",
    "rho_mos_rating",  "neg_rho_mos_rating",  "p_mos_rating",  "verdict_mos_rating",
    "rho_mos_zscore",  "neg_rho_mos_zscore",  "p_mos_zscore",  "verdict_mos_zscore",
]

DETAILED_FIELDS = [
    "metric_label", "backbone", "layer_name", "gamma_str",
    "group_idx", "mos_r_mean", "mos_r_min", "mos_r_max", "mmd_score",
]


def collect_detailed_rows(scores: list, groups: list,
                           metric_label: str, backbone: str,
                           layer_name: str, gamma_str: str) -> list:
    return [
        {
            "metric_label": metric_label,
            "backbone":     backbone,
            "layer_name":   layer_name,
            "gamma_str":    gamma_str,
            "group_idx":    i + 1,
            "mos_r_mean":   g["mos_r"],
            "mos_r_min":    g["mos_r_min"],
            "mos_r_max":    g["mos_r_max"],
            "mmd_score":    score,
        }
        for i, (score, g) in enumerate(zip(scores, groups))
    ]


def save_results(detailed_rows: list, summary_rows: list, ts: str,
                 n_anchor: int, n_groups: int, group_size: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"coco{n_anchor}_g{n_groups}x{group_size}"

    det_path = OUT_DIR / f"raise_monotonicity_{tag}_detailed_{ts}.csv"
    with open(det_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETAILED_FIELDS)
        w.writeheader()
        w.writerows(detailed_rows)
    logger.info(f"CSV detaille : {det_path}")

    sum_path = OUT_DIR / f"raise_monotonicity_{tag}_summary_{ts}.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(f"CSV resume  : {sum_path}")

    t_crit   = t_dist.ppf(0.975, df=n_groups - 2)
    rho_crit = t_crit / math.sqrt(t_crit ** 2 + n_groups - 2)

    sep = "-" * 145
    logger.info("\n" + sep)
    logger.info(
        f"RESUME — MMD(groupe_{group_size}, ancrage_{n_anchor}_COCO) vs MOS  |  "
        f"n={n_groups} groupes  |  seuil sig. p<0.05 : |rho| > {rho_crit:.3f}"
    )
    logger.info(
        f"  {'Metrique':55s}  {'gamma':>10s}  "
        f"{'rho(MOS_R)':>9s} {'-rho(MOS_R)':>9s} {'p':>9s}  "
        f"{'rho(MOS_Z)':>9s} {'-rho(MOS_Z)':>9s}  {'verdict':>8s}"
    )
    logger.info(sep)
    for r in sorted(summary_rows, key=lambda x: -x["neg_rho_mos_rating"]):
        sig  = "*" if r["p_mos_rating"] < 0.05 else " "
        icon = "v" if r["neg_rho_mos_rating"] > 0.5 else (
               "~" if r["neg_rho_mos_rating"] > 0.2 else "x")
        logger.info(
            f"  {r['metric_label']:55s}  {r['gamma_str']:>10s}  "
            f"{r['rho_mos_rating']:+9.4f} {r['neg_rho_mos_rating']:+9.4f} "
            f"{r['p_mos_rating']:>8.3e}{sig} "
            f"{r['rho_mos_zscore']:+9.4f} {r['neg_rho_mos_zscore']:+9.4f}  "
            f"{r['verdict_mos_rating']:>8s}  {icon}"
        )
    logger.info(sep)
    logger.info(f"\nResultats dans : {OUT_DIR}")


# ============================================================
# Evaluation MMD pour un backbone (mode distribution + point)
# ============================================================

def evaluate_backbone_mmd(backbone: str, layer_idx: int, layer_display: str,
                          anchor_paths: list,
                          groups_dist: list, groups_point: list | None,
                          detailed_rows: list, summary_rows: list,
                          detailed_pt: list, summary_pt: list):
    """Évalue un backbone en mode distribution et optionnellement en mode point."""
    logger.info("\n" + "=" * 70)
    logger.info(f"PHASE MMD_RBF — {backbone} / layer {layer_idx} ({layer_display})")
    logger.info("=" * 70)

    extractor, anchor_feat, bs = setup_gram_extractor(
        backbone, layer_idx, layer_display, anchor_paths
    )

    # --- Extraction features pour tous les groupes distribution ---
    logger.info(f"  Extraction features ({len(groups_dist)} groupes x {GROUP_SIZE} images)...")
    all_group_feat = []
    for g_idx, grp in enumerate(groups_dist):
        pils = [Image.open(p).convert("RGB") for p in grp["paths"]]
        gf   = extractor.extract(pils, fit_transform=False, batch_size=bs)
        all_group_feat.append(gf)
        if (g_idx + 1) % 10 == 0 or g_idx == 0:
            logger.info(f"    Groupe {g_idx+1}/{len(groups_dist)} : {gf.shape}")

    # --- Pré-calcul distances ---
    logger.info(f"  Pre-calcul D_rr, D_rg, D_gg...")
    D_rr, D_rgs, D_ggs = precompute_sq_distances(anchor_feat, all_group_feat)
    logger.info(f"  D_rr={D_rr.shape}  D_rg={D_rgs[0].shape}  D_gg={D_ggs[0].shape}")

    triu          = D_rr[np.triu_indices(D_rr.shape[0], k=1)]
    gamma_med_val = 1.0 / (2.0 * float(np.median(triu)) + 1e-8)
    logger.info(f"  Gamma mediane : {gamma_med_val:.6e}")
    n_r = anchor_feat.shape[0]

    # --- MODE DISTRIBUTION : N groupes de GROUP_SIZE ---
    logger.info(f"\n  --- Mode distribution ({len(groups_dist)} groupes x {GROUP_SIZE}) ---")
    for gamma in GAMMAS:
        actual_gamma = gamma_med_val if gamma is None else gamma
        glabel       = gamma_label(gamma)
        metric_label = f"MMD_{backbone}_l{layer_idx}_g{glabel}"
        try:
            K_rr_g = np.exp(-actual_gamma * D_rr)
            term1  = (np.sum(K_rr_g) - np.trace(K_rr_g)) / (n_r * (n_r - 1))
            del K_rr_g

            scores = [
                mmd2_unbiased(term1, D_rg, D_gg, actual_gamma)
                for D_rg, D_gg in zip(D_rgs, D_ggs)
            ]
            detailed_rows += collect_detailed_rows(
                scores, groups_dist, metric_label, backbone, layer_display, glabel
            )
            summary_rows.append(
                compute_spearman_summary(
                    scores, groups_dist, metric_label,
                    backbone, layer_display, glabel, actual_gamma, GROUP_SIZE
                )
            )
        except Exception as exc:
            logger.error(f"  ERREUR gamma={glabel} : {exc}", exc_info=True)

    # --- MODE POINT : 480 groupes de 1 image ---
    if groups_point is not None:
        logger.info(f"\n  --- Mode point (N=1, {len(groups_point)} images) ---")

        # Extraction features point par point (on concatène tout puis on sépare)
        all_point_paths = [g["paths"][0] for g in groups_point]
        logger.info(f"  Extraction {len(all_point_paths)} images individuelles...")
        all_point_feat = extractor.extract(all_point_paths, fit_transform=False, batch_size=bs)
        logger.info(f"  Features point : {all_point_feat.shape}")

        # Pré-calcul D_ry pour chaque point
        r    = anchor_feat.astype(np.float32)
        r_sq = np.sum(r ** 2, axis=1, keepdims=True)   # (n_r, 1)
        p    = all_point_feat.astype(np.float32)
        p_sq = np.sum(p ** 2, axis=1, keepdims=True)   # (480, 1)
        D_rp = np.maximum(r_sq + p_sq.T - 2.0 * (r @ p.T), 0.0)  # (n_r, 480)

        for gamma in GAMMAS:
            actual_gamma = gamma_med_val if gamma is None else gamma
            glabel       = gamma_label(gamma)
            metric_label = f"MMDpt_{backbone}_l{layer_idx}_g{glabel}"
            try:
                K_rr_g = np.exp(-actual_gamma * D_rr)
                term1  = (np.sum(K_rr_g) - np.trace(K_rr_g)) / (n_r * (n_r - 1))
                del K_rr_g

                scores = [
                    mmd2_point(term1, D_rp[:, i:i+1], actual_gamma)
                    for i in range(len(groups_point))
                ]
                detailed_pt += collect_detailed_rows(
                    scores, groups_point, metric_label, backbone, layer_display, glabel
                )
                summary_pt.append(
                    compute_spearman_summary(
                        scores, groups_point, metric_label,
                        backbone, layer_display, glabel, actual_gamma, 1
                    )
                )
            except Exception as exc:
                logger.error(f"  ERREUR point gamma={glabel} : {exc}", exc_info=True)

    # Nettoyage
    extractor.cleanup()
    del extractor, all_group_feat, anchor_feat, D_rr, D_rgs, D_ggs
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# Evaluation CMMD
# ============================================================

def evaluate_cmmd(anchor_paths: list, device: str,
                  groups_dist: list, groups_point: list | None,
                  detailed_rows: list, summary_rows: list,
                  detailed_pt: list, summary_pt: list):
    logger.info("\n" + "=" * 70)
    logger.info("PHASE CMMD — CLIP ViT-L/14@336 (protocole Google)")
    logger.info("=" * 70)

    clip_model, image_proc, ref_embeds, gamma_clip, term1_clip = setup_cmmd(
        anchor_paths, device
    )
    n_r = ref_embeds.shape[0]
    glabel       = f"{gamma_clip:.2e}"
    metric_label = "CMMD_CLIP_ViT-L14"

    # --- Mode distribution ---
    logger.info(f"  Extraction embeddings CLIP ({len(groups_dist)} groupes x {GROUP_SIZE})...")
    all_group_embeds = []
    for g_idx, grp in enumerate(groups_dist):
        ge = extract_group_clip(clip_model, image_proc, grp["paths"], device)
        all_group_embeds.append(ge)
        if (g_idx + 1) % 10 == 0 or g_idx == 0:
            logger.info(f"    Groupe {g_idx+1}/{len(groups_dist)} : {ge.shape}")

    D_rr_c, D_rgs_c, D_ggs_c = precompute_sq_distances(ref_embeds, all_group_embeds)

    scores = [
        mmd2_unbiased(term1_clip, D_rg, D_gg, gamma_clip)
        for D_rg, D_gg in zip(D_rgs_c, D_ggs_c)
    ]
    detailed_rows += collect_detailed_rows(
        scores, groups_dist, metric_label, "CLIP_ViT-L/14", "768d", glabel
    )
    summary_rows.append(
        compute_spearman_summary(
            scores, groups_dist, metric_label,
            "CLIP_ViT-L/14", "768d", glabel, gamma_clip, GROUP_SIZE
        )
    )

    # --- Mode point ---
    if groups_point is not None:
        logger.info(f"\n  --- Mode point CMMD (N=1, {len(groups_point)} images) ---")
        all_point_paths = [g["paths"][0] for g in groups_point]
        all_point_embeds = extract_group_clip(
            clip_model, image_proc, all_point_paths, device
        )
        logger.info(f"  Embeddings point : {all_point_embeds.shape}")

        r    = ref_embeds.astype(np.float32)
        r_sq = np.sum(r ** 2, axis=1, keepdims=True)
        p    = all_point_embeds.astype(np.float32)
        p_sq = np.sum(p ** 2, axis=1, keepdims=True)
        D_rp = np.maximum(r_sq + p_sq.T - 2.0 * (r @ p.T), 0.0)

        K_rr = np.exp(-gamma_clip * (r_sq + r_sq.T - 2.0 * (r @ r.T)))
        term1_pt = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))

        scores_pt = [
            mmd2_point(term1_pt, D_rp[:, i:i+1], gamma_clip)
            for i in range(len(groups_point))
        ]
        metric_label_pt = "CMMDpt_CLIP_ViT-L14"
        detailed_pt += collect_detailed_rows(
            scores_pt, groups_point, metric_label_pt, "CLIP_ViT-L/14", "768d", glabel
        )
        summary_pt.append(
            compute_spearman_summary(
                scores_pt, groups_point, metric_label_pt,
                "CLIP_ViT-L/14", "768d", glabel, gamma_clip, 1
            )
        )

    del clip_model, image_proc, ref_embeds, all_group_embeds
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Monotonie RAISE — {N_GROUPS} groupes de {GROUP_SIZE} images vs ancrage COCO"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cmmd", action="store_true",
                        help="Ignorer la phase CMMD")
    parser.add_argument("--skip-point", action="store_true",
                        help="Ignorer le mode point (N=1)")
    parser.add_argument("--n-anchor", type=int, default=N_ANCHOR,
                        help=f"Nombre d'images COCO pour l'ancrage (defaut: {N_ANCHOR})")
    return parser.parse_args()


def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_cmmd  = ENABLE_CMMD and not args.skip_cmmd
    run_point = ENABLE_POINT_MODE and not args.skip_point

    logger.info("=" * 70)
    logger.info(f"Test de monotonie RAISE")
    logger.info(f"  Mode distribution : {N_GROUPS} groupes x {GROUP_SIZE} images")
    if run_point:
        logger.info(f"  Mode point        : 480 groupes x 1 image")
    logger.info(f"  Ancrage : {args.n_anchor} images COCO-5000 (seed={ANCHOR_SEED})")
    logger.info(f"  Backbones MMD : {[(b, l) for b, l, _ in MMD_BACKBONE_CONFIGS]}")
    if run_cmmd:
        logger.info(f"  CMMD : {CLIP_MODEL_ID} (gamma={CMMD_GAMMA})")
    logger.info(f"  Gammas  : {[gamma_label(g) for g in GAMMAS]}")
    logger.info(f"  Device  : {args.device}")
    logger.info("=" * 70)

    # Chargement
    anchor_paths = load_coco_anchor(n=args.n_anchor)
    mos_dict     = load_mos_dict()
    fake_items   = load_raise_fake(mos_dict)

    # Groupes distribution
    logger.info(f"\nFormation des {N_GROUPS} groupes de {GROUP_SIZE} images...")
    groups_dist = form_groups(fake_items, GROUP_SIZE, N_GROUPS)
    for i, g in enumerate(groups_dist):
        if i % 10 == 0:
            logger.info(f"  Groupe {i+1:3d} : MOS [{g['mos_r_min']:.1f} ... {g['mos_r_max']:.1f}]  "
                        f"moy={g['mos_r']:.1f}")

    # Groupes point (N=1)
    groups_point = None
    if run_point:
        groups_point = form_point_groups(fake_items)
        logger.info(f"Mode point : {len(groups_point)} groupes de 1 image")

    # Résultats
    detailed_rows, summary_rows = [], []
    detailed_pt,   summary_pt   = [], []

    # PHASE 1 : MMD_RBF par backbone
    for backbone, layer_idx, layer_display in MMD_BACKBONE_CONFIGS:
        try:
            evaluate_backbone_mmd(
                backbone, layer_idx, layer_display, anchor_paths,
                groups_dist, groups_point,
                detailed_rows, summary_rows,
                detailed_pt, summary_pt,
            )
        except Exception as exc:
            logger.error(f"  ERREUR backbone {backbone} : {exc}", exc_info=True)

    # PHASE 2 : CMMD
    if run_cmmd:
        try:
            evaluate_cmmd(
                anchor_paths, args.device,
                groups_dist, groups_point,
                detailed_rows, summary_rows,
                detailed_pt, summary_pt,
            )
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)
    else:
        logger.info("\nCMMD ignoré")

    # Sauvegarde mode distribution
    if summary_rows:
        logger.info("\n\n" + "=" * 70)
        logger.info("RÉSULTATS MODE DISTRIBUTION")
        logger.info("=" * 70)
        save_results(detailed_rows, summary_rows, ts, args.n_anchor, N_GROUPS, GROUP_SIZE)

    # Sauvegarde mode point
    if summary_pt:
        logger.info("\n\n" + "=" * 70)
        logger.info("RÉSULTATS MODE POINT (N=1)")
        logger.info("=" * 70)
        save_results(detailed_pt, summary_pt, f"{ts}_point", args.n_anchor,
                     len(groups_point), 1)


if __name__ == "__main__":
    main()
