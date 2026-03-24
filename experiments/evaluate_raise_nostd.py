"""
evaluate_raise_nostd.py
=======================
Test de monotonicité MOS <-> distance sur le dataset RAISE.
Variante SANS standardisation des vecteurs Gram.

Protocole :
  - 40 groupes de 12 images (distribution vs distribution, MMD² non-biaisé)
  - Ancrage COCO : 1000 images
  - Aucune normalisation / standardisation des vecteurs Gram extraits
  - Gamma : heuristique médiane (+ quelques valeurs fixes de référence)

Comparaison :
  - SD-VAE   l9  (backbone VAE SD)
  - DC-AE    l10 (backbone DC-AE f64c128)
  - DINOv2-B l5  (backbone DINOv2)
  - CMMD     CLIP ViT-L/14@336 (protocole Google — σ=10, sans normalisation)

Convention signe :
  MOS élevé = image réaliste -> proche des réels -> MMD faible
  -> rho(MMD, MOS) < 0  ->  on rapporte -rho (positif si métrique cohérente)

Usage :
    python experiments/evaluate_raise_nostd.py
    python experiments/evaluate_raise_nostd.py --skip-cmmd
    python experiments/evaluate_raise_nostd.py --device cpu

Résultats -> results/raise_nostd/
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
from scipy.stats import spearmanr, t as t_dist

from gram_mmd.config import BACKBONE_CONFIGS
from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION EXPÉRIENCE                       ║
# ╚════════════════════════════════════════════════════════════════════╝

BASE_DIR            = Path(__file__).resolve().parent.parent
RAISE_ROOT          = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_TRAIN_IMAGES  = RAISE_ROOT / "images" / "train_images"
RAISE_TEST_IMAGES   = RAISE_ROOT / "images" / "test_images"
RAISE_TRAIN_RATINGS = RAISE_ROOT / "ratings" / "train.csv"
RAISE_TEST_RATINGS  = RAISE_ROOT / "ratings" / "test.csv"
COCO_DIR            = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR             = BASE_DIR / "results" / "raise_nostd"

# --- Ancrage ---
N_ANCHOR    = 1000
ANCHOR_SEED = 42

# --- Mode point (N=1) : 480 images individuelles vs ancrage ---
GROUP_SIZE = 1    # 1 image par groupe (point vs distribution, MMD² biaisé)
N_GROUPS   = 480  # 480 images RAISE

# --- Backbones MMD (backbone, layer_index, display_name) ---
MMD_BACKBONE_CONFIGS = [
    ("sd_vae",        9,  "sd_vae_l9"),
    ("dc_ae",         10, "dc_ae_l10"),
    ("dinov2_vitb14", 5,  "dinov2_l5"),
]

# --- Gammas ---
# None = heuristique mediane (adaptatif, recommande sans standardisation)
# Valeurs fixes pour comparaison (sans std, les distances sont plus grandes)
GAMMAS = [None, 1e-7, 1e-6, 1e-5, 1e-4]

# --- CMMD ---
ENABLE_CMMD   = True
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005   # sigma=10 -> gamma = 1/(2*sigma^2) = 0.005

# --- Batch sizes ---
BATCH_SIZE_DEFAULT = 64
BATCH_SIZE_LARGE   = 8   # DC-AE, SD-VAE (512x512)
BATCH_SIZE_CLIP    = 32

# ╔════════════════════════════════════════════════════════════════════╗
# ║                  FIN DE LA CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════╝


def gamma_label(gamma):
    return "median" if gamma is None else f"{gamma:.2e}"


def get_batch_size(backbone: str) -> int:
    if backbone in ("dc_ae", "flux_vae", "sd_vae"):
        return BATCH_SIZE_LARGE
    return BATCH_SIZE_DEFAULT


# ============================================================
# Chargement des donnees
# ============================================================

def load_coco_anchor(n: int = N_ANCHOR, seed: int = ANCHOR_SEED) -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if not all_imgs:
        raise FileNotFoundError(f"Aucune image trouvee dans {COCO_DIR}")
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
    """480 images IA RAISE (f*.png), triees par MOS_Rating croissant."""
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
        f"Attendu {group_size * n_groups} images, trouve {len(fake_items)}"
    groups = []
    for i in range(n_groups):
        g = fake_items[i * group_size : (i + 1) * group_size]
        groups.append({
            "paths":     [x[0] for x in g],
            "mos_r":     float(np.mean([x[1] for x in g])),
            "mos_z":     float(np.mean([x[2] for x in g])),
            "mos_r_min": g[0][1],
            "mos_r_max": g[-1][1],
        })
    return groups


# ============================================================
# Extraction de features — SANS standardisation
# ============================================================

def setup_gram_extractor_nostd(backbone: str, layer_idx: int, layer_name: str,
                                anchor_paths: list):
    """
    Cree un FeatureExtractor gram_spatial et extrait les features d'ancrage
    SANS StandardScaler (fit_transform=False).
    """
    batch_size = get_batch_size(backbone)
    logger.info(f"\n[MMD_RBF_NoStd] {backbone}/layer{layer_idx} ({layer_name})  "
                f"batch_size={batch_size}")

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

    logger.info(f"  Extraction ancrage ({len(anchor_paths)} images) — sans standardisation...")
    # fit_transform=False -> aucun StandardScaler applique
    anchor_feat = extractor.extract(anchor_paths, fit_transform=False, batch_size=batch_size)
    logger.info(f"  Features ancrage : {anchor_feat.shape}  "
                f"mean={anchor_feat.mean():.4e}  std={anchor_feat.std():.4e}")
    return extractor, anchor_feat, batch_size


# ============================================================
# Distances pre-calculees
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


def mmd2_point(term1_rr: float, K_ry: np.ndarray) -> float:
    """MMD² biaise pour 1 point y vs distribution X (N_y = 1).
    score = mean_Krr + k(y,y) - 2/n * sum(K_ry)
    k(y,y) = exp(0) = 1
    K_ry : vecteur (n_r,) des valeurs de kernel entre y et chaque ancre.
    """
    n_r   = len(K_ry)
    term3 = 2.0 * float(K_ry.sum()) / n_r
    return float(term1_rr + 1.0 - term3)


# ============================================================
# Spearman
# ============================================================

def compute_spearman_summary(scores: list, groups: list,
                              metric_label: str, backbone: str,
                              layer_name: str, gamma_str: str,
                              gamma_eff: float, group_size: int) -> dict:
    s  = np.array(scores)
    mr = np.array([g["mos_r"] for g in groups])
    mz = np.array([g["mos_z"] for g in groups])

    rho_r, p_r = spearmanr(s, mr)
    rho_z, p_z = spearmanr(s, mz)

    def verdict(neg_rho):
        if   neg_rho >  0.5: return "bon"
        elif neg_rho >  0.2: return "partiel"
        elif neg_rho > -0.2: return "neutre"
        else:                return "inverse"

    logger.info(
        f"  {metric_label:60s}  "
        f"MOS_R: rho={rho_r:+.4f} -rho={-rho_r:+.4f} p={p_r:.3e}  |  "
        f"MOS_Z: rho={rho_z:+.4f} -rho={-rho_z:+.4f} p={p_z:.3e}"
    )
    return {
        "metric_label":       metric_label,
        "backbone":           backbone,
        "layer_name":         layer_name,
        "gamma_str":          gamma_str,
        "gamma_eff":          f"{gamma_eff:.6e}",
        "n_groups":           len(scores),
        "group_size":         group_size,
        "rho_mos_rating":     float(rho_r),
        "neg_rho_mos_rating": float(-rho_r),
        "p_mos_rating":       float(p_r),
        "verdict_mos_rating": verdict(-rho_r),
        "rho_mos_zscore":     float(rho_z),
        "neg_rho_mos_zscore": float(-rho_z),
        "p_mos_zscore":       float(p_z),
        "verdict_mos_zscore": verdict(-rho_z),
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


def collect_detailed_rows(scores, groups, metric_label, backbone, layer_name, gamma_str):
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
    tag = f"coco{n_anchor}_g{n_groups}x{group_size}_nostd"

    det_path = OUT_DIR / f"raise_nostd_{tag}_detailed_{ts}.csv"
    with open(det_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETAILED_FIELDS)
        w.writeheader()
        w.writerows(detailed_rows)
    logger.info(f"CSV detaille : {det_path}")

    sum_path = OUT_DIR / f"raise_nostd_{tag}_summary_{ts}.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(f"CSV resume  : {sum_path}")

    t_crit   = t_dist.ppf(0.975, df=n_groups - 2)
    rho_crit = t_crit / math.sqrt(t_crit ** 2 + n_groups - 2)

    sep = "-" * 150
    logger.info("\n" + sep)
    logger.info(
        f"RESUME SANS STANDARDISATION — "
        f"MMD(groupe_{group_size}, ancrage_{n_anchor}_COCO) vs MOS  |  "
        f"n={n_groups} groupes  |  seuil sig. p<0.05 : |rho| > {rho_crit:.3f}"
    )
    logger.info(
        f"  {'Metrique':60s}  {'gamma':>10s}  "
        f"{'rho(MOS_R)':>9s} {'-rho(MOS_R)':>11s} {'p':>9s}  "
        f"{'rho(MOS_Z)':>9s} {'-rho(MOS_Z)':>11s}  {'verdict':>8s}"
    )
    logger.info(sep)
    for r in sorted(summary_rows, key=lambda x: -x["neg_rho_mos_rating"]):
        sig  = "*" if r["p_mos_rating"] < 0.05 else " "
        icon = "v" if r["neg_rho_mos_rating"] > 0.5 else (
               "~" if r["neg_rho_mos_rating"] > 0.2 else "x")
        logger.info(
            f"  {r['metric_label']:60s}  {r['gamma_str']:>10s}  "
            f"{r['rho_mos_rating']:+9.4f} {r['neg_rho_mos_rating']:+11.4f} "
            f"{r['p_mos_rating']:>9.3e}{sig} "
            f"{r['rho_mos_zscore']:+9.4f} {r['neg_rho_mos_zscore']:+11.4f}  "
            f"{r['verdict_mos_rating']:>8s}  {icon}"
        )
    logger.info(sep)
    logger.info(f"\nResultats dans : {OUT_DIR}")


# ============================================================
# Evaluation MMD — backbone (sans standardisation)
# ============================================================

def evaluate_backbone_mmd(backbone: str, layer_idx: int, layer_display: str,
                           anchor_paths: list, groups: list,
                           detailed_rows: list, summary_rows: list):
    """Mode point N=1 : 480 images individuelles vs ancrage COCO (MMD² biaisé)."""
    logger.info("\n" + "=" * 70)
    logger.info(f"BACKBONE — {backbone} / layer {layer_idx} ({layer_display})  [sans std, point]")
    logger.info("=" * 70)

    extractor, anchor_feat, bs = setup_gram_extractor_nostd(
        backbone, layer_idx, layer_display, anchor_paths
    )

    # Extraction de toutes les 480 images en une passe
    all_paths = [g["paths"][0] for g in groups]
    logger.info(f"  Extraction {len(all_paths)} images RAISE (une passe)...")
    query_feat = extractor.extract(all_paths, fit_transform=False, batch_size=bs)
    logger.info(f"  Features requetes : {query_feat.shape}")

    # Distances pre-calculees
    r    = anchor_feat.astype(np.float32)
    q    = query_feat.astype(np.float32)
    r_sq = np.einsum("ij,ij->i", r, r)   # (n_r,)
    q_sq = np.einsum("ij,ij->i", q, q)   # (480,)
    # D_rr : (n_r, n_r)
    D_rr = np.maximum(r_sq[:, None] + r_sq[None, :] - 2.0 * (r @ r.T), 0.0)
    # D_rq : (n_r, 480)  — distances ancre -> requete
    D_rq = np.maximum(r_sq[:, None] + q_sq[None, :] - 2.0 * (r @ q.T), 0.0)

    triu          = D_rr[np.triu_indices(D_rr.shape[0], k=1)]
    median_dist   = float(np.median(triu))
    gamma_med_val = 1.0 / (2.0 * median_dist + 1e-8)
    logger.info(f"  Distance mediane D_rr   = {median_dist:.6e}")
    logger.info(f"  Gamma mediane (1/2·med) = {gamma_med_val:.6e}")

    n_r = r.shape[0]

    for gamma in GAMMAS:
        actual_gamma = gamma_med_val if gamma is None else gamma
        glabel       = gamma_label(gamma)
        metric_label = f"MMDpt_nostd_{backbone}_l{layer_idx}_{glabel}"
        try:
            # term1 biaise = mean(K_rr) (inclut diagonale = 1)
            K_rr_g = np.exp(-actual_gamma * D_rr)
            term1  = float(K_rr_g.mean())
            del K_rr_g

            # K_rq : (n_r, 480)
            K_rq = np.exp(-actual_gamma * D_rq)
            k_sample = float(K_rq.mean())
            logger.info(f"    gamma={glabel:>10s}  K_rq_mean={k_sample:.4f}  term1={term1:.6f}")

            # score[i] = term1 + 1 - 2/n_r * sum(K_rq[:,i])
            scores = [
                mmd2_point(term1, K_rq[:, i])
                for i in range(len(groups))
            ]
            del K_rq
            detailed_rows += collect_detailed_rows(
                scores, groups, metric_label, backbone, layer_display, glabel
            )
            summary_rows.append(
                compute_spearman_summary(
                    scores, groups, metric_label, backbone,
                    layer_display, glabel, actual_gamma, GROUP_SIZE
                )
            )
        except Exception as exc:
            logger.error(f"  ERREUR gamma={glabel} : {exc}", exc_info=True)

    extractor.cleanup()
    del extractor, anchor_feat, query_feat, r, q, D_rr, D_rq
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# Evaluation CMMD (Google — sigma=10, sans normalisation)
# ============================================================

def evaluate_cmmd(anchor_paths: list, device: str,
                  groups: list,
                  detailed_rows: list, summary_rows: list):
    """
    CMMD tel que defini dans le papier Google :
      k(x, y) = exp(-||x - y||^2 / (2*sigma^2))  avec sigma=10
      gamma = 1/(2*10^2) = 0.005
      Embeddings CLIP ViT-L/14@336, aucune normalisation.
    """
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    logger.info("\n" + "=" * 70)
    logger.info("CMMD — CLIP ViT-L/14@336 (Google, sigma=10, sans normalisation)")
    logger.info("=" * 70)

    logger.info(f"  Chargement {CLIP_MODEL_ID}...")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def extract_clip(paths: list) -> np.ndarray:
        all_emb = []
        for i in range(0, len(paths), BATCH_SIZE_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE_CLIP]]
            pv    = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                all_emb.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
        return np.concatenate(all_emb, axis=0)

    logger.info(f"  Extraction ancrage ({len(anchor_paths)} images)...")
    ref_embeds = extract_clip(anchor_paths)
    logger.info(f"  Embeddings ancrage : {ref_embeds.shape}  "
                f"mean={ref_embeds.mean():.4e}  std={ref_embeds.std():.4e}")

    r_sq  = np.sum(ref_embeds ** 2, axis=1, keepdims=True)
    D_rr  = np.maximum(r_sq + r_sq.T - 2.0 * (ref_embeds @ ref_embeds.T), 0.0)
    n_r   = ref_embeds.shape[0]
    K_rr  = np.exp(-CMMD_GAMMA * D_rr)
    term1 = (np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1))
    del K_rr

    triu_d      = D_rr[np.triu_indices(n_r, k=1)]
    median_dist = float(np.median(triu_d))
    logger.info(f"  CMMD gamma=0.005  sigma=10  median_D_rr={median_dist:.4e}  term1={term1:.6f}")

    # Extraction des 480 images en une passe
    all_paths    = [g["paths"][0] for g in groups]
    logger.info(f"  Extraction 480 embeddings RAISE...")
    query_embeds = extract_clip(all_paths)
    logger.info(f"  Embeddings requetes : {query_embeds.shape}")

    r_sq = np.sum(ref_embeds ** 2, axis=1)     # (n_r,)
    q_sq = np.sum(query_embeds ** 2, axis=1)   # (480,)
    D_rq = np.maximum(r_sq[:, None] + q_sq[None, :] - 2.0 * (ref_embeds @ query_embeds.T), 0.0)

    K_rq   = np.exp(-CMMD_GAMMA * D_rq)
    glabel = f"{CMMD_GAMMA:.2e}"
    metric_label = "CMMDpt_CLIP_ViT-L14_sigma10"

    # term1 biaise = mean(K_rr)
    K_rr_c = np.exp(-CMMD_GAMMA * (r_sq[:, None] + r_sq[None, :] - 2.0 * (ref_embeds @ ref_embeds.T)))
    term1_c = float(K_rr_c.mean())
    del K_rr_c

    scores = [mmd2_point(term1_c, K_rq[:, i]) for i in range(len(groups))]
    del K_rq

    detailed_rows += collect_detailed_rows(
        scores, groups, metric_label, "CLIP_ViT-L14", "768d", glabel
    )
    summary_rows.append(
        compute_spearman_summary(
            scores, groups, metric_label, "CLIP_ViT-L14",
            "768d", glabel, CMMD_GAMMA, GROUP_SIZE
        )
    )

    del model, proc, ref_embeds, query_embeds
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Monotonie RAISE sans std — {N_GROUPS} groupes de {GROUP_SIZE} images"
    )
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cmmd", action="store_true", help="Ignorer CMMD")
    parser.add_argument("--n-anchor",  type=int, default=N_ANCHOR,
                        help=f"Nombre d'images COCO (defaut: {N_ANCHOR})")
    return parser.parse_args()


def main():
    args   = parse_args()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = args.device

    logger.info("=" * 70)
    logger.info("Test de monotonicite RAISE — SANS STANDARDISATION — MODE POINT (N=1)")
    logger.info(f"  {N_GROUPS} images individuelles (point vs distribution, MMD² biaise)")
    logger.info(f"  Ancrage : {args.n_anchor} images COCO-5000 (seed={ANCHOR_SEED})")
    logger.info(f"  Backbones : {[(b, l) for b, l, _ in MMD_BACKBONE_CONFIGS]}")
    logger.info(f"  Gammas  : {[gamma_label(g) for g in GAMMAS]}")
    logger.info(f"  CMMD    : {'active' if ENABLE_CMMD and not args.skip_cmmd else 'desactive'}")
    logger.info(f"  Device  : {device}")
    logger.info("=" * 70)

    anchor_paths = load_coco_anchor(n=args.n_anchor)
    mos_dict     = load_mos_dict()
    fake_items   = load_raise_fake(mos_dict)

    logger.info(f"\nFormation des {N_GROUPS} groupes de {GROUP_SIZE} images...")
    groups = form_groups(fake_items, GROUP_SIZE, N_GROUPS)
    for i, g in enumerate(groups):
        if i % 10 == 0:
            logger.info(f"  Groupe {i+1:3d} : MOS [{g['mos_r_min']:.1f} ... {g['mos_r_max']:.1f}]  "
                        f"moy={g['mos_r']:.1f}")

    detailed_rows, summary_rows = [], []

    # PHASE 1 — MMD Gram sans standardisation
    for backbone, layer_idx, layer_display in MMD_BACKBONE_CONFIGS:
        try:
            evaluate_backbone_mmd(
                backbone, layer_idx, layer_display, anchor_paths, groups,
                detailed_rows, summary_rows,
            )
        except Exception as exc:
            logger.error(f"ERREUR backbone {backbone} : {exc}", exc_info=True)

    # PHASE 2 — CMMD
    run_cmmd = ENABLE_CMMD and not args.skip_cmmd
    if run_cmmd:
        try:
            evaluate_cmmd(anchor_paths, device, groups, detailed_rows, summary_rows)
        except Exception as exc:
            logger.error(f"ERREUR CMMD : {exc}", exc_info=True)
    else:
        logger.info("\nCMMD ignore (--skip-cmmd)")

    if summary_rows:
        save_results(detailed_rows, summary_rows, ts, args.n_anchor, N_GROUPS, GROUP_SIZE)
    else:
        logger.warning("Aucun resultat a sauvegarder.")


if __name__ == "__main__":
    main()
