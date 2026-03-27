"""
evaluate_kadid_coco_distdist.py
================================
KADID-10k  —  distribution-vs-distribution MMD²
avec 1000 images COCO comme ancrage.

Groupage :
  - 81 images de référence × 125 distorsions = 10 125 images dégradées
  - Pour chaque référence, tri par DMOS croissant (pire → meilleur)
  - 125 groupes × 81 images :
      groupe k = la k-ème image (rangée par DMOS) de chacune des 81 refs
  - mean_MOS[k] = moyenne des DMOS au rang k  (croissant)

Score MMD² non biaisé dist-vs-dist :
  MMD²_u(G_k, COCO) = mean_K_qq_triu + mean_K_rr_triu - 2 · mean_K_rq

Backbones : SD-VAE L11, DC-AE L8, DINOv2 L3
Gammas    : γ/10, γ_med, 10×γ_med  (γ_med = heuristique médiane sur ancrage)
Référence : CMMD (CLIP ViT-L/14@336, γ fixe = 0.005)

Optimisation :
  - Extraction features : UNE SEULE passe par backbone (ancrage + 10 125 KADID)
  - Distances : pré-calculées une fois, réutilisées pour les 3 gammas
  - GPU kernel : tous les 3 gammas évalués en parallèle

Usage :
    .venv/bin/python experiments/evaluate_kadid_coco_distdist.py
    .venv/bin/python experiments/evaluate_kadid_coco_distdist.py --skip-cmmd
"""

import argparse
import csv
import gc
import glob as glob_module
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import spearmanr, kendalltau

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor, ImageDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
COCO_DIR   = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR    = BASE_DIR / "results" / "kadid_coco_distdist"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR  = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
CACHE_DIR  = OUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

N_ANCHOR    = 1000
ANCHOR_SEED = 42
N_GROUPS    = 125   # rangs de distorsion
N_PER_GROUP = 81    # une image par référence

# (backbone, layer, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",        11, "SD-VAE L11",   4),
    ("dc_ae",          8, "DC-AE L8",     2),
    ("dinov2_vitb14",  3, "DINOv2 L3",    4),
]

GAMMA_MULTIPLIERS = {
    "div10":  0.1,    # γ/10
    "median": 1.0,    # γ_med
    "10x":   10.0,    # 10 × γ_med
}

CMMD_GAMMA      = 0.005
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
BATCH_SIZE_CLIP = 32

# ═══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════

def load_kadid_groups():
    """
    Retourne :
      groups    : liste de N_GROUPS listes, chaque sous-liste = (dmos, path) × N_PER_GROUP
      mean_mos  : np.ndarray (125,) — DMOS moyen par rang (croissant)

    Rang 0   = MOS le plus bas  (pire qualité)
    Rang 124 = MOS le plus haut (meilleure qualité)
    """
    rows = []
    with open(DMOS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "path":    str(IMAGES_DIR / r["dist_img"]),
                "ref_img": r["ref_img"],
                "dmos":    float(r["dmos"]),
            })
    logger.info(f"KADID CSV : {len(rows)} images dégradées")

    by_ref = defaultdict(list)
    for r in rows:
        by_ref[r["ref_img"]].append((r["dmos"], r["path"]))

    ref_ids = sorted(by_ref.keys())
    n_refs = len(ref_ids)
    if n_refs != N_PER_GROUP:
        raise ValueError(f"Expected {N_PER_GROUP} refs, got {n_refs}")

    for ref_id in ref_ids:
        by_ref[ref_id].sort(key=lambda x: x[0])
        if len(by_ref[ref_id]) != N_GROUPS:
            raise ValueError(f"Ref {ref_id}: expected {N_GROUPS} images, got {len(by_ref[ref_id])}")

    # 125 groupes × 81 images
    groups = []
    mean_mos = []
    for k in range(N_GROUPS):
        grp = [(by_ref[ref_id][k][0], by_ref[ref_id][k][1]) for ref_id in ref_ids]
        groups.append(grp)
        mean_mos.append(float(np.mean([x[0] for x in grp])))

    logger.info(f"Groupes : {len(groups)} × {N_PER_GROUP} images")
    logger.info(f"  mean_MOS : {mean_mos[0]:.3f} (rang 0) → {mean_mos[-1]:.3f} (rang 124)")
    return groups, np.array(mean_mos)


def load_coco_anchor():
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = sorted(rng.choice(len(all_imgs), size=N_ANCHOR, replace=False).tolist())
    paths = [all_imgs[i] for i in idx]
    logger.info(f"Ancrage COCO : {len(paths)} images (seed={ANCHOR_SEED})")
    return paths


# ═══════════════════════════════════════════════════════════════════════
#  CACHE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _cache_key(backbone, layer):
    return f"{backbone}_L{layer:02d}"


def load_features_cache(backbone, layer):
    key  = _cache_key(backbone, layer)
    fanc = CACHE_DIR / f"{key}_anc.npy"
    fdst = CACHE_DIR / f"{key}_dist.npy"
    if fanc.exists() and fdst.exists():
        logger.info(f"  Cache features trouvé : {key}")
        # feat_dist ouvert en mmap : les tranches sont lues à la demande (pas de 5GB en RAM)
        return np.load(fanc), np.load(fdst, mmap_mode='r')
    return None, None


def save_features_cache(backbone, layer, feat_anc):
    """Sauvegarde uniquement feat_anc — feat_dist est déjà sur disque (mmap)."""
    key = _cache_key(backbone, layer)
    np.save(CACHE_DIR / f"{key}_anc.npy", feat_anc)
    logger.info(f"  Cache feat_anc sauvegardé : {key}")


def load_distances_cache(backbone, layer):
    key = _cache_key(backbone, layer)
    path = CACHE_DIR / f"{key}_distances.npz"
    if path.exists():
        logger.info(f"  Cache distances trouvé : {key}")
        d = np.load(path)
        return d["D_rr_triu"], d["D_qq_triu"], d["D_rq"]
    return None, None, None


def save_distances_cache(backbone, layer, D_rr_triu, D_qq_triu, D_rq):
    key = _cache_key(backbone, layer)
    np.savez_compressed(
        CACHE_DIR / f"{key}_distances.npz",
        D_rr_triu=D_rr_triu, D_qq_triu=D_qq_triu, D_rq=D_rq,
    )
    logger.info(f"  Cache distances sauvegardé : {key}")


def load_cmmd_cache():
    path = CACHE_DIR / "cmmd_scores.npz"
    if path.exists():
        logger.info("  Cache CMMD trouvé")
        d = np.load(path)
        return {k: d[k] for k in d.files}
    return None


def save_cmmd_cache(scores: dict):
    np.savez_compressed(CACHE_DIR / "cmmd_scores.npz", **scores)
    logger.info("  Cache CMMD sauvegardé")


# ═══════════════════════════════════════════════════════════════════════
#  EXTRACTION FEATURES
# ═══════════════════════════════════════════════════════════════════════

def make_extractor(backbone, layer, device):
    return FeatureExtractor(
        backbone=backbone, layer=layer,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )


def extract_all(ext, anchor_paths, all_dist_paths, batch_size, dist_cache_path):
    """
    Extraction ancrage + KADID.
    feat_dist est écrit INCRÉMENTALEMENT sur disque via mmap :
    jamais plus de (batch_size × D) features en RAM pour KADID.
    Retourne feat_anc (RAM, ~500MB) et feat_dist (mmap, lecture par tranche).
    """
    t0 = time.time()
    feat_anc = ext.extract(anchor_paths, fit_transform=True,
                           batch_size=batch_size).astype(np.float32)
    logger.info(f"  Ancrage  : {feat_anc.shape}  ({time.time()-t0:.1f}s)")

    D      = feat_anc.shape[1]
    n_dist = len(all_dist_paths)

    # Tableau mmap : écrit sur disque au fur et à mesure, jamais 5GB en RAM
    feat_dist = np.lib.format.open_memmap(
        str(dist_cache_path), mode='w+', dtype=np.float32, shape=(n_dist, D)
    )

    dataset = ImageDataset(all_dist_paths, transform=ext.image_transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=(ext.device == 'cuda'),
                         persistent_workers=True)

    idx = 0
    t1 = time.time()
    for batch_imgs, _ in loader:
        raw    = ext.extract_batch(batch_imgs)
        scaled = ext.transform_features(raw, fit_pca=False).astype(np.float32)
        feat_dist[idx:idx + scaled.shape[0]] = scaled
        idx += scaled.shape[0]
        if idx % 2000 < batch_size or idx >= n_dist:
            logger.info(f"    KADID : {idx}/{n_dist} ({idx/n_dist:.0%})  {time.time()-t1:.0f}s")

    feat_dist.flush()
    logger.info(f"  KADID    : {feat_dist.shape}  ({time.time()-t1:.1f}s)")
    return feat_anc, feat_dist  # feat_dist : mmap, données lues par tranche


def compute_gammas(feat_anc: np.ndarray) -> dict:
    """γ heuristique médiane sur les paires de l'ancrage."""
    anc_sq = np.einsum("ij,ij->i", feat_anc, feat_anc)
    D = np.maximum(
        anc_sq[:, None] + anc_sq[None, :] - 2.0 * (feat_anc @ feat_anc.T), 0.0
    )
    triu = D[np.triu_indices(len(feat_anc), k=1)]
    triu = triu[triu > 0]
    g_med = float(1.0 / (2.0 * np.median(triu) + 1e-8))
    logger.info(f"  median(d²) = {np.median(triu):.4e}  →  γ_med = {g_med:.4e}")
    gammas = {tag: mult * g_med for tag, mult in GAMMA_MULTIPLIERS.items()}
    for tag, g in gammas.items():
        logger.info(f"    {tag:8s}: γ = {g:.4e}")
    return gammas


# ═══════════════════════════════════════════════════════════════════════
#  DISTANCES PRÉ-CALCULÉES (communes aux 3 gammas)
# ═══════════════════════════════════════════════════════════════════════

GPU_CHUNK = 250   # lignes d'ancrage traitées en une fois sur GPU


def sq_dist_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance L2² sur CPU (fallback)."""
    return np.maximum(
        (A * A).sum(1, keepdims=True) + (B * B).sum(1, keepdims=True).T
        - 2.0 * (A @ B.T), 0.0
    )


def precompute_distances(feat_anc, feat_dist_grouped, device):
    """
    feat_anc           : (1000, D)  — numpy (ou mmap, copié par tranche)
    feat_dist_grouped  : (125, 81, D) — numpy ou mmap

    Calcul sur GPU si disponible (torch.cdist), avec découpage en chunks
    pour les 1000 lignes d'ancrage (évite l'OOM VRAM).

    Retourne :
      D_rr_triu  : (499500,)       — intra-ancrage upper-triangle
      D_qq_triu  : (125, 3240)     — intra-groupe upper-triangle
      D_rq       : (125, 1000, 81) — distances croisées ancrage × groupe
    """
    n_anc   = feat_anc.shape[0]           # 1000
    n_q     = feat_dist_grouped.shape[1]  # 81
    n_pairs = n_q * (n_q - 1) // 2       # 3240
    triu_qq  = np.triu_indices(n_q,   k=1)
    triu_anc = np.triu_indices(n_anc, k=1)

    # Copie contiguë depuis mmap pour les transferts GPU
    anc_np = np.ascontiguousarray(feat_anc, dtype=np.float32)

    t0 = time.time()

    # ── D_rr (ancrage × ancrage) ────────────────────────────────────
    if device == "cuda":
        anc_t   = torch.from_numpy(anc_np).to(device)
        D_rr    = torch.cdist(anc_t, anc_t, p=2).pow(2)
        D_rr_triu = D_rr[triu_anc[0], triu_anc[1]].cpu().numpy().astype(np.float32)
        del D_rr, anc_t
        torch.cuda.empty_cache()
    else:
        D_rr_triu = sq_dist_cpu(anc_np, anc_np)[triu_anc].astype(np.float32)
    logger.info(f"  D_rr done ({time.time()-t0:.1f}s)")

    D_qq_triu = np.empty((N_GROUPS, n_pairs), dtype=np.float32)
    D_rq      = np.empty((N_GROUPS, n_anc, n_q), dtype=np.float32)

    for k in range(N_GROUPS):
        # Force copie depuis mmap → array C-contiguë
        fq_np = np.ascontiguousarray(feat_dist_grouped[k], dtype=np.float32)

        if device == "cuda":
            fq_t = torch.from_numpy(fq_np).to(device)

            # D_qq : (81, 81)
            D_qq = torch.cdist(fq_t, fq_t, p=2).pow(2)
            D_qq_triu[k] = D_qq[triu_qq[0], triu_qq[1]].cpu().numpy()
            del D_qq

            # D_rq : ancrage en chunks → (GPU_CHUNK, 81) à la fois
            for i in range(0, n_anc, GPU_CHUNK):
                chunk = torch.from_numpy(anc_np[i:i + GPU_CHUNK]).to(device)
                D_rq[k, i:i + GPU_CHUNK] = (
                    torch.cdist(chunk, fq_t, p=2).pow(2).cpu().numpy()
                )
                del chunk
            del fq_t
            torch.cuda.empty_cache()
        else:
            D_qq = sq_dist_cpu(fq_np, fq_np)
            D_qq_triu[k] = D_qq[triu_qq]
            D_rq[k] = sq_dist_cpu(anc_np, fq_np)

        if (k + 1) % 50 == 0 or k + 1 == N_GROUPS:
            logger.info(f"    distances : {k+1}/{N_GROUPS}  ({time.time()-t0:.1f}s)")

    logger.info(f"  D_rr_triu {D_rr_triu.shape}  D_qq_triu {D_qq_triu.shape}  D_rq {D_rq.shape}")
    return D_rr_triu, D_qq_triu, D_rq


# ═══════════════════════════════════════════════════════════════════════
#  SCORING MMD² DIST-VS-DIST
# ═══════════════════════════════════════════════════════════════════════

def compute_mmd2_groups(D_rr_triu, D_qq_triu, D_rq, gammas: dict, device):
    """
    MMD²_u(groupe_k, COCO) = mean_K_qq_triu + mean_K_rr_triu - 2·mean_K_rq

    Tous les gammas sont évalués en un seul passage GPU.
    Retourne {gamma_tag: np.ndarray (125,)}
    """
    D_rr_t = torch.from_numpy(D_rr_triu).to(device)  # (499500,)
    D_qq_t = torch.from_numpy(D_qq_triu).to(device)  # (125, 3240)
    D_rq_t = torch.from_numpy(D_rq).to(device)        # (125, 1000, 81)

    results = {}
    with torch.no_grad():
        for tag, g in gammas.items():
            gf = float(g)
            K_rr_mean = torch.exp(-gf * D_rr_t).mean()          # scalar
            K_qq_mean = torch.exp(-gf * D_qq_t).mean(dim=1)     # (125,)
            K_rq_mean = torch.exp(-gf * D_rq_t).mean(dim=(1, 2))  # (125,)
            mmd2 = K_qq_mean + K_rr_mean - 2.0 * K_rq_mean      # (125,)
            results[tag] = mmd2.cpu().numpy().astype(np.float64)
            logger.info(
                f"    [{tag}] MMD² : min={results[tag].min():.4e}  "
                f"max={results[tag].max():.4e}  mean={results[tag].mean():.4e}"
            )

    del D_rr_t, D_qq_t, D_rq_t
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  CMMD (CLIP ViT-L/14@336, γ fixe = 0.005)
# ═══════════════════════════════════════════════════════════════════════

def run_cmmd(anchor_paths, groups, device):
    """
    Embedding CLIP des 1000 COCO + 10 125 KADID.
    Scoring MMD²_u dist-vs-dist pour chaque groupe.
    Retourne np.ndarray (125,).
    """
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    logger.info(f"  CMMD : chargement {CLIP_MODEL_ID}...")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def embed_paths(paths_list):
        embs = []
        for i in range(0, len(paths_list), BATCH_SIZE_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths_list[i:i + BATCH_SIZE_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
            if (i // BATCH_SIZE_CLIP + 1) % 50 == 0:
                logger.info(f"    CLIP : {min(i + BATCH_SIZE_CLIP, len(paths_list))}/{len(paths_list)}")
        return np.concatenate(embs, axis=0).astype(np.float32)

    logger.info(f"  CMMD : embedding ancrage ({N_ANCHOR} imgs)...")
    emb_anc = embed_paths(anchor_paths)

    all_paths_flat = [item[1] for grp in groups for item in grp]
    logger.info(f"  CMMD : embedding KADID ({len(all_paths_flat)} imgs)...")
    t0 = time.time()
    emb_all = embed_paths(all_paths_flat)
    logger.info(f"  CMMD : embedding terminé en {time.time()-t0:.1f}s")

    del model, proc
    torch.cuda.empty_cache()
    gc.collect()

    # Reshape (125, 81, D_clip)
    emb_grps = emb_all.reshape(N_GROUPS, N_PER_GROUP, -1)
    del emb_all

    logger.info("  CMMD : calcul γ_med (embeddings CLIP)...")
    gammas = compute_gammas(emb_anc)
    gammas["fixed"] = CMMD_GAMMA   # original paper gamma

    logger.info("  CMMD : pré-calcul distances...")
    D_rr_triu, D_qq_triu, D_rq = precompute_distances(emb_anc, emb_grps, device)
    del emb_anc, emb_grps
    gc.collect()
    mmd2 = compute_mmd2_groups(D_rr_triu, D_qq_triu, D_rq, gammas, device)
    del D_rr_triu, D_qq_triu, D_rq
    torch.cuda.empty_cache()
    return mmd2  # dict {"div10", "median", "10x"}


# ═══════════════════════════════════════════════════════════════════════
#  STATISTIQUES
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(mmd2_scores: np.ndarray, mean_mos: np.ndarray) -> dict:
    """
    mmd2_scores : (125,) — un score par groupe
    mean_mos    : (125,) — DMOS moyen par rang (croissant : pire → meilleur)

    Spearman ρ < 0 attendu (plus dégradé → DMOS bas → MMD² élevé).
    Monotonie stricte : proportion de (k, k+1) où MMD²[k] > MMD²[k+1].
    """
    sp,  sp_p = spearmanr(mmd2_scores, mean_mos)
    kt,  kt_p = kendalltau(mmd2_scores, mean_mos)

    # Fraction de paires consécutives correctement ordonnées
    n_mono = int(np.sum(mmd2_scores[:-1] > mmd2_scores[1:]))
    mono   = n_mono / (N_GROUPS - 1)

    return {
        "spearman":    float(sp),
        "spearman_p":  float(sp_p),
        "kendall":     float(kt),
        "kendall_p":   float(kt_p),
        "mono_strict": mono,
    }


# ═══════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════

BACKBONE_COLORS = {
    "SD-VAE L11": "#1976D2",
    "DC-AE L8":   "#388E3C",
    "DINOv2 L3":  "#F57C00",
    "CMMD":       "#E91E63",
}
GAMMA_MARKERS = {
    "div10":  "v",
    "median": "o",
    "10x":    "s",
    "fixed":  "D",
}
GAMMA_LABELS = {
    "div10":  "γ_med/10",
    "median": "γ_med",
    "10x":    "γ_med×10",
    "fixed":  f"γ={CMMD_GAMMA}",
}
# CMMD gamma tags to include in results (10x excluded)
CMMD_TAGS = {"div10", "median", "fixed"}


def plot_scatter(all_results, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f9f9f9")

    # Points to annotate without arrow (text directly next to point)
    NO_ARROW = {"GMMD SD-VAE L11 γ_med", "GMMD SD-VAE L11 γ_med/10"}

    # Per-annotation offset overrides (xytext in points)
    _offsets = {
        "GMMD DINOv2 L3 γ_med":        (14, -18),
        "GMMD SD-VAE L11 γ_med":        (5, 4),
        "GMMD SD-VAE L11 γ_med/10":     (5, 4),
        f"CMMD γ={CMMD_GAMMA}":         (14, -14),
    }
    _default_offset = (14, 8)

    labeled_backbones = set()
    for entry in all_results:
        bname  = entry.get("backbone_name", "CMMD")
        g_tag  = entry["gamma_tag"]
        sp     = -entry["stats"]["spearman"]
        kt     = -entry["stats"]["kendall"]
        color  = BACKBONE_COLORS.get(bname, "#9C27B0")
        marker = GAMMA_MARKERS.get(g_tag, "o")
        ax.scatter(kt, sp, color=color, marker=marker, s=100,
                   edgecolors="black", linewidths=0.7, zorder=5)

        # Decide whether to annotate this point
        annotate = False
        if g_tag == "median" and bname not in labeled_backbones:
            annotate = True
            labeled_backbones.add(bname)
        elif entry["label"] == "GMMD SD-VAE L11 γ_med/10":
            annotate = True
        elif entry["label"] == f"CMMD γ={CMMD_GAMMA}":
            annotate = True

        if annotate:
            offset = _offsets.get(entry["label"], _default_offset)
            use_arrow = entry["label"] not in NO_ARROW
            ax.annotate(entry["label"], (kt, sp),
                        textcoords="offset points", xytext=offset,
                        fontsize=6.5, zorder=6,
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        lw=0.8, shrinkA=0, shrinkB=3)
                        if use_arrow else None)

    color_handles = [
        Line2D([0],[0], marker="o", color=c, linestyle="None", markersize=8,
               markeredgecolor="black", markeredgewidth=0.6, label=n)
        for n, c in BACKBONE_COLORS.items()
        if any(e.get("backbone_name") == n for e in all_results)
    ]
    marker_handles = [
        Line2D([0],[0], marker=m, color="gray", linestyle="None", markersize=8,
               markeredgecolor="black", markeredgewidth=0.5, label=GAMMA_LABELS[t])
        for t, m in GAMMA_MARKERS.items()
        if any(e["gamma_tag"] == t for e in all_results)
    ]
    leg1 = ax.legend(handles=color_handles, fontsize=8, loc="upper left",
                     title="Backbone", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, fontsize=8, loc="lower right",
              title="γ", framealpha=0.9)

    ax.set_xlabel("-Kendall τ", fontsize=12)
    ax.set_ylabel("-Spearman ρ", fontsize=12)
    ax.set_title(
        "KADID-10k  ·  1000 COCO anchor\n"
        "-Spearman ρ vs -Kendall τ",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / "01_scatter_spearman_vs_mono.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


def plot_tripanel(all_results, out_dir: Path):
    sorted_res = sorted(all_results,
                        key=lambda x: -x["stats"]["spearman"], reverse=True)
    labels  = [e["label"] for e in sorted_res]
    sp_vals = [-e["stats"]["spearman"] for e in sorted_res]
    kt_vals = [-e["stats"]["kendall"]  for e in sorted_res]
    colors  = [BACKBONE_COLORS.get(e.get("backbone_name", "CMMD"), "#9C27B0")
               for e in sorted_res]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(labels))

    def bar_panel(ax, vals, ylabel):
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.18 if vals else 1)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_facecolor("#f9f9f9")

    bar_panel(axes[0], sp_vals, "-Spearman ρ")
    bar_panel(axes[1], kt_vals, "-Kendall τ")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    axes[0].set_title(
        "KADID-10k  ·  1 000 COCO anchor\n"
        "Tri par -Spearman ρ décroissant",
        fontsize=16, fontweight="bold"
    )

    handles = [
        Line2D([0],[0], color=c, linewidth=8, label=n)
        for n, c in BACKBONE_COLORS.items()
        if any(e.get("backbone_name") == n for e in all_results)
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out = out_dir / "02_tripanel_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
#  SAUVEGARDE CSV
# ═══════════════════════════════════════════════════════════════════════

def save_csv(all_results, ts):
    path = OUT_DIR / f"summary_{ts}.csv"
    fields = ["label", "backbone", "layer", "gamma_tag", "gamma_val",
              "spearman", "spearman_p", "kendall", "kendall_p", "mono_strict"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in all_results:
            st = e["stats"]
            w.writerow({
                "label":       e["label"],
                "backbone":    e.get("backbone_name", "CMMD"),
                "layer":       e.get("layer", ""),
                "gamma_tag":   e["gamma_tag"],
                "gamma_val":   f"{e['gamma_val']:.4e}" if e.get('gamma_val') is not None else "",
                "spearman":    f"{st['spearman']:.6f}",
                "spearman_p":  f"{st['spearman_p']:.4e}",
                "kendall":     f"{st['kendall']:.6f}",
                "kendall_p":   f"{st['kendall_p']:.4e}",
                "mono_strict": f"{st['mono_strict']:.6f}",
            })
    logger.info(f"CSV : {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cmmd", action="store_true")
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    t_total = time.time()
    logger.info("=" * 85)
    logger.info("KADID-10k  —  dist-vs-dist MMD²  ·  1 000 COCO anchor")
    logger.info(f"  Device : {args.device}")

    # ── Données ──────────────────────────────────────────────────────
    groups, mean_mos = load_kadid_groups()
    anchor_paths = load_coco_anchor()

    # Chemins KADID dans l'ordre groupe-majeur  [grp0_ref0, grp0_ref1, ..., grp124_ref80]
    all_dist_paths = [item[1] for grp in groups for item in grp]   # 10 125 chemins

    all_results = []

    # ── Backbones ─────────────────────────────────────────────────────
    for backbone, layer, display_name, batch_sz in BACKBONE_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Backbone : {display_name}")

        # ── Features : cache ou extraction ──────────────────────────
        feat_anc, feat_all = load_features_cache(backbone, layer)
        if feat_anc is None:
            ext = make_extractor(backbone, layer, args.device)
            dist_path = CACHE_DIR / f"{_cache_key(backbone, layer)}_dist.npy"
            feat_anc, feat_all = extract_all(
                ext, anchor_paths, all_dist_paths, batch_sz, dist_path
            )
            ext.cleanup()
            del ext
            torch.cuda.empty_cache()
            gc.collect()
            save_features_cache(backbone, layer, feat_anc)
            # feat_all est déjà sur disque (mmap écrit pendant l'extraction)

        # ── Gammas ──────────────────────────────────────────────────
        gammas = compute_gammas(feat_anc)

        # ── Reshape KADID : (125, 81, D) ────────────────────────────
        D = feat_all.shape[1]
        feat_grps = feat_all.reshape(N_GROUPS, N_PER_GROUP, D)
        del feat_all
        gc.collect()

        # ── Distances : cache ou calcul ─────────────────────────────
        D_rr_triu, D_qq_triu, D_rq = load_distances_cache(backbone, layer)
        if D_rr_triu is None:
            logger.info(f"  Pré-calcul distances (GPU)...")
            D_rr_triu, D_qq_triu, D_rq = precompute_distances(
                feat_anc, feat_grps, args.device
            )
            save_distances_cache(backbone, layer, D_rr_triu, D_qq_triu, D_rq)
        del feat_anc, feat_grps
        gc.collect()

        # ── Scoring MMD² pour 3 gammas simultanément ────────────────
        logger.info(f"  Scoring MMD² (3 gammas)...")
        mmd2_per_gamma = compute_mmd2_groups(
            D_rr_triu, D_qq_triu, D_rq, gammas, args.device
        )
        del D_rr_triu, D_qq_triu, D_rq
        torch.cuda.empty_cache()
        gc.collect()

        # ── Stats par gamma ─────────────────────────────────────────
        for tag, scores in mmd2_per_gamma.items():
            stats = compute_stats(scores, mean_mos)
            glabel = GAMMA_LABELS[tag]
            label  = f"GMMD {display_name} {glabel}"
            logger.info(
                f"  [{label}]  "
                f"ρ={stats['spearman']:+.4f}  "
                f"τ={stats['kendall']:+.4f}  "
                f"mono={stats['mono_strict']:.1%}"
            )
            all_results.append({
                "label":         label,
                "backbone_name": display_name,
                "layer":         layer,
                "gamma_tag":     tag,
                "gamma_val":     gammas[tag],
                "stats":         stats,
            })

    # ── CMMD ──────────────────────────────────────────────────────────
    if not args.skip_cmmd:
        logger.info(f"\n{'='*60}")
        logger.info("  CMMD (CLIP ViT-L/14@336, γ_med × {1/10, 1, 10})")
        cmmd_scores = load_cmmd_cache()
        if cmmd_scores is None:
            cmmd_scores = run_cmmd(anchor_paths, groups, args.device)
            save_cmmd_cache(cmmd_scores)
        for tag, scores in cmmd_scores.items():
            if tag not in CMMD_TAGS:
                continue
            stats = compute_stats(scores, mean_mos)
            glabel = GAMMA_LABELS.get(tag, tag)
            label  = f"CMMD {glabel}"
            logger.info(
                f"  [{label}]  "
                f"ρ={stats['spearman']:+.4f}  "
                f"τ={stats['kendall']:+.4f}  "
                f"mono={stats['mono_strict']:.1%}"
            )
            all_results.append({
                "label":         label,
                "backbone_name": "CMMD",
                "layer":         "",
                "gamma_tag":     tag,
                "gamma_val":     None,
                "stats":         stats,
            })

    # ── Sauvegarde ────────────────────────────────────────────────────
    save_csv(all_results, ts)
    plot_scatter(all_results, PLOTS_DIR)
    plot_tripanel(all_results, PLOTS_DIR)

    # ── Résumé console ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  {'Label':<40}  {'Spearman':>9}  {'|ρ|':>6}  {'Kendall':>8}  {'Mono':>7}")
    print("-" * 80)
    for e in sorted(all_results, key=lambda x: abs(x["stats"]["spearman"]), reverse=True):
        st = e["stats"]
        print(
            f"  {e['label']:<40}  "
            f"{st['spearman']:>+9.4f}  "
            f"{abs(st['spearman']):>6.4f}  "
            f"{st['kendall']:>+8.4f}  "
            f"{st['mono_strict']:>7.1%}"
        )
    print("=" * 80)
    logger.info(f"\nTemps total : {time.time()-t_total:.1f}s")
    logger.info(f"Résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
