"""
raise_top3_coco_distdist.py
============================
RAISE  —  distribution-vs-distribution MMD² (γ médiane heuristique)
avec 1 000 images COCO comme ancrage.

Configs testées :
  - DINOv2-ViT-B/14  layer 5  (blocks.4)
  - DC-AE            layer 12 (encoder.down_blocks.3.2)
  - SD-VAE           layer 13 (encoder.mid_block.attentions.0)
  - CMMD             CLIP ViT-L/14@336, γ fixe=0.005 + γ_med

Groupage RAISE :
  - Images filtrées (préfixe "f"), triées par MOS croissant
  - 24 groupes × 20 images (consécutifs)
  - group_mos[k] = MOS moyen des 20 images au rang k

Score MMD²_u dist-vs-dist :
  MMD²_u(G_k, COCO) = mean_K_qq_triu + mean_K_rr_triu − 2·mean_K_rq

Optimisation :
  - Une seule passe d'extraction par backbone
  - Distances pré-calculées et mises en cache
  - GPU kernel batch (tous gammas en parallèle)

Usage :
    .venv/bin/python experiments/raise_top3_coco_distdist.py
    .venv/bin/python experiments/raise_top3_coco_distdist.py --skip-cmmd
"""

import argparse
import csv
import gc
import glob as glob_module
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import kendalltau, spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor, ImageDataset
from torch.utils.data import DataLoader

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

OUT_DIR   = BASE_DIR / "results" / "raise_top3_distdist"
PLOTS_DIR = OUT_DIR / "plots"
CACHE_DIR = OUT_DIR / "cache"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

N_ANCHOR    = 1000
ANCHOR_SEED = 42
N_GROUPS    = 24
GROUP_SIZE  = 20

# (backbone, layer_idx, display_name, batch_size)
# batch_size adapté à l'input_size : dc_ae/sd_vae plus gros → bs réduit
BACKBONE_CONFIGS = [
    ("dinov2_vitb14", 5,  "DINOv2 L5",  4),
    ("dc_ae",         12, "DC-AE L12",  1),  # bs=1 : input 512px, activations lourdes
    ("sd_vae",        13, "SD-VAE M13", 8),
]

CMMD_GAMMA    = 0.005
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
BATCH_CLIP    = 32
GPU_CHUNK     = 250   # lignes d'ancrage traitées en une fois sur GPU

BACKBONE_COLORS = {
    "DINOv2 L5":  "#F57C00",
    "DC-AE L12":  "#388E3C",
    "SD-VAE M13": "#1976D2",
    "CMMD":       "#E91E63",
}


# ═══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════════════

def load_raise_items():
    """Retourne (paths, mos_array) triés par MOS croissant (images "f*" seulement)."""
    mos_dict = {}
    for cp in [RAISE_TRAIN_RATINGS, RAISE_TEST_RATINGS]:
        with open(cp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = float(row["MOS_Rating"])

    all_imgs = {}
    for folder in [RAISE_TRAIN_IMAGES, RAISE_TEST_IMAGES]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[Path(p).name] = p

    items = [
        (path, mos_dict[fname])
        for fname, path in all_imgs.items()
        if fname.startswith("f") and fname in mos_dict
    ]
    items.sort(key=lambda x: x[1])

    n_needed = N_GROUPS * GROUP_SIZE
    if len(items) < n_needed:
        raise ValueError(
            f"RAISE : seulement {len(items)} images disponibles, "
            f"besoin de {n_needed} ({N_GROUPS}×{GROUP_SIZE})"
        )

    logger.info(
        f"RAISE : {len(items)} images  MOS [{items[0][1]:.2f}…{items[-1][1]:.2f}]"
    )
    return [x[0] for x in items], np.array([x[1] for x in items])


def make_groups(all_paths, all_mos):
    """
    Groupe consécutif : groupe k = images [k*GROUP_SIZE : (k+1)*GROUP_SIZE]
    Retourne (grouped_paths, group_mos) où group_mos[k] = MOS moyen.
    """
    grouped_paths = [
        all_paths[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
        for g in range(N_GROUPS)
    ]
    group_mos = np.array([
        all_mos[g * GROUP_SIZE:(g + 1) * GROUP_SIZE].mean()
        for g in range(N_GROUPS)
    ])
    logger.info(
        f"Groupes : {N_GROUPS} × {GROUP_SIZE}  "
        f"MOS [{group_mos[0]:.2f}…{group_mos[-1]:.2f}]"
    )
    return grouped_paths, group_mos


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
        return np.load(fanc), np.load(fdst, mmap_mode="r")
    return None, None


def save_features_cache(backbone, layer, feat_anc):
    key = _cache_key(backbone, layer)
    np.save(CACHE_DIR / f"{key}_anc.npy", feat_anc)
    logger.info(f"  Cache feat_anc sauvegardé : {key}")


def load_distances_cache(backbone, layer):
    # D_qq et D_rq dépendent du groupage → clé inclut N_GROUPS×GROUP_SIZE
    key = _cache_key(backbone, layer)
    # Nouveau nommage (avec groupage)
    path = CACHE_DIR / f"{key}_dist_{N_GROUPS}x{GROUP_SIZE}.npz"
    # Rétrocompatibilité : ancien nommage sans groupage (run 24×20 initial)
    path_legacy = CACHE_DIR / f"{key}_distances.npz"
    if path.exists():
        logger.info(f"  Cache distances trouvé : {key} ({N_GROUPS}×{GROUP_SIZE})")
        d = np.load(path)
        return d["D_rr_triu"], d["D_qq_triu"], d["D_rq"]
    if path_legacy.exists():
        logger.info(f"  Cache distances legacy trouvé : {key}")
        d = np.load(path_legacy)
        return d["D_rr_triu"], d["D_qq_triu"], d["D_rq"]
    return None, None, None


def save_distances_cache(backbone, layer, D_rr_triu, D_qq_triu, D_rq):
    key  = _cache_key(backbone, layer)
    path = CACHE_DIR / f"{key}_dist_{N_GROUPS}x{GROUP_SIZE}.npz"
    np.savez_compressed(path, D_rr_triu=D_rr_triu, D_qq_triu=D_qq_triu, D_rq=D_rq)
    logger.info(f"  Cache distances sauvegardé : {key} ({N_GROUPS}×{GROUP_SIZE})")


def load_cmmd_cache():
    # Scores dépendants du groupage → clé inclut N_GROUPS×GROUP_SIZE
    path = CACHE_DIR / f"cmmd_scores_{N_GROUPS}x{GROUP_SIZE}.npz"
    if path.exists():
        logger.info(f"  Cache CMMD trouvé ({N_GROUPS}×{GROUP_SIZE})")
        d = np.load(path)
        return {k: d[k] for k in d.files}
    return None


def save_cmmd_cache(scores: dict):
    path = CACHE_DIR / f"cmmd_scores_{N_GROUPS}x{GROUP_SIZE}.npz"
    np.savez_compressed(path, **scores)
    logger.info(f"  Cache CMMD sauvegardé ({N_GROUPS}×{GROUP_SIZE})")


def load_cmmd_embeddings():
    """Embeddings CLIP bruts — indépendants du groupage."""
    # Cherche d'abord dans le cache courant, puis dans raise_top3_distdist de référence
    for cache_dir in [CACHE_DIR, BASE_DIR / "results" / "raise_top3_distdist" / "cache"]:
        path = cache_dir / "cmmd_embeddings.npz"
        if path.exists():
            logger.info(f"  Cache embeddings CLIP trouvé : {path}")
            d = np.load(path)
            return d["emb_anc"], d["emb_all"]
    return None, None


def save_cmmd_embeddings(emb_anc, emb_all):
    path = CACHE_DIR / "cmmd_embeddings.npz"
    np.savez_compressed(path, emb_anc=emb_anc, emb_all=emb_all)
    logger.info("  Cache embeddings CLIP sauvegardé")


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


def extract_all(ext, anchor_paths, all_query_paths, batch_size, dist_cache_path):
    """
    Extraction ancrage + RAISE (query).
    feat_dist écrit incrémentalement via mmap pour limiter la RAM.
    Retourne feat_anc (RAM) et feat_dist (mmap, lecture par tranche).
    """
    t0 = time.time()
    feat_anc = ext.extract(
        anchor_paths, fit_transform=True, batch_size=batch_size
    ).astype(np.float32)
    logger.info(f"  Ancrage  : {feat_anc.shape}  ({time.time()-t0:.1f}s)")

    D      = feat_anc.shape[1]
    n_qry  = len(all_query_paths)

    feat_dist = np.lib.format.open_memmap(
        str(dist_cache_path), mode="w+", dtype=np.float32, shape=(n_qry, D)
    )

    dataset = ImageDataset(all_query_paths, transform=ext.image_transform)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(ext.device == "cuda"),
        persistent_workers=True,
    )

    idx = 0
    t1  = time.time()
    for batch_imgs, _ in loader:
        raw    = ext.extract_batch(batch_imgs)
        scaled = ext.transform_features(raw, fit_pca=False).astype(np.float32)
        feat_dist[idx:idx + scaled.shape[0]] = scaled
        idx += scaled.shape[0]
        if idx % 100 < batch_size or idx >= n_qry:
            logger.info(f"    RAISE : {idx}/{n_qry} ({idx/n_qry:.0%})  {time.time()-t1:.0f}s")

    feat_dist.flush()
    logger.info(f"  RAISE dist : {feat_dist.shape}  ({time.time()-t1:.1f}s)")
    return feat_anc, feat_dist


# ═══════════════════════════════════════════════════════════════════════
#  GAMMA MÉDIANE
# ═══════════════════════════════════════════════════════════════════════

def compute_gamma_med(feat_anc: np.ndarray) -> float:
    """γ_med = 1 / (2 · median(||xi - xj||²))  sur les paires de l'ancrage."""
    anc_sq = np.einsum("ij,ij->i", feat_anc, feat_anc)
    D = np.maximum(
        anc_sq[:, None] + anc_sq[None, :] - 2.0 * (feat_anc @ feat_anc.T), 0.0
    )
    triu_vals = D[np.triu_indices(len(feat_anc), k=1)]
    triu_vals = triu_vals[triu_vals > 0]
    g_med = float(1.0 / (2.0 * np.median(triu_vals) + 1e-8))
    logger.info(f"  median(d²) = {np.median(triu_vals):.4e}  →  γ_med = {g_med:.4e}")
    return g_med


# ═══════════════════════════════════════════════════════════════════════
#  DISTANCES PRÉ-CALCULÉES
# ═══════════════════════════════════════════════════════════════════════

def sq_dist_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum(
        (A * A).sum(1, keepdims=True) + (B * B).sum(1, keepdims=True).T
        - 2.0 * (A @ B.T), 0.0
    )


def precompute_distances(feat_anc, feat_dist_grouped, device):
    """
    feat_anc           : (N_ANCHOR, D)
    feat_dist_grouped  : (N_GROUPS, GROUP_SIZE, D)

    Retourne :
      D_rr_triu  : (N_ANCHOR*(N_ANCHOR-1)//2,)
      D_qq_triu  : (N_GROUPS, GROUP_SIZE*(GROUP_SIZE-1)//2)
      D_rq       : (N_GROUPS, N_ANCHOR, GROUP_SIZE)

    D_rr calculé par blocs (GPU_CHUNK × GPU_CHUNK) pour éviter l'OOM
    sur les Gram de grande dimension (DC-AE : D=131K, DINOv2 : D=295K).
    """
    n_anc    = feat_anc.shape[0]
    n_q      = feat_dist_grouped.shape[1]
    n_pairs  = n_q * (n_q - 1) // 2
    triu_qq  = np.triu_indices(n_q,   k=1)
    triu_anc = np.triu_indices(n_anc, k=1)

    # Évite la copie si feat_anc est déjà C-contigu float32 (ex: chargé depuis cache)
    if feat_anc.flags["C_CONTIGUOUS"] and feat_anc.dtype == np.float32:
        anc_np = feat_anc
    else:
        anc_np = np.ascontiguousarray(feat_anc, dtype=np.float32)

    t0 = time.time()

    # ── D_rr (intra-ancrage) — calcul par blocs ──────────────────────
    # Chaque bloc GPU = GPU_CHUNK × GPU_CHUNK lignes → VRAM max ≈ 2 × GPU_CHUNK × D × 4B
    # ex. DC-AE : 2 × 250 × 131K × 4 ≈ 262 MB (vs 525 MB en un seul passage)
    D_rr_full = np.empty((n_anc, n_anc), dtype=np.float32)

    for i in range(0, n_anc, GPU_CHUNK):
        ci = anc_np[i:i + GPU_CHUNK]
        if device == "cuda":
            ci_t = torch.from_numpy(ci).to(device)
            for j in range(0, n_anc, GPU_CHUNK):
                cj_t = torch.from_numpy(anc_np[j:j + GPU_CHUNK]).to(device)
                block = torch.cdist(ci_t, cj_t, p=2).pow(2).cpu().numpy()
                ni, nj = len(ci), len(anc_np[j:j + GPU_CHUNK])
                D_rr_full[i:i + ni, j:j + nj] = block
                del cj_t
                torch.cuda.empty_cache()
            del ci_t
        else:
            for j in range(0, n_anc, GPU_CHUNK):
                cj = anc_np[j:j + GPU_CHUNK]
                D_rr_full[i:i + len(ci), j:j + len(cj)] = sq_dist_cpu(ci, cj)

    D_rr_triu = D_rr_full[triu_anc[0], triu_anc[1]].astype(np.float32)
    del D_rr_full
    gc.collect()
    logger.info(f"  D_rr done ({time.time()-t0:.1f}s)")

    D_qq_triu = np.empty((N_GROUPS, n_pairs),       dtype=np.float32)
    D_rq      = np.empty((N_GROUPS, n_anc, n_q),    dtype=np.float32)

    for k in range(N_GROUPS):
        fq_np = np.ascontiguousarray(feat_dist_grouped[k], dtype=np.float32)

        if device == "cuda":
            fq_t  = torch.from_numpy(fq_np).to(device)

            D_qq  = torch.cdist(fq_t, fq_t, p=2).pow(2)
            D_qq_triu[k] = D_qq[triu_qq[0], triu_qq[1]].cpu().numpy()
            del D_qq

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
            D_rq[k]      = sq_dist_cpu(anc_np, fq_np)

        if (k + 1) % 8 == 0 or k + 1 == N_GROUPS:
            logger.info(f"    distances : {k+1}/{N_GROUPS}  ({time.time()-t0:.1f}s)")

    logger.info(
        f"  D_rr_triu {D_rr_triu.shape}  "
        f"D_qq_triu {D_qq_triu.shape}  "
        f"D_rq {D_rq.shape}"
    )
    return D_rr_triu, D_qq_triu, D_rq


# ═══════════════════════════════════════════════════════════════════════
#  SCORING MMD²
# ═══════════════════════════════════════════════════════════════════════

def compute_mmd2_groups(D_rr_triu, D_qq_triu, D_rq, gamma: float, device):
    """
    Calcule MMD²_u(groupe_k, COCO) pour un seul gamma.
    Retourne np.ndarray (N_GROUPS,).
    """
    D_rr_t = torch.from_numpy(D_rr_triu).to(device)   # (N_pairs_rr,)
    D_qq_t = torch.from_numpy(D_qq_triu).to(device)   # (N_GROUPS, N_pairs_qq)
    D_rq_t = torch.from_numpy(D_rq).to(device)        # (N_GROUPS, N_ANCHOR, GROUP_SIZE)

    with torch.no_grad():
        gf = float(gamma)
        K_rr_mean = torch.exp(-gf * D_rr_t).mean()
        K_qq_mean = torch.exp(-gf * D_qq_t).mean(dim=1)
        K_rq_mean = torch.exp(-gf * D_rq_t).mean(dim=(1, 2))
        mmd2 = (K_qq_mean + K_rr_mean - 2.0 * K_rq_mean).cpu().numpy().astype(np.float64)

    del D_rr_t, D_qq_t, D_rq_t
    torch.cuda.empty_cache()
    logger.info(
        f"  MMD² (γ={gamma:.3e}) : "
        f"min={mmd2.min():.4e}  max={mmd2.max():.4e}  mean={mmd2.mean():.4e}"
    )
    return mmd2


# ═══════════════════════════════════════════════════════════════════════
#  CMMD
# ═══════════════════════════════════════════════════════════════════════

def run_cmmd(anchor_paths, grouped_paths, device):
    """
    CLIP ViT-L/14@336.
    γ_med (heuristique sur embeddings CLIP) + γ fixe=0.005.
    Retourne dict {"median": (N_GROUPS,), "fixed": (N_GROUPS,)}.
    Les embeddings bruts sont mis en cache indépendamment du groupage.
    """
    emb_anc, emb_all = load_cmmd_embeddings()

    if emb_anc is None:
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        logger.info(f"  CMMD : chargement {CLIP_MODEL_ID}...")
        model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

        def embed_paths(paths_list):
            embs = []
            for i in range(0, len(paths_list), BATCH_CLIP):
                batch = [Image.open(p).convert("RGB") for p in paths_list[i:i + BATCH_CLIP]]
                pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
                with torch.no_grad():
                    embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
                if (i // BATCH_CLIP + 1) % 20 == 0:
                    logger.info(
                        f"    CLIP : {min(i + BATCH_CLIP, len(paths_list))}/{len(paths_list)}"
                    )
            return np.concatenate(embs, axis=0).astype(np.float32)

        logger.info(f"  CMMD : embedding ancrage ({N_ANCHOR} imgs)...")
        emb_anc = embed_paths(anchor_paths)

        all_qry = [p for grp in grouped_paths for p in grp]
        logger.info(f"  CMMD : embedding RAISE ({len(all_qry)} imgs)...")
        t0 = time.time()
        emb_all = embed_paths(all_qry)
        logger.info(f"  CMMD : embedding terminé en {time.time()-t0:.1f}s")

        del model, proc
        torch.cuda.empty_cache()
        gc.collect()
        save_cmmd_embeddings(emb_anc, emb_all)
    else:
        # Vérifier que emb_all correspond au bon nombre d'images
        n_expected = N_GROUPS * GROUP_SIZE
        if emb_all.shape[0] != n_expected:
            raise ValueError(
                f"Cache embeddings : {emb_all.shape[0]} images, "
                f"attendu {n_expected} ({N_GROUPS}×{GROUP_SIZE}). "
                f"Supprimer le cache cmmd_embeddings.npz et relancer."
            )

    emb_grps = emb_all.reshape(N_GROUPS, GROUP_SIZE, -1)
    del emb_all

    # γ_med sur embeddings CLIP
    g_med = compute_gamma_med(emb_anc)

    D_rr_triu, D_qq_triu, D_rq = precompute_distances(emb_anc, emb_grps, device)
    del emb_anc, emb_grps
    gc.collect()

    scores_med   = compute_mmd2_groups(D_rr_triu, D_qq_triu, D_rq, g_med,      device)
    scores_fixed = compute_mmd2_groups(D_rr_triu, D_qq_triu, D_rq, CMMD_GAMMA, device)

    del D_rr_triu, D_qq_triu, D_rq
    torch.cuda.empty_cache()

    return {"median": scores_med, "fixed": scores_fixed, "gamma_med": np.array([g_med])}


# ═══════════════════════════════════════════════════════════════════════
#  STATISTIQUES
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(mmd2_scores: np.ndarray, group_mos: np.ndarray) -> dict:
    """
    mmd2_scores : (N_GROUPS,) — un score par groupe
    group_mos   : (N_GROUPS,) — MOS moyen par groupe (croissant : pire → meilleur)

    ρ < 0 attendu : plus dégradé → MOS bas → MMD² élevé.
    mono_strict : fraction de paires consécutives avec MMD²[k] > MMD²[k+1].
    """
    sp, sp_p = spearmanr(mmd2_scores, group_mos)
    kt, kt_p = kendalltau(mmd2_scores, group_mos)
    n_mono   = int(np.sum(mmd2_scores[:-1] > mmd2_scores[1:]))
    mono     = n_mono / (N_GROUPS - 1)
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

def plot_scatter(all_results, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f9f9f9")

    for entry in all_results:
        bname  = entry.get("backbone_name", "CMMD")
        g_tag  = entry["gamma_tag"]
        sp     = -entry["stats"]["spearman"]
        kt     = -entry["stats"]["kendall"]
        color  = BACKBONE_COLORS.get(bname, "#9C27B0")
        marker = "D" if g_tag == "fixed" else "o"
        ax.scatter(kt, sp, color=color, marker=marker, s=110,
                   edgecolors="black", linewidths=0.7, zorder=5)
        ax.annotate(
            entry["label"], (kt, sp),
            textcoords="offset points", xytext=(8, 5),
            fontsize=6.5, zorder=6,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3),
        )

    color_handles = [
        Line2D([0], [0], marker="o", color=c, linestyle="None", markersize=8,
               markeredgecolor="black", markeredgewidth=0.6, label=n)
        for n, c in BACKBONE_COLORS.items()
        if any(e.get("backbone_name") == n for e in all_results)
    ]
    ax.legend(handles=color_handles, fontsize=8, loc="upper left",
              title="Backbone", framealpha=0.9)
    ax.set_xlabel("-Kendall τ", fontsize=12)
    ax.set_ylabel("-Spearman ρ", fontsize=12)
    ax.set_title(
        f"RAISE  ·  1 000 COCO anchor  ·  {N_GROUPS}×{GROUP_SIZE}\n"
        "-Spearman ρ vs -Kendall τ",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / "01_scatter_spearman_vs_kendall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


def plot_barplot(all_results, out_dir: Path):
    sorted_res = sorted(all_results, key=lambda x: -abs(x["stats"]["spearman"]))
    labels  = [e["label"]                       for e in sorted_res]
    sp_vals = [-e["stats"]["spearman"]           for e in sorted_res]
    kt_vals = [-e["stats"]["kendall"]            for e in sorted_res]
    mono_v  = [e["stats"]["mono_strict"]         for e in sorted_res]
    colors  = [BACKBONE_COLORS.get(e.get("backbone_name", "CMMD"), "#9C27B0")
               for e in sorted_res]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    x = np.arange(len(labels))

    for ax, vals, ylabel in zip(
        axes,
        [sp_vals, kt_vals, mono_v],
        ["-Spearman ρ", "-Kendall τ", "Mono stricte"],
    ):
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2 if vals else 1)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_facecolor("#f9f9f9")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_title(
        f"RAISE  ·  1 000 COCO anchor  ·  {N_GROUPS}×{GROUP_SIZE}  ·  γ médiane\n"
        "Tri par |Spearman ρ| décroissant",
        fontsize=11, fontweight="bold",
    )
    handles = [
        Line2D([0], [0], color=c, linewidth=8, label=n)
        for n, c in BACKBONE_COLORS.items()
        if any(e.get("backbone_name") == n for e in all_results)
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    out = out_dir / "02_barplot_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


def plot_mmd2_profile(all_results, group_mos, out_dir: Path):
    """Profil MMD² en fonction du rang MOS pour chaque config."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#f9f9f9")
    ranks = np.arange(N_GROUPS)

    for entry in all_results:
        bname = entry.get("backbone_name", "CMMD")
        color = BACKBONE_COLORS.get(bname, "#9C27B0")
        ls    = "--" if entry["gamma_tag"] == "fixed" else "-"
        scores = entry["mmd2_scores"]
        # normalise 0→1 pour superposer
        mn, mx = scores.min(), scores.max()
        if mx > mn:
            scores_norm = (scores - mn) / (mx - mn)
        else:
            scores_norm = scores - mn
        ax.plot(ranks, scores_norm, color=color, linestyle=ls,
                linewidth=1.5, label=entry["label"], alpha=0.85)

    ax.set_xlabel("Rang MOS (0 = pire qualité)", fontsize=11)
    ax.set_ylabel("MMD² normalisé [0-1]", fontsize=11)
    ax.set_title(
        f"Profil MMD² dist-vs-dist · RAISE {N_GROUPS}×{GROUP_SIZE} · COCO anchor",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / "03_mmd2_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
#  SAUVEGARDE CSV
# ═══════════════════════════════════════════════════════════════════════

def save_csv(all_results, ts):
    path = OUT_DIR / f"summary_{ts}.csv"
    fields = [
        "label", "backbone", "layer", "gamma_tag", "gamma_val",
        "spearman", "spearman_p", "kendall", "kendall_p", "mono_strict",
    ]
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
                "gamma_val":   f"{e['gamma_val']:.4e}" if e.get("gamma_val") is not None else "",
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
    parser.add_argument("--skip-cmmd", action="store_true",
                        help="Passer le calcul CMMD (CLIP)")
    parser.add_argument("--n-groups",  type=int, default=24,
                        help="Nombre de groupes (défaut: 24)")
    parser.add_argument("--group-size", type=int, default=20,
                        help="Taille de chaque groupe (défaut: 20)")
    args = parser.parse_args()

    # Surcharge des constantes globales si différentes des valeurs par défaut
    global N_GROUPS, GROUP_SIZE, OUT_DIR, PLOTS_DIR, CACHE_DIR  # noqa: PLW0603
    N_GROUPS   = args.n_groups
    GROUP_SIZE = args.group_size
    run_tag    = f"{N_GROUPS}x{GROUP_SIZE}"

    if run_tag != "24x20":   # répertoire dédié si config différente
        OUT_DIR   = BASE_DIR / "results" / f"raise_top3_distdist_{run_tag}"
        PLOTS_DIR = OUT_DIR / "plots"
        CACHE_DIR = OUT_DIR / "cache"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(exist_ok=True)
        CACHE_DIR.mkdir(exist_ok=True)
        # Les features anc/dist sont les mêmes (même ancre, même images RAISE)
        # → lien symbolique ou copie du cache existant si disponible
        src_cache = BASE_DIR / "results" / "raise_top3_distdist" / "cache"
        for backbone, layer, _, _ in BACKBONE_CONFIGS:
            key = f"{backbone}_L{layer:02d}"
            for suffix in ["_anc.npy", "_dist.npy"]:
                src = src_cache / f"{key}{suffix}"
                dst = CACHE_DIR / f"{key}{suffix}"
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info(f"  Cache features copié : {key}{suffix}")
        # CMMD scores aussi indépendants du groupage (scores par image, regroupés après)
        src_cmmd = src_cache / "cmmd_scores.npz"
        dst_cmmd = CACHE_DIR / "cmmd_scores_raw.npz"
        if src_cmmd.exists() and not dst_cmmd.exists():
            import shutil
            shutil.copy2(src_cmmd, dst_cmmd)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_total = time.time()
    logger.info("=" * 80)
    logger.info(
        f"RAISE dist-vs-dist · {N_GROUPS}×{GROUP_SIZE} · 1 000 COCO anchor · γ_med"
    )
    logger.info(f"  Device : {args.device}")

    # ── Données ──────────────────────────────────────────────────────
    all_paths, all_mos = load_raise_items()
    grouped_paths, group_mos = make_groups(all_paths, all_mos)
    anchor_paths = load_coco_anchor()

    # Liste plate dans l'ordre groupe-majeur pour l'extraction
    all_qry_flat = [p for grp in grouped_paths for p in grp]   # 480 chemins

    all_results = []

    # ── Backbones ─────────────────────────────────────────────────────
    for backbone, layer, display_name, batch_sz in BACKBONE_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Backbone : {display_name}  (layer={layer})")

        # ── Features : cache ou extraction ──────────────────────────
        feat_anc, feat_all = load_features_cache(backbone, layer)
        if feat_anc is None:
            ext = make_extractor(backbone, layer, args.device)
            dist_path = CACHE_DIR / f"{_cache_key(backbone, layer)}_dist.npy"
            feat_anc, feat_all = extract_all(
                ext, anchor_paths, all_qry_flat, batch_sz, dist_path
            )
            ext.cleanup()
            del ext
            torch.cuda.empty_cache()
            gc.collect()
            save_features_cache(backbone, layer, feat_anc)

        # ── Gamma médiane ────────────────────────────────────────────
        g_med = compute_gamma_med(feat_anc)

        # ── Reshape RAISE : (N_GROUPS, GROUP_SIZE, D) ────────────────
        # Copie en RAM avant de fermer la mmap (évite les accès implicites
        # après flush, qui peuvent garder des pages VRAM/RAM fantômes)
        D        = feat_all.shape[1]
        feat_grps = np.array(feat_all, dtype=np.float32).reshape(N_GROUPS, GROUP_SIZE, D)
        del feat_all
        gc.collect()
        torch.cuda.empty_cache()

        # ── Distances : cache ou calcul ─────────────────────────────
        D_rr_triu, D_qq_triu, D_rq = load_distances_cache(backbone, layer)
        if D_rr_triu is None:
            logger.info("  Pré-calcul distances (GPU)...")
            D_rr_triu, D_qq_triu, D_rq = precompute_distances(
                feat_anc, feat_grps, args.device
            )
            save_distances_cache(backbone, layer, D_rr_triu, D_qq_triu, D_rq)
        del feat_anc, feat_grps
        gc.collect()

        # ── Scoring MMD² ─────────────────────────────────────────────
        logger.info("  Scoring MMD² (γ_med)...")
        mmd2 = compute_mmd2_groups(D_rr_triu, D_qq_triu, D_rq, g_med, args.device)
        del D_rr_triu, D_qq_triu, D_rq
        torch.cuda.empty_cache()
        gc.collect()

        stats = compute_stats(mmd2, group_mos)
        label = f"GMMD {display_name} γ_med"
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
            "gamma_tag":     "median",
            "gamma_val":     g_med,
            "stats":         stats,
            "mmd2_scores":   mmd2,
        })

    # ── CMMD ──────────────────────────────────────────────────────────
    if not args.skip_cmmd:
        logger.info(f"\n{'='*60}")
        logger.info("  CMMD (CLIP ViT-L/14@336)")
        cmmd_cache = load_cmmd_cache()
        if cmmd_cache is None:
            cmmd_cache = run_cmmd(anchor_paths, grouped_paths, args.device)
            save_cmmd_cache(cmmd_cache)

        cmmd_g_med = float(cmmd_cache["gamma_med"][0])

        for tag, scores in [
            ("median", cmmd_cache["median"]),
            ("fixed",  cmmd_cache["fixed"]),
        ]:
            stats = compute_stats(scores, group_mos)
            g_val = cmmd_g_med if tag == "median" else CMMD_GAMMA
            glabel = f"γ_med" if tag == "median" else f"γ={CMMD_GAMMA}"
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
                "gamma_val":     g_val,
                "stats":         stats,
                "mmd2_scores":   scores,
            })

    # ── Sauvegarde ────────────────────────────────────────────────────
    save_csv(all_results, ts)
    plot_scatter(all_results, PLOTS_DIR)
    plot_barplot(all_results, PLOTS_DIR)
    plot_mmd2_profile(all_results, group_mos, PLOTS_DIR)

    # ── Résumé console ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  {'Label':<38}  {'ρ':>8}  {'|ρ|':>6}  {'τ':>8}  {'Mono':>7}")
    print("-" * 78)
    for e in sorted(all_results, key=lambda x: abs(x["stats"]["spearman"]), reverse=True):
        st = e["stats"]
        print(
            f"  {e['label']:<38}  "
            f"{st['spearman']:>+8.4f}  "
            f"{abs(st['spearman']):>6.4f}  "
            f"{st['kendall']:>+8.4f}  "
            f"{st['mono_strict']:>7.1%}"
        )
    print("=" * 78)
    logger.info(f"\nTemps total : {time.time()-t_total:.1f}s")
    logger.info(f"Résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
