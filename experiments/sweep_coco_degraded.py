"""
Sweep backbone / layer / gamma sur le dataset COCO-dégradé.

Pour chaque combinaison (backbone, layer, gamma) :
  - Ancrage = les 200 images COCO originales (propres) du dataset COCO-dégradé
  - On calcule le MMD² point (1 image dégradée vs distribution des 200 originales)
  - On mesure 3 métriques de monotonie vs le niveau de dégradation (1→5) :
      * Spearman ρ           : corrélation globale MMD vs niveau (principale)
      * Strict mono rate     : % de séquences (ref×type) parfaitement ordonnées
      * Kendall τ moyen      : monotonie partielle par séquence

Cache features sur disque → le sweep gamma est O(1) après extraction.

Usage :
    .venv/bin/python experiments/sweep_coco_degraded.py
    .venv/bin/python experiments/sweep_coco_degraded.py --n-eval-refs 200
    .venv/bin/python experiments/sweep_coco_degraded.py --backbones sd_vae,vgg19 --no-cache
"""

import argparse
import csv
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.config import BACKBONE_CONFIGS
from gram_mmd.features import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_DIR          = BASE_DIR / "dataset" / "coco_5000"
COCO_DEG_DIR      = BASE_DIR / "dataset" / "coco_degraded"
OUT_DIR           = BASE_DIR / "results" / "sweep_coco_degraded"
CACHE_DIR         = OUT_DIR / "cache"

N_EVAL_REFS       = 200  # toutes les refs COCO-dégradées disponibles
# Ancrage = les 200 images originales (propres) du dataset COCO-dégradé

# 5 types de dégradations représentatifs (sur les 25 KADID disponibles)
# 200 refs × 5 types × 5 niveaux = 5 000 images d'évaluation
EVAL_DIST_TYPES   = [1, 10, 11, 21, 25]
# 1=gaussian_blur, 10=jpeg, 11=white_noise, 21=pixelate, 25=contrast_change

# ---------------------------------------------------------------------------
# Gammas à tester (10 valeurs)
# None = médiane heuristique (adaptatif)
# ---------------------------------------------------------------------------
GAMMAS = [
    None,       # médiane heuristique
    2.8e-5,     # calibré features standardisées SD-VAE
    1e-7, 5e-7,
    1e-6, 5e-6,
    1e-5, 5e-5,
    1e-4, 1e-3,
]

# ---------------------------------------------------------------------------
# Backbones et leurs batch sizes
# ---------------------------------------------------------------------------
ALL_BACKBONES = {
    "sd_vae":        {"batch_size": 16, "input_size": 256},
    "dc_ae":         {"batch_size": 8,  "input_size": 512},
    "dinov2_vitb14": {"batch_size": 8,  "input_size": 518},
    "vgg19":         {"batch_size": 32, "input_size": 224},
    "lpips_vgg":     {"batch_size": 32, "input_size": 224},
}


# ===========================================================================
# Helpers
# ===========================================================================

def gamma_label(g) -> str:
    return "med" if g is None else f"{g:.1e}"


def load_ref_anchor() -> list:
    """Charge les 200 images originales (propres) du dataset COCO-dégradé comme ancrage."""
    ref_dir = COCO_DEG_DIR / "references"
    paths = sorted(ref_dir.glob("*.png")) + sorted(ref_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"Aucune image dans {ref_dir}")
    return [str(p) for p in paths]


def load_eval_metadata(n_refs: int,
                        dist_types: list = None) -> tuple:
    """
    Charge les métadonnées du dataset COCO-dégradé.

    n_refs     : nombre de références à utiliser (parmi les 200 disponibles)
    dist_types : liste des types de dégradation à inclure (None = tous les 25)

    Retourne :
        paths  : chemins vers les images dégradées
        levels : niveaux de dégradation (1-5)
        seq_ids: identifiant de séquence (ref_idx × n_types + type_rank)
    """
    meta_path = COCO_DEG_DIR / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv introuvable : {meta_path}")

    dist_filter = set(dist_types) if dist_types is not None else None

    rows = []
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Références disponibles dans l'ordre d'apparition
    all_refs = []
    seen = set()
    for r in rows:
        if r["ref_img"] not in seen:
            all_refs.append(r["ref_img"])
            seen.add(r["ref_img"])

    if n_refs > len(all_refs):
        raise ValueError(f"Demande {n_refs} refs mais seulement {len(all_refs)} disponibles")

    selected_refs = set(all_refs[:n_refs])
    ref_to_idx    = {r: i for i, r in enumerate(all_refs[:n_refs])}

    # Rangs des types filtrés (pour seq_id compact)
    if dist_filter:
        sorted_types = sorted(dist_filter)
        type_to_rank = {t: i for i, t in enumerate(sorted_types)}
        n_types = len(sorted_types)
    else:
        n_types = 25

    img_dir = COCO_DEG_DIR / "images"
    paths, levels, seq_ids = [], [], []

    for r in rows:
        if r["ref_img"] not in selected_refs:
            continue
        dist_type = int(r["dist_type"])
        if dist_filter and dist_type not in dist_filter:
            continue

        ref_idx = ref_to_idx[r["ref_img"]]
        level   = int(r["level"])
        rank    = type_to_rank[dist_type] if dist_filter else (dist_type - 1)
        seq_id  = ref_idx * n_types + rank

        paths.append(str(img_dir / r["dist_img"]))
        levels.append(level)
        seq_ids.append(seq_id)

    n_types_used = len(dist_filter) if dist_filter else 25
    logger.info(f"Images d'évaluation : {len(paths)} "
                f"({n_refs} refs × {n_types_used} types × 5 niveaux = "
                f"{n_refs * n_types_used * 5} attendus)")
    return paths, np.array(levels, dtype=np.int32), np.array(seq_ids, dtype=np.int32)


# ===========================================================================
# Métriques de monotonie
# ===========================================================================

def compute_metrics(scores: np.ndarray, levels: np.ndarray,
                    seq_ids: np.ndarray) -> dict:
    """
    scores  : (N,) MMD² score par image
    levels  : (N,) niveau de dégradation 1-5
    seq_ids : (N,) identifiant de séquence (ref × dist_type)

    Retourne :
        spearman_rho     : corrélation globale (principale)
        strict_mono_rate : % séquences parfaitement ordonnées
        kendall_tau_mean : Kendall τ moyen par séquence
    """
    # --- Spearman ρ global ---
    rho, _ = spearmanr(scores, levels)

    # --- Métriques par séquence ---
    unique_seqs = np.unique(seq_ids)
    strict_count = 0
    tau_values   = []

    for sid in unique_seqs:
        mask = seq_ids == sid
        if mask.sum() < 2:
            continue
        s = scores[mask]
        l = levels[mask]
        # Trier par niveau croissant
        order = np.argsort(l)
        s_sorted = s[order]

        # Strict monotonie : strictement croissant
        if np.all(np.diff(s_sorted) > 0):
            strict_count += 1

        # Kendall τ entre scores et niveaux
        tau, _ = kendalltau(l, s)
        if np.isfinite(tau):
            tau_values.append(tau)

    strict_rate = strict_count / len(unique_seqs) if unique_seqs.size > 0 else 0.0
    tau_mean    = float(np.mean(tau_values)) if tau_values else 0.0

    return {
        "spearman_rho":     float(rho),
        "strict_mono_rate": float(strict_rate),
        "kendall_tau_mean": tau_mean,
        "n_images":         int(len(scores)),
        "n_sequences":      int(len(unique_seqs)),
    }


# ===========================================================================
# Cache des matrices de distances (D_rr et D_rq)
# Beaucoup plus petit que les features : D_rq (200×5000) = 4 Mo vs 2.6 Go
# ===========================================================================

CHUNK_IMAGES = 500  # images par chunk GPU pour éviter OOM RAM


def get_dist_cache_path(backbone: str, layer: int, kind: str,
                        n_refs: int = None, dist_types: list = None,
                        use_std: bool = False) -> Path:
    """kind = 'rr' ou 'rq'"""
    suffix = "_std" if use_std else ""
    if kind == "rr":
        return CACHE_DIR / f"{backbone}_l{layer:02d}_Drr_ref200{suffix}.npy"
    else:
        types_str = "-".join(str(t) for t in sorted(dist_types or []))
        return CACHE_DIR / f"{backbone}_l{layer:02d}_Drq_r{n_refs}_t{types_str}{suffix}.npy"


def get_anchor_feat_cache_path(backbone: str, layer: int) -> Path:
    return CACHE_DIR / f"{backbone}_l{layer:02d}_anchor_ref200.npy"


def compute_and_cache_distances(extractor: FeatureExtractor,
                                 anchor_paths: list,
                                 query_paths: list,
                                 layer_idx: int,
                                 batch_size: int,
                                 device: str,
                                 use_cache: bool,
                                 n_refs: int,
                                 dist_types: list,
                                 use_std: bool = False) -> tuple:
    """
    Calcule D_rr (n_r×n_r) et D_rq (n_r×n_q) sur GPU, query par chunks.

    - Jamais plus de CHUNK_IMAGES features query en RAM simultanément
    - D_rq = 4 Mo (200×5000) vs 2.6 Go pour les features brutes
    - Cache D_rr et D_rq sur disque pour les runs suivants

    Retourne (D_rr, D_rq, feat_dim).
    """
    drr_path = get_dist_cache_path(extractor.backbone_name, layer_idx, "rr",
                                    use_std=use_std)
    drq_path = get_dist_cache_path(extractor.backbone_name, layer_idx, "rq",
                                    n_refs, dist_types, use_std=use_std)

    # --- Chargement depuis cache distances ---
    if use_cache and drr_path.exists() and drq_path.exists():
        D_rr = np.load(drr_path)
        D_rq = np.load(drq_path)
        logger.info(f"    D cache chargé : D_rr{D_rr.shape}  D_rq{D_rq.shape}")
        return D_rr, D_rq, 0   # feat_dim inconnu mais non utilisé dans les métriques

    # --- Extraction anchor ---
    # Avec standardisation : toujours ré-extraire pour fitter le StandardScaler.
    # Sans standardisation : utiliser le cache feature si disponible.
    anchor_cache = get_anchor_feat_cache_path(extractor.backbone_name, layer_idx)
    if not use_std and use_cache and anchor_cache.exists():
        r_np = np.load(anchor_cache).astype(np.float32)
        logger.info(f"    Anchor depuis cache feat : {r_np.shape}")
    else:
        logger.info(f"    Extraction anchor ({len(anchor_paths)} images)...")
        r_np = extractor.extract(anchor_paths,
                                  fit_transform=use_std,   # fit StandardScaler si use_std
                                  batch_size=batch_size).astype(np.float32)
        if use_cache and not use_std:
            np.save(anchor_cache, r_np)

    feat_dim = r_np.shape[1]

    # --- D_rr sur GPU ---
    r = torch.tensor(r_np, dtype=torch.float32, device=device)
    r_sq = (r * r).sum(dim=1)
    D_rr = torch.clamp(r_sq[:, None] + r_sq[None, :] - 2.0 * (r @ r.T),
                        min=0.0).cpu().numpy()

    # --- D_rq sur GPU, query par chunks ---
    n_q = len(query_paths)
    n_r = len(anchor_paths)
    D_rq = np.zeros((n_r, n_q), dtype=np.float32)

    n_chunks = (n_q + CHUNK_IMAGES - 1) // CHUNK_IMAGES
    logger.info(f"    Calcul D_rq par {n_chunks} chunks de {CHUNK_IMAGES} images...")

    for ci, start in enumerate(range(0, n_q, CHUNK_IMAGES)):
        chunk_paths = query_paths[start:start + CHUNK_IMAGES]
        chunk_np = extractor.extract(chunk_paths, fit_transform=False,
                                      batch_size=batch_size).astype(np.float32)
        q = torch.tensor(chunk_np, dtype=torch.float32, device=device)
        q_sq = (q * q).sum(dim=1)
        D_chunk = torch.clamp(r_sq[:, None] + q_sq[None, :] - 2.0 * (r @ q.T),
                               min=0.0).cpu().numpy()
        D_rq[:, start:start + len(chunk_paths)] = D_chunk
        del q, q_sq, D_chunk, chunk_np
        torch.cuda.empty_cache()
        gc.collect()
        if (ci + 1) % 2 == 0 or ci == n_chunks - 1:
            logger.info(f"      chunk {ci+1}/{n_chunks}")

    del r, r_sq
    torch.cuda.empty_cache()

    if use_cache:
        np.save(drr_path, D_rr)
        np.save(drq_path, D_rq)
        logger.info(f"    D_rr ({D_rr.nbytes//1024} Ko) et D_rq ({D_rq.nbytes//1024} Ko) sauvegardés")

    return D_rr, D_rq, feat_dim


# ===========================================================================
# Calcul MMD² (mode point)
# ===========================================================================

def compute_mmd_scores(anchor_feat: np.ndarray,
                        query_feat: np.ndarray,
                        gamma: float) -> np.ndarray:
    """LEGACY — non utilisé dans le sweep principal (remplacé par D_rq direct)."""
    r = anchor_feat.astype(np.float32)
    q = query_feat.astype(np.float32)

    r_sq = np.einsum("ij,ij->i", r, r)   # (n_r,)
    q_sq = np.einsum("ij,ij->i", q, q)   # (n_q,)

    D_rr = np.maximum(r_sq[:, None] + r_sq[None, :] - 2.0 * (r @ r.T), 0.0)
    D_rq = np.maximum(r_sq[:, None] + q_sq[None, :] - 2.0 * (r @ q.T), 0.0)

    K_rr  = np.exp(-gamma * D_rr)
    term1 = float(K_rr.mean())
    del K_rr

    K_rq   = np.exp(-gamma * D_rq)           # (n_r, n_q)
    term3  = 2.0 * K_rq.sum(axis=0) / r.shape[0]  # (n_q,)
    scores = term1 + 1.0 - term3
    del K_rq, D_rr, D_rq

    return scores.astype(np.float64)


# ===========================================================================
# Main sweep
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval-refs", type=int, default=N_EVAL_REFS)
    p.add_argument("--backbones",   type=str, default=None,
                   help="Sous-ensemble ex: 'sd_vae,vgg19'  (défaut=tous)")
    p.add_argument("--no-cache",    action="store_true",
                   help="Désactiver le cache features sur disque")
    p.add_argument("--use-std",     action="store_true",
                   help="Standardiser les features Gram (StandardScaler fitté sur l'ancrage)")
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top-k",       type=int, default=20,
                   help="Afficher les top-K configs dans le résumé")
    return p.parse_args()


def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Sélection des backbones
    backbones_to_run = list(ALL_BACKBONES.keys())
    if args.backbones:
        backbones_to_run = [b.strip() for b in args.backbones.split(",")]

    use_cache = not args.no_cache

    logger.info("=" * 70)
    logger.info("SWEEP backbone / layer / gamma sur COCO-dégradé")
    logger.info(f"  Backbones   : {backbones_to_run}")
    logger.info(f"  Eval refs   : {args.n_eval_refs} refs × {len(EVAL_DIST_TYPES)} types × 5 niveaux "
                f"= {args.n_eval_refs * len(EVAL_DIST_TYPES) * 5} images")
    logger.info(f"  Dist types  : {EVAL_DIST_TYPES}")
    logger.info(f"  Ancrage     : 200 images originales COCO-dégradé (références propres)")
    logger.info(f"  Gammas      : {len(GAMMAS)} ({[gamma_label(g) for g in GAMMAS]})")
    logger.info(f"  Cache       : {'activé' if use_cache else 'désactivé'}")
    logger.info(f"  Device      : {args.device}")
    logger.info("=" * 70)

    # Charger les données d'évaluation
    eval_paths, levels, seq_ids = load_eval_metadata(args.n_eval_refs, EVAL_DIST_TYPES)
    anchor_paths = load_ref_anchor()
    logger.info(f"Ancrage chargé : {len(anchor_paths)} images originales (références propres)")

    all_results = []

    for backbone_name in backbones_to_run:
        if backbone_name not in BACKBONE_CONFIGS:
            logger.warning(f"Backbone inconnu : {backbone_name} — ignoré")
            continue

        bb_config  = ALL_BACKBONES.get(backbone_name, {"batch_size": 8})
        batch_size = bb_config["batch_size"]
        layer_names = BACKBONE_CONFIGS[backbone_name]["layer_names"]
        n_layers    = len(layer_names)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"BACKBONE : {backbone_name}  ({n_layers} layers)")
        logger.info(f"{'=' * 70}")

        for layer_idx in tqdm(sorted(layer_names.keys()),
                               desc=f"{backbone_name}", unit="layer"):
            logger.info(f"\n  Layer {layer_idx} — {layer_names[layer_idx]}")

            # --- Charger l'extracteur ---
            try:
                extractor = FeatureExtractor(
                    backbone=backbone_name,
                    layer=layer_idx,
                    device=args.device,
                    transform_config={
                        "name": "gram_spatial",
                        "use_gram": True,
                        "use_pca": False,
                        "gram_patches": False,
                        "gram_averaging": "spatial",
                    },
                )
            except Exception as e:
                logger.error(f"    Impossible de charger {backbone_name} l{layer_idx}: {e}")
                continue

            # --- Calcul D_rr et D_rq sur GPU (query par chunks, pas d'OOM) ---
            try:
                D_rr, D_rq, feat_dim = compute_and_cache_distances(
                    extractor, anchor_paths, eval_paths, layer_idx,
                    batch_size, args.device, use_cache,
                    args.n_eval_refs, EVAL_DIST_TYPES,
                    use_std=args.use_std,
                )
            except Exception as e:
                logger.error(f"    Distance échouée : {e}")
                extractor.cleanup()
                continue

            extractor.cleanup()
            torch.cuda.empty_cache()

            # Médiane heuristique sur D_rr
            triu_vals   = D_rr[np.triu_indices(D_rr.shape[0], k=1)]
            median_dist = float(np.median(triu_vals))
            gamma_med   = 1.0 / (2.0 * median_dist + 1e-8)
            logger.info(f"    median_D_rr={median_dist:.3e}  gamma_med={gamma_med:.3e}  "
                        f"feat_dim={feat_dim}")

            # --- Sweep gammas ---
            for gamma_raw in GAMMAS:
                actual_gamma = gamma_med if gamma_raw is None else gamma_raw
                glabel       = gamma_label(gamma_raw)

                try:
                    n_r = D_rr.shape[0]
                    K_rr   = np.exp(-actual_gamma * D_rr)
                    term1  = float(K_rr.mean())
                    del K_rr

                    K_rq   = np.exp(-actual_gamma * D_rq)
                    term3  = 2.0 * K_rq.sum(axis=0) / n_r
                    scores = (term1 + 1.0 - term3).astype(np.float64)
                    k_rq_mean = float(K_rq.mean())
                    del K_rq

                    metrics = compute_metrics(scores, levels, seq_ids)
                    metrics.update({
                        "backbone":    backbone_name,
                        "layer":       layer_idx,
                        "layer_name":  layer_names[layer_idx],
                        "gamma_raw":   gamma_raw if gamma_raw is not None else "med",
                        "gamma_label": glabel,
                        "gamma_value": actual_gamma,
                        "feat_dim":    feat_dim,
                        "median_dist": median_dist,
                        "k_rq_mean":   k_rq_mean,
                        "term1":       term1,
                    })
                    all_results.append(metrics)

                    logger.info(
                        f"    gamma={glabel:>8s}  "
                        f"ρ={metrics['spearman_rho']:+.4f}  "
                        f"mono={metrics['strict_mono_rate']:.3f}  "
                        f"τ={metrics['kendall_tau_mean']:+.4f}  "
                        f"K_rq={k_rq_mean:.4f}"
                    )
                except Exception as e:
                    logger.error(f"    gamma={glabel} ERREUR : {e}")

            del D_rr, D_rq
            gc.collect()

        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------------------------------------------------
    # Sauvegarder les résultats
    # -----------------------------------------------------------------------
    if not all_results:
        logger.error("Aucun résultat généré.")
        return

    # Trier par Spearman ρ décroissant
    all_results.sort(key=lambda x: -x["spearman_rho"])

    csv_path = OUT_DIR / f"sweep_results_{ts}.csv"
    fieldnames = [
        "backbone", "layer", "layer_name", "gamma_label", "gamma_value",
        "spearman_rho", "strict_mono_rate", "kendall_tau_mean",
        "feat_dim", "median_dist", "k_rq_mean", "term1",
        "n_images", "n_sequences", "gamma_raw",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    logger.info(f"\nRésultats sauvegardés : {csv_path}  ({len(all_results)} lignes)")

    # -----------------------------------------------------------------------
    # Afficher le top-K
    # -----------------------------------------------------------------------
    k = min(args.top_k, len(all_results))
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TOP {k} configurations (par Spearman ρ)")
    logger.info(f"{'=' * 70}")
    header = f"{'Backbone':<15} {'L':>3} {'Gamma':>10}  {'ρ':>7}  {'mono':>6}  {'τ':>7}  {'dim':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in all_results[:k]:
        logger.info(
            f"{r['backbone']:<15} {r['layer']:>3} {r['gamma_label']:>10}  "
            f"{r['spearman_rho']:>+7.4f}  {r['strict_mono_rate']:>6.3f}  "
            f"{r['kendall_tau_mean']:>+7.4f}  {r['feat_dim']:>8}"
        )

    # Top-1 par backbone
    logger.info(f"\n--- Top-1 par backbone ---")
    seen_bb = set()
    for r in all_results:
        if r["backbone"] not in seen_bb:
            logger.info(
                f"  {r['backbone']:<15} l{r['layer']:>2} gamma={r['gamma_label']:>10}  "
                f"ρ={r['spearman_rho']:+.4f}  mono={r['strict_mono_rate']:.3f}  "
                f"τ={r['kendall_tau_mean']:+.4f}"
            )
            seen_bb.add(r["backbone"])

    logger.info("\nTerminé.")


if __name__ == "__main__":
    main()
