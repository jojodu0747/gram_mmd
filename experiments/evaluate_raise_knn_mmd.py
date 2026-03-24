"""
evaluate_raise_knn_mmd.py
=========================
kNN-MMD² en mode point pour le test de monotonicité RAISE.

Pour chaque image IA RAISE (480 images), on calcule la distance à
l'ancrage COCO (1000 images) en restreignant aux k plus proches voisins :

    kNN-MMD²(y, X) = mean(K_rr^k)  +  1  -  (2/k) * Σ_{i ∈ kNN(y,X)} k(x_i, y)

Différentes valeurs de k sont testées.
CMMD (CLIP global) sert de baseline de comparaison.

Usage :
    python experiments/evaluate_raise_knn_mmd.py
    python experiments/evaluate_raise_knn_mmd.py --skip-cmmd
    python experiments/evaluate_raise_knn_mmd.py --device cpu

Résultats -> results/raise_knn_mmd/
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor, ImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

RAISE_ROOT          = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_TRAIN_IMAGES  = RAISE_ROOT / "images" / "train_images"
RAISE_TEST_IMAGES   = RAISE_ROOT / "images" / "test_images"
RAISE_TRAIN_RATINGS = RAISE_ROOT / "ratings" / "train.csv"
RAISE_TEST_RATINGS  = RAISE_ROOT / "ratings" / "test.csv"
COCO_DIR            = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR             = BASE_DIR / "results" / "raise_knn_mmd"

# Ancrage COCO
N_ANCHOR    = 1000
ANCHOR_SEED = 42

# Valeurs de k à tester. k=1000 = ancrage complet = global MMD biaisé.
K_VALUES = [1, 5, 10, 25, 50, 100, 250, 500, 1000]

# Gamma RBF — même valeur que les autres expériences
GAMMA = 2.8e-5

# Backbones (backbone_name, layer_idx, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",        9,  "SD-VAE l9",   16),
    ("dc_ae",        10,  "DC-AE l10",    4),
    ("dinov2_vitb14", 5,  "DINOv2-B l5",  8),
]

# CMMD baseline
ENABLE_CMMD   = True
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005
BATCH_SIZE_CLIP = 32


# ============================================================
# Chargement des données
# ============================================================

def load_coco_anchor() -> list:
    imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if not imgs:
        raise FileNotFoundError(f"Aucune image trouvée dans {COCO_DIR}")
    if N_ANCHOR < len(imgs):
        rng = np.random.default_rng(ANCHOR_SEED)
        idx = sorted(rng.choice(len(imgs), size=N_ANCHOR, replace=False).tolist())
        imgs = [imgs[i] for i in idx]
    logger.info(f"Ancrage COCO : {len(imgs)} images (seed={ANCHOR_SEED})")
    return imgs


def load_raise_data() -> list:
    """Retourne liste de (path, mos_rating, mos_zscore) pour les 480 images IA,
    triée par MOS_Rating croissant."""
    mos_dict = {}
    for p in [RAISE_TRAIN_RATINGS, RAISE_TEST_RATINGS]:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = (
                    float(row["MOS_Rating"]),
                    float(row["MOS_ZScore"]),
                )

    all_imgs = {}
    for folder in [RAISE_TRAIN_IMAGES, RAISE_TEST_IMAGES]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[os.path.basename(p)] = p

    items = [
        (path, mos_dict[fname][0], mos_dict[fname][1])
        for fname, path in all_imgs.items()
        if fname.startswith("f") and fname in mos_dict
    ]
    items.sort(key=lambda x: x[1])
    logger.info(f"Images IA RAISE : {len(items)}  "
                f"MOS [{items[0][1]:.2f} … {items[-1][1]:.2f}]")
    return items


# ============================================================
# GPU helpers
# ============================================================

def _compute_D2_gpu(query: np.ndarray, ref: np.ndarray,
                    ref_sq: np.ndarray, device: str) -> np.ndarray:
    """Distance L2² (n_query, n_ref) calculée sur GPU."""
    q_t   = torch.from_numpy(query).to(device)
    r_t   = torch.from_numpy(ref).to(device)
    rsq_t = torch.from_numpy(ref_sq).to(device)
    with torch.no_grad():
        qsq_t = (q_t * q_t).sum(dim=1)
        D2_t  = torch.clamp(
            qsq_t[:, None] + rsq_t[None, :] - 2.0 * (q_t @ r_t.T),
            min=0.0,
        )
    return D2_t.cpu().numpy()


# ============================================================
# kNN-MMD scoring
# ============================================================

def compute_knn_mmd_scores(
    query_feats: np.ndarray,   # (n_query, D)
    ref_feats:   np.ndarray,   # (n_anchor, D)
    K_rr_full:   np.ndarray,   # (n_anchor, n_anchor)
    D2_qr:       np.ndarray,   # (n_query, n_anchor)  — préalculé sur GPU
    k: int,
) -> np.ndarray:
    """
    kNN-MMD² biaisé pour chaque point de requête, restreint aux k plus proches voisins.
    Retourne un tableau float64 (n_query,).
    """
    n_q    = query_feats.shape[0]
    n_anc  = ref_feats.shape[0]
    k_eff  = min(k, n_anc)
    K_qr   = np.exp(-GAMMA * D2_qr)   # (n_q, n_anc)
    scores = np.empty(n_q, dtype=np.float64)

    for i in range(n_q):
        nn_idx   = np.argpartition(D2_qr[i], k_eff - 1)[:k_eff]
        K_rr_k   = K_rr_full[np.ix_(nn_idx, nn_idx)]
        term1    = float(K_rr_k.mean())
        term3    = 2.0 / k_eff * float(K_qr[i, nn_idx].sum())
        scores[i] = term1 + 1.0 - term3

    return scores


# ============================================================
# Evaluation d'un backbone
# ============================================================

def evaluate_backbone(
    backbone_name: str,
    layer_idx: int,
    display_name: str,
    batch_size: int,
    anchor_paths: list,
    raise_items: list,
    device: str,
) -> dict:
    """
    Extrait les features Gram, calcule la kNN-MMD pour chaque k,
    et retourne {k: {"scores": np.ndarray, "rho": float, "neg_rho": float}}.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"BACKBONE : {display_name}  ({backbone_name} / layer {layer_idx})")
    logger.info("=" * 70)

    # --- Feature extractor ---
    extractor = FeatureExtractor(
        backbone=backbone_name,
        layer=layer_idx,
        transform_config={
            "name":           "gram_spatial",
            "use_gram":       True,
            "use_pca":        False,
            "gram_patches":   False,
            "gram_averaging": "spatial",
        },
    )

    # --- Ancrage ---
    logger.info(f"  Extraction ancrage ({len(anchor_paths)} images)…")
    ref_feats = extractor.extract(
        anchor_paths, fit_transform=True, batch_size=batch_size
    ).astype(np.float32)
    logger.info(f"  Features ancrage : {ref_feats.shape}")

    # --- Ancre : matrice de noyaux K_rr_full ---
    ref_sq  = (ref_feats * ref_feats).sum(axis=1)
    D2_rr   = np.maximum(
        ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref_feats @ ref_feats.T),
        0.0,
    )
    K_rr_full = np.exp(-GAMMA * D2_rr).astype(np.float32)
    logger.info(f"  K_rr_full : {K_rr_full.shape}  mean={K_rr_full.mean():.4f}")

    # --- Requêtes : 480 images RAISE ---
    query_paths = [item[0] for item in raise_items]
    mos_r       = np.array([item[1] for item in raise_items])

    logger.info(f"  Extraction requêtes ({len(query_paths)} images)…")
    query_feats = extractor.extract(
        query_paths, fit_transform=False, batch_size=batch_size
    ).astype(np.float32)
    logger.info(f"  Features requêtes : {query_feats.shape}")

    # --- Distance L2² sur GPU ---
    logger.info("  Calcul D2_qr sur GPU…")
    D2_qr = _compute_D2_gpu(query_feats, ref_feats, ref_sq, device)
    logger.info(f"  D2_qr : {D2_qr.shape}")

    # --- kNN-MMD pour chaque k ---
    results = {}
    for k in K_VALUES:
        scores = compute_knn_mmd_scores(
            query_feats, ref_feats, K_rr_full, D2_qr, k
        )
        rho, p = spearmanr(scores, mos_r)
        results[k] = {"scores": scores, "rho": float(rho), "neg_rho": float(-rho), "p": float(p)}
        logger.info(
            f"  k={k:>5}  rho={rho:+.4f}  -rho={-rho:+.4f}  p={p:.3e}"
        )

    # Nettoyage
    extractor.cleanup()
    del ref_feats, query_feats, D2_rr, K_rr_full, D2_qr
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# Baseline CMMD (CLIP global)
# ============================================================

def evaluate_cmmd(anchor_paths: list, raise_items: list, device: str) -> dict:
    """CMMD standard (global, pas de kNN) comme baseline."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    logger.info("\n" + "=" * 70)
    logger.info(f"CMMD — {CLIP_MODEL_ID}")
    logger.info("=" * 70)

    model = (
        CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
        .to(device).eval()
    )
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def encode(paths):
        embeds = []
        for i in range(0, len(paths), BATCH_SIZE_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i+BATCH_SIZE_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                embeds.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
        return np.concatenate(embeds, axis=0).astype(np.float32)

    logger.info(f"  Embeddings ancrage ({len(anchor_paths)} images)…")
    ref = encode(anchor_paths)

    query_paths = [item[0] for item in raise_items]
    mos_r       = np.array([item[1] for item in raise_items])

    logger.info(f"  Embeddings requêtes ({len(query_paths)} images)…")
    qry = encode(query_paths)

    # Calcul global MMD² biaisé (k=N_ANCHOR)
    ref_sq    = (ref * ref).sum(axis=1)
    D2_rr     = np.maximum(ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T), 0.0)
    K_rr_full = np.exp(-CMMD_GAMMA * D2_rr).astype(np.float32)
    mean_Krr  = float(K_rr_full.mean())

    qry_sq = (qry * qry).sum(axis=1)
    D2_qr  = np.maximum(ref_sq[:, None] + qry_sq[None, :] - 2.0 * (ref @ qry.T), 0.0)
    K_qr   = np.exp(-CMMD_GAMMA * D2_qr)   # (n_anc, n_query)

    scores = mean_Krr + 1.0 - 2.0 * K_qr.mean(axis=0)   # (n_query,)
    rho, p = spearmanr(scores, mos_r)

    logger.info(f"  CMMD global  rho={rho:+.4f}  -rho={-rho:+.4f}  p={p:.3e}")

    del model, proc, ref, qry, D2_rr, K_rr_full, D2_qr, K_qr
    torch.cuda.empty_cache()
    gc.collect()

    return {"scores": scores, "rho": float(rho), "neg_rho": float(-rho), "p": float(p)}


# ============================================================
# Plots
# ============================================================

def plot_neg_rho_vs_k(backbone_results: dict, cmmd_neg_rho: float | None,
                       out_path: Path):
    """
    Courbe -rho(kNN-MMD, MOS) vs k pour chaque backbone.
    CMMD (global) affiché en ligne pointillée.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"SD-VAE l9":  "#1f77b4",
              "DC-AE l10":  "#d62728",
              "DINOv2-B l5": "#2ca02c"}
    markers = {"SD-VAE l9": "o", "DC-AE l10": "s", "DINOv2-B l5": "^"}

    for name, k_results in backbone_results.items():
        ks      = sorted(k_results.keys())
        neg_rho = [k_results[k]["neg_rho"] for k in ks]
        ax.plot(ks, neg_rho,
                color=colors.get(name, "gray"),
                marker=markers.get(name, "o"),
                label=name, linewidth=2, markersize=6)

    if cmmd_neg_rho is not None:
        ax.axhline(cmmd_neg_rho, color="purple", linestyle="--", linewidth=2,
                   label=f"CMMD ({cmmd_neg_rho:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("k (nombre de voisins)", fontsize=12)
    ax.set_ylabel("−ρ (Spearman MOS_Rating)", fontsize=12)
    ax.set_title("kNN-MMD² — monotonicité RAISE (480 images IA vs COCO)\n"
                 "Valeur positive = corrélation correcte", fontsize=13)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES], rotation=30)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Plot sauvegardé : {out_path}")


def plot_scores_vs_mos(backbone_results: dict, raise_items: list,
                        cmmd_result: dict | None, out_path: Path):
    """
    Scatter : score kNN-MMD vs MOS_Rating pour le meilleur k de chaque backbone + CMMD.
    """
    mos_r = np.array([item[1] for item in raise_items])

    colors = {"SD-VAE l9":  "#1f77b4",
              "DC-AE l10":  "#d62728",
              "DINOv2-B l5": "#2ca02c"}

    n = len(backbone_results) + (1 if cmmd_result else 0)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    ax_iter = iter(axes)
    for name, k_results in backbone_results.items():
        best_k  = max(k_results, key=lambda k: k_results[k]["neg_rho"])
        scores  = k_results[best_k]["scores"]
        neg_rho = k_results[best_k]["neg_rho"]
        ax = next(ax_iter)
        ax.scatter(mos_r, scores, s=6, alpha=0.5, color=colors.get(name, "gray"))
        ax.set_xlabel("MOS_Rating", fontsize=10)
        ax.set_ylabel("kNN-MMD² score", fontsize=10)
        ax.set_title(f"{name}\nk={best_k}  −ρ={neg_rho:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    if cmmd_result:
        ax = next(ax_iter)
        ax.scatter(mos_r, cmmd_result["scores"], s=6, alpha=0.5, color="purple")
        ax.set_xlabel("MOS_Rating", fontsize=10)
        ax.set_ylabel("CMMD score", fontsize=10)
        ax.set_title(f"CMMD\n−ρ={cmmd_result['neg_rho']:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Score vs MOS — meilleur k par backbone (RAISE, mode point)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Plot sauvegardé : {out_path}")


# ============================================================
# Sauvegarde CSV
# ============================================================

def save_csv(backbone_results: dict, cmmd_result: dict | None, out_path: Path):
    rows = []
    for name, k_results in backbone_results.items():
        for k in sorted(k_results):
            r = k_results[k]
            rows.append({
                "backbone":  name,
                "k":         k,
                "rho":       r["rho"],
                "neg_rho":   r["neg_rho"],
                "p_value":   r["p"],
            })
    if cmmd_result:
        rows.append({
            "backbone": "CMMD",
            "k":        N_ANCHOR,
            "rho":      cmmd_result["rho"],
            "neg_rho":  cmmd_result["neg_rho"],
            "p_value":  cmmd_result["p"],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "k", "rho", "neg_rho", "p_value"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  CSV sauvegardé : {out_path}")


def print_summary(backbone_results: dict, cmmd_result: dict | None):
    SEP = "=" * 90
    logger.info("\n" + SEP)
    logger.info("RÉSUMÉ — kNN-MMD² RAISE (480 images IA vs COCO, mode point)")
    logger.info(SEP)
    logger.info(f"  {'Backbone':<20} {'k':>6}  {'rho':>9}  {'-rho':>9}  {'p':>10}  verdict")
    logger.info("-" * 90)

    for name, k_results in backbone_results.items():
        best_k = max(k_results, key=lambda k: k_results[k]["neg_rho"])
        for k in sorted(k_results):
            r    = k_results[k]
            star = " ★" if k == best_k else ""
            v    = ("bon" if r["neg_rho"] > 0.5 else
                    "partiel" if r["neg_rho"] > 0.2 else
                    "neutre" if r["neg_rho"] > -0.2 else "inverse")
            logger.info(
                f"  {name:<20} {k:>6}  {r['rho']:>+9.4f}  {r['neg_rho']:>+9.4f}  "
                f"{r['p']:>10.3e}  {v}{star}"
            )

    if cmmd_result:
        r = cmmd_result
        v = ("bon" if r["neg_rho"] > 0.5 else
             "partiel" if r["neg_rho"] > 0.2 else
             "neutre" if r["neg_rho"] > -0.2 else "inverse")
        logger.info(
            f"  {'CMMD (global)':<20} {N_ANCHOR:>6}  {r['rho']:>+9.4f}  "
            f"{r['neg_rho']:>+9.4f}  {r['p']:>10.3e}  {v}"
        )
    logger.info(SEP)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="kNN-MMD² sur RAISE (480 images IA vs COCO, mode point)"
    )
    p.add_argument("--device", default=None,
                   help="cuda / cpu (auto-detect si omis)")
    p.add_argument("--skip-cmmd", action="store_true",
                   help="Ne pas évaluer CMMD")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("kNN-MMD² — RAISE monotonicité (mode point, N=1)")
    logger.info(f"  K_VALUES = {K_VALUES}")
    logger.info(f"  Gamma    = {GAMMA:.2e}")
    logger.info(f"  N_ANCHOR = {N_ANCHOR}  (COCO)")
    logger.info(f"  Device   = {device}")
    logger.info("=" * 70)

    anchor_paths = load_coco_anchor()
    raise_items  = load_raise_data()

    # ── Backbones ────────────────────────────────────────────────────────
    backbone_results = {}
    for backbone_name, layer_idx, display_name, batch_size in BACKBONE_CONFIGS:
        try:
            k_results = evaluate_backbone(
                backbone_name, layer_idx, display_name, batch_size,
                anchor_paths, raise_items, device,
            )
            backbone_results[display_name] = k_results
        except Exception as exc:
            logger.error(f"  ERREUR {display_name} : {exc}", exc_info=True)

    # ── CMMD ─────────────────────────────────────────────────────────────
    cmmd_result = None
    if ENABLE_CMMD and not args.skip_cmmd:
        try:
            cmmd_result = evaluate_cmmd(anchor_paths, raise_items, device)
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)

    # ── Résultats ─────────────────────────────────────────────────────────
    print_summary(backbone_results, cmmd_result)

    save_csv(
        backbone_results, cmmd_result,
        OUT_DIR / f"raise_knn_mmd_summary_{ts}.csv",
    )
    plot_neg_rho_vs_k(
        backbone_results,
        cmmd_result["neg_rho"] if cmmd_result else None,
        OUT_DIR / f"raise_knn_mmd_neg_rho_vs_k_{ts}.png",
    )
    plot_scores_vs_mos(
        backbone_results, raise_items, cmmd_result,
        OUT_DIR / f"raise_knn_mmd_scatter_{ts}.png",
    )

    logger.info(f"\nTous les résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
