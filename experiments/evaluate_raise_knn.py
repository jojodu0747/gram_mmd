"""
evaluate_raise_knn.py
=====================
kNN-MMD biaisé pour RAISE monotonicity.

Pour chaque image IA y, on calcule le MMD biaisé (N=1) mais en utilisant
seulement les k plus proches voisins de y dans l'ancrage COCO :

    X_k(y) = k plus proches voisins de y dans X

    MMD²_k(y, X) = mean(K_rr^k) + k(y,y) - (2/k) · Σ_{i ∈ X_k} k(x_i, y)

avec k(y,y) = 1 (kernel RBF).

Avantage vs MMD global : K_rr^k est calculé localement (sur des voisins
proches de y), donc adaptatif. Le terme mean(K_rr) global est dominé par
des paires lointaines qui ne reflètent pas la position de y.

Compare :
  - kNN-MMD pour k ∈ K_VALUES (méthode principale)
  - MMD point global (baseline, k = n_anchor)

Les features sont :
  - Gram spatial, standardisé (StandardScaler fitté sur ancrage)
  - PCA(PCA_COMPONENTS) pour réduire la dimension si nécessaire
  - Backbones : DC-AE l10, DINOv2-B l5
  - CLIP ViT-L/14@336 (embeddings bruts, pas de gram)

Usage :
    python experiments/evaluate_raise_knn.py
    python experiments/evaluate_raise_knn.py --skip-cmmd
    python experiments/evaluate_raise_knn.py --device cpu
    python experiments/evaluate_raise_knn.py --n-anchor 500

Résultats -> results/raise_knn/
"""

import argparse
import csv
import gc
import glob as glob_module
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR      = Path(__file__).resolve().parent.parent
RAISE_ROOT    = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_IMAGES  = [RAISE_ROOT / "images" / "train_images",
                 RAISE_ROOT / "images" / "test_images"]
RAISE_RATINGS = [RAISE_ROOT / "ratings" / "train.csv",
                 RAISE_ROOT / "ratings" / "test.csv"]
COCO_DIR      = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR       = BASE_DIR / "results" / "raise_knn"

N_ANCHOR    = 1000
ANCHOR_SEED = 42

# k values to test
K_VALUES = [1, 5, 10, 25, 50, 100, 200, 500]

# Fixed gamma (same as evaluate_kadid_monotonicity.py)
MMD_GAMMA = 2.8e-5

PCA_COMPONENTS = None  # Pas de réduction dimensionnelle

# Backbones: (backbone_name, layer_idx, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",         7, "SD-VAE l7",       8),
    ("dc_ae",         10, "DC-AE l10",       8),
    ("dinov2_vitb14",  5, "DINOv2-B l5",    64),
]

# CMMD (CLIP ViT-L/14@336) — embeddings bruts 768d, pas de gram
ENABLE_CMMD     = True
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32


# ============================================================
# Data loading
# ============================================================

def load_coco_anchor(n: int = N_ANCHOR, seed: int = ANCHOR_SEED) -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if not all_imgs:
        raise FileNotFoundError(f"Aucune image dans {COCO_DIR}")
    if n < len(all_imgs):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(all_imgs), size=n, replace=False).tolist())
        all_imgs = [all_imgs[i] for i in idx]
    logger.info(f"Ancrage COCO : {len(all_imgs)} images (seed={seed})")
    return all_imgs


def load_raise_fake() -> list:
    """Returns list of (path, MOS_Rating, MOS_ZScore) sorted by MOS_Rating."""
    mos_dict = {}
    for csv_path in RAISE_RATINGS:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = (
                    float(row["MOS_Rating"]),
                    float(row["MOS_ZScore"]),
                )
    all_imgs = {}
    for folder in RAISE_IMAGES:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[os.path.basename(p)] = p

    items = [
        (path, *mos_dict[fname])
        for fname, path in all_imgs.items()
        if fname.startswith("f") and fname in mos_dict
    ]
    items.sort(key=lambda x: x[1])
    logger.info(f"RAISE IA : {len(items)} images  MOS [{items[0][1]:.1f} … {items[-1][1]:.1f}]")
    return items


# ============================================================
# Feature extraction
# ============================================================

def apply_pca(ref_feats: np.ndarray, qry_feats: np.ndarray,
              n_components: int) -> tuple:
    """PCA fittée sur ref, appliquée sur ref + qry."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    ref_pca = pca.fit_transform(ref_feats).astype(np.float32)
    qry_pca = pca.transform(qry_feats).astype(np.float32)
    var_exp = float(pca.explained_variance_ratio_.sum()) * 100
    logger.info(f"  PCA({n_components}) : variance expliquée = {var_exp:.1f}%")
    return ref_pca, qry_pca


def extract_gram_features(backbone: str, layer_idx: int, batch_size: int,
                           anchor_paths: list, query_paths: list) -> tuple:
    """
    Gram spatial + StandardScaler (fitté sur ancrage) + PCA optionnelle.
    Returns (ref_feats, qry_feats).
    """
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
    logger.info(f"  Extraction ancrage ({len(anchor_paths)} imgs, bs={batch_size})...")
    ref_feats = extractor.extract(anchor_paths, fit_transform=True, batch_size=batch_size)
    logger.info(f"  → ref : {ref_feats.shape}  ({ref_feats.nbytes / 1e6:.0f} MB)")

    logger.info(f"  Extraction query ({len(query_paths)} imgs)...")
    qry_feats = extractor.extract(query_paths, fit_transform=False, batch_size=batch_size)
    logger.info(f"  → qry : {qry_feats.shape}  ({qry_feats.nbytes / 1e6:.0f} MB)")

    extractor.cleanup()
    torch.cuda.empty_cache()
    gc.collect()

    if PCA_COMPONENTS is not None and ref_feats.shape[1] > PCA_COMPONENTS:
        ref_feats, qry_feats = apply_pca(ref_feats, qry_feats, PCA_COMPONENTS)

    return ref_feats, qry_feats


def extract_clip_features(paths: list, model, proc, device: str,
                          batch_size: int = BATCH_SIZE_CLIP) -> np.ndarray:
    all_emb = []
    for i in range(0, len(paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            all_emb.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ============================================================
# kNN-MMD scoring
# ============================================================

def build_kernel_cache(ref_feats: np.ndarray, gamma: float,
                       device: str = "cpu") -> tuple:
    """
    Pré-calcule D2_rr et K_rr_full sur GPU si disponible (r @ r.T coûteux
    pour de grandes dimensions gram).
    Retourne tout en numpy sur CPU.
    """
    r_t = torch.from_numpy(ref_feats.astype(np.float32)).to(device)
    with torch.no_grad():
        r_sq_t = (r_t * r_t).sum(dim=1)
        D2_rr_t = torch.clamp(
            r_sq_t[:, None] + r_sq_t[None, :] - 2.0 * (r_t @ r_t.T), min=0.0
        )
        K_rr_t = torch.exp(-gamma * D2_rr_t)
    r_sq    = r_sq_t.cpu().numpy()
    D2_rr   = D2_rr_t.cpu().numpy()
    K_rr_full = K_rr_t.cpu().numpy()
    del r_t, r_sq_t, D2_rr_t, K_rr_t
    torch.cuda.empty_cache()
    return r_sq, D2_rr, K_rr_full


def compute_knn_mmd_scores(query_feats: np.ndarray,
                            ref_feats: np.ndarray,
                            r_sq: np.ndarray,
                            K_rr_full: np.ndarray,
                            D2_qr: np.ndarray,
                            gamma: float,
                            k: int) -> np.ndarray:
    """
    kNN-MMD biaisé : pour chaque query y, MMD²(y, X_k) où
    X_k = k plus proches voisins de y dans X.

    MMD²_k(y, X) = mean(K_rr^k) + 1 - (2/k) · Σ_{i ∈ X_k} k(x_i, y)

    Pré-requis : K_rr_full et D2_qr déjà calculés.
    """
    n_q = query_feats.shape[0]
    n_r = ref_feats.shape[0]
    k_eff = min(k, n_r)

    # Kernel query → anchor (n_q × n_r)
    K_qr = np.exp(-gamma * D2_qr)

    scores = np.empty(n_q, dtype=np.float64)
    for i in range(n_q):
        # k plus proches voisins (indices non triés, argpartition = O(n))
        idx = np.argpartition(D2_qr[i], k_eff - 1)[:k_eff]

        # Kernel ref-ref sur les k voisins : extrait de K_rr_full
        K_rr_k = K_rr_full[np.ix_(idx, idx)]
        term1  = float(K_rr_k.mean())

        # Kernel query-ref sur les k voisins
        term3 = 2.0 / k_eff * float(K_qr[i, idx].sum())

        scores[i] = term1 + 1.0 - term3

    return scores


def compute_global_mmd_point(K_rr_full: np.ndarray,
                              D2_qr: np.ndarray,
                              gamma: float) -> np.ndarray:
    """MMD biaisé global (utilise TOUT l'ancrage — baseline)."""
    n_r = K_rr_full.shape[0]
    term1 = float(K_rr_full.mean())
    K_qr  = np.exp(-gamma * D2_qr)              # (n_q, n_r)
    term3 = 2.0 / n_r * K_qr.sum(axis=1)       # (n_q,)
    return term1 + 1.0 - term3


def precompute_D2_qr(query_feats: np.ndarray, ref_feats: np.ndarray,
                     r_sq: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Matrice (n_q × n_r) des distances carrées query→anchor.
    Utilise le GPU si device != 'cpu' (q @ r.T coûteux en haute dimension).
    """
    q_t = torch.from_numpy(query_feats.astype(np.float32)).to(device)
    r_t = torch.from_numpy(ref_feats.astype(np.float32)).to(device)
    r_sq_t = torch.from_numpy(r_sq.astype(np.float32)).to(device)
    with torch.no_grad():
        q_sq_t = (q_t * q_t).sum(dim=1)
        D2_t = torch.clamp(
            q_sq_t[:, None] + r_sq_t[None, :] - 2.0 * (q_t @ r_t.T), min=0.0
        )
    D2_qr = D2_t.cpu().numpy()
    del q_t, r_t, r_sq_t, q_sq_t, D2_t
    torch.cuda.empty_cache()
    return D2_qr


# ============================================================
# Reporting
# ============================================================

def spearman_report(scores: np.ndarray, mos: np.ndarray, label: str) -> dict:
    rho, p = spearmanr(scores, mos)
    verdict = ("bon"     if -rho > 0.5  else
               "partiel" if -rho > 0.2  else
               "neutre"  if -rho > -0.2 else "inverse")
    sig = "*" if p < 0.05 else " "
    logger.info(
        f"  {label:65s}  -rho={-rho:+.4f}  p={p:.3e}{sig}  [{verdict}]"
    )
    return {
        "metric_label":       label,
        "n":                  len(scores),
        "rho_mos_rating":     float(rho),
        "neg_rho_mos_rating": float(-rho),
        "p_mos_rating":       float(p),
        "verdict":            verdict,
    }


# ============================================================
# Save + plot
# ============================================================

SUMMARY_FIELDS  = ["metric_label", "n", "rho_mos_rating", "neg_rho_mos_rating",
                    "p_mos_rating", "verdict"]
DETAILED_FIELDS = ["metric_label", "img_idx", "mos_rating", "mos_zscore", "score"]


def save_results(summary_rows: list, detailed_rows: list, ts: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sum_path = OUT_DIR / f"knn_mmd_summary_{ts}.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary_rows)

    det_path = OUT_DIR / f"knn_mmd_detailed_{ts}.csv"
    with open(det_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETAILED_FIELDS)
        w.writeheader()
        w.writerows(detailed_rows)

    logger.info(f"\nCSV résumé  : {sum_path}")
    logger.info(f"CSV détaillé : {det_path}")

    # Console table
    sep = "-" * 85
    n   = summary_rows[0]["n"] if summary_rows else 0
    logger.info("\n" + sep)
    logger.info(f"RÉSUMÉ kNN-MMD — N={n} images RAISE IA vs {N_ANCHOR} COCO")
    logger.info(sep)
    logger.info(f"  {'Métrique':65s}  {'-rho':>7s}  {'p':>9s}  {'verdict':>8s}")
    logger.info(sep)
    for r in sorted(summary_rows, key=lambda x: -x["neg_rho_mos_rating"]):
        sig = "*" if r["p_mos_rating"] < 0.05 else " "
        logger.info(
            f"  {r['metric_label']:65s}  {r['neg_rho_mos_rating']:+7.4f}  "
            f"{r['p_mos_rating']:>8.3e}{sig}  {r['verdict']:>8s}"
        )
    logger.info(sep)


def plot_knn_mmd_curve(summary_rows: list, ts: str):
    """Line chart: -rho vs k pour chaque backbone (kNN-MMD seulement)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Grouper par backbone
    backbones: dict[str, list] = {}
    for r in summary_rows:
        lbl = r["metric_label"]
        if "_k" not in lbl:
            continue
        try:
            k = int(lbl.rsplit("_k", 1)[1])
        except (IndexError, ValueError):
            continue
        prefix = lbl.rsplit("_k", 1)[0]
        backbones.setdefault(prefix, []).append((k, r["neg_rho_mos_rating"]))

    if not backbones:
        return

    COLORS = ["#9C27B0", "#FF9800", "#4CAF50", "#2196F3"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for (prefix, points), color in zip(sorted(backbones.items()), COLORS):
        points.sort()
        ks   = [p[0] for p in points]
        rhos = [p[1] for p in points]
        label = (prefix
                 .replace("kNN_MMD_", "")
                 .replace("_vitb14", "")
                 .replace("dinov2", "DINOv2-B")
                 .replace("dc_ae",  "DC-AE")
                 .replace("CLIP_ViT-L14", "CLIP ViT-L/14"))
        ax.plot(ks, rhos, marker="o", color=color, linewidth=2, label=label)
        for k_val, rho in zip(ks, rhos):
            ax.annotate(f"{rho:+.3f}", (k_val, rho),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7.5, color=color)

    # Baseline global MMD
    for r in summary_rows:
        if r["metric_label"].startswith("MMDpt_") or r["metric_label"].startswith("CMMDpt_"):
            ax.axhline(r["neg_rho_mos_rating"], linestyle="--", linewidth=1.2, alpha=0.6,
                       label=f"MMD global ({r['metric_label'].split('_')[1]})")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("k (nombre de voisins dans l'ancrage)", fontsize=11)
    ax.set_ylabel("−ρ (Spearman)  [>0 : bon]", fontsize=11)
    pca_str = f"  |  PCA({PCA_COMPONENTS})" if PCA_COMPONENTS else ""
    ax.set_title(
        f"kNN-MMD vs MOS — RAISE IA (N=480)\n"
        f"Ancrage : {N_ANCHOR} COCO  |  gamma={MMD_GAMMA:.2e}{pca_str}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_DIR / f"knn_mmd_curve_{ts}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique : {path}")


# ============================================================
# Backbone evaluation
# ============================================================

def evaluate_backbone(backbone: str, layer_idx: int, disp_name: str,
                       batch_size: int, anchor_paths: list,
                       query_paths: list, mos_r: np.ndarray,
                       mos_z: np.ndarray, gamma: float,
                       summary_rows: list, detailed_rows: list,
                       device: str = "cpu"):

    logger.info(f"\n{'='*68}")
    logger.info(f"Backbone : {disp_name}")
    logger.info(f"{'='*68}")

    ref_feats, qry_feats = extract_gram_features(
        backbone, layer_idx, batch_size, anchor_paths, query_paths
    )

    # Pré-calculs sur GPU (évite le bottleneck CPU pour les hautes dimensions)
    logger.info(f"  Calcul distances sur {device}...")
    r_sq, D2_rr, K_rr_full = build_kernel_cache(ref_feats, gamma, device=device)
    D2_qr = precompute_D2_qr(qry_feats, ref_feats, r_sq, device=device)
    logger.info(f"  D2_qr : {D2_qr.shape}  K_rr : {K_rr_full.shape}")

    def record(label, scores):
        row = spearman_report(scores, mos_r, label)
        summary_rows.append(row)
        for i, (s, mr, mz) in enumerate(zip(scores, mos_r, mos_z)):
            detailed_rows.append({
                "metric_label": label,
                "img_idx":      i + 1,
                "mos_rating":   float(mr),
                "mos_zscore":   float(mz),
                "score":        float(s),
            })

    # kNN-MMD pour chaque k
    logger.info(f"  kNN-MMD ({len(K_VALUES)} valeurs de k)...")
    for k in K_VALUES:
        sc = compute_knn_mmd_scores(
            qry_feats, ref_feats, r_sq, K_rr_full, D2_qr, gamma, k
        )
        record(f"kNN_MMD_{backbone}_l{layer_idx}_k{k}", sc)

    # Baseline : MMD global (k = n_anchor)
    logger.info(f"  MMD global (baseline)...")
    sc_global = compute_global_mmd_point(K_rr_full, D2_qr, gamma)
    record(f"MMDpt_{backbone}_l{layer_idx}_global", sc_global)

    del ref_feats, qry_feats, r_sq, D2_rr, K_rr_full, D2_qr
    gc.collect()


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="kNN-MMD RAISE — MMD biaisé avec k plus proches voisins"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cmmd", action="store_true")
    parser.add_argument("--n-anchor", type=int, default=N_ANCHOR)
    parser.add_argument("--gamma", type=float, default=MMD_GAMMA)
    return parser.parse_args()


def main():
    args   = parse_args()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = args.device
    gamma  = args.gamma

    logger.info("=" * 68)
    logger.info("kNN-MMD RAISE — MMD biaisé avec k plus proches voisins COCO")
    logger.info(f"  k values      : {K_VALUES}")
    logger.info(f"  Ancrage       : {args.n_anchor} COCO (seed={ANCHOR_SEED})")
    logger.info(f"  Gamma         : {gamma:.2e}")
    logger.info(f"  PCA           : {PCA_COMPONENTS if PCA_COMPONENTS else 'désactivé'}")
    logger.info(f"  Device        : {device}")
    logger.info("=" * 68)

    anchor_paths = load_coco_anchor(n=args.n_anchor)
    items        = load_raise_fake()
    query_paths  = [x[0] for x in items]
    mos_r        = np.array([x[1] for x in items])
    mos_z        = np.array([x[2] for x in items])

    summary_rows:  list[dict] = []
    detailed_rows: list[dict] = []

    # ── Gram spatial backbones ────────────────────────────────────────────
    for backbone, layer_idx, disp_name, batch_size in BACKBONE_CONFIGS:
        try:
            evaluate_backbone(
                backbone, layer_idx, disp_name, batch_size,
                anchor_paths, query_paths, mos_r, mos_z, gamma,
                summary_rows, detailed_rows, device=device,
            )
        except Exception as exc:
            logger.error(f"ERREUR {backbone}: {exc}", exc_info=True)

    # ── CLIP embeddings ───────────────────────────────────────────────────
    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info(f"\n{'='*68}")
        logger.info("CLIP ViT-L/14@336 — embeddings 768d")
        logger.info(f"{'='*68}")
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
            clip_model = clip_model.to(device).eval()
            clip_proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

            logger.info(f"  Extraction ancrage...")
            ref_clip = extract_clip_features(anchor_paths, clip_model, clip_proc, device)
            logger.info(f"  Extraction query...")
            qry_clip = extract_clip_features(query_paths, clip_model, clip_proc, device)
            logger.info(f"  ref={ref_clip.shape}  qry={qry_clip.shape}")

            r_sq_c, _, K_rr_c = build_kernel_cache(ref_clip, CMMD_GAMMA, device=device)
            D2_qr_c = precompute_D2_qr(qry_clip, ref_clip, r_sq_c, device=device)

            def record_clip(label, scores):
                row = spearman_report(scores, mos_r, label)
                summary_rows.append(row)
                for i, (s, mr, mz) in enumerate(zip(scores, mos_r, mos_z)):
                    detailed_rows.append({
                        "metric_label": label,
                        "img_idx": i + 1,
                        "mos_rating": float(mr),
                        "mos_zscore": float(mz),
                        "score": float(s),
                    })

            logger.info(f"  kNN-MMD CLIP ({len(K_VALUES)} valeurs de k)...")
            for k in K_VALUES:
                sc = compute_knn_mmd_scores(
                    qry_clip, ref_clip, r_sq_c, K_rr_c, D2_qr_c, CMMD_GAMMA, k
                )
                record_clip(f"kNN_MMD_CLIP_ViT-L14_k{k}", sc)

            sc_global = compute_global_mmd_point(K_rr_c, D2_qr_c, CMMD_GAMMA)
            record_clip("CMMDpt_CLIP_ViT-L14_global", sc_global)

            del clip_model, clip_proc, ref_clip, qry_clip, K_rr_c, D2_qr_c
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as exc:
            logger.error(f"ERREUR CLIP: {exc}", exc_info=True)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    if summary_rows:
        save_results(summary_rows, detailed_rows, ts)
        plot_knn_mmd_curve(summary_rows, ts)

    logger.info(f"\nTous les résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
