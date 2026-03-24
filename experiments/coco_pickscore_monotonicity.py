"""
coco_pickscore_monotonicity.py
================================
Test de monotonicité PickScore ↔ MMD sur COCO 5k :
  • 2000 images COCO scorées avec PickScore (réalisme)
  • 100 groupes de 20 images classées par PickScore croissant
  • Ancrage : 1000 autres images COCO (indices distincts, seed=42)
  • Métriques : Spearman ρ, Kendall τ, monotonicité stricte (% paires)
  • Comparaison : Gram-MMD (sd_vae L11, dinov2_vitb14 L3) vs CMMD

Sorties dans results/coco_pickscore_mono/ :
  coco_pickscore_mono_detailed_{ts}.csv
  coco_pickscore_mono_summary_{ts}.csv
  coco_pickscore_scores_{ts}.csv          ← scores PickScore par image

Usage :
    .venv/bin/python experiments/coco_pickscore_monotonicity.py
    .venv/bin/python experiments/coco_pickscore_monotonicity.py --skip-cmmd
    .venv/bin/python experiments/coco_pickscore_monotonicity.py --scores-cache path/to/scores.csv
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
from scipy.stats import spearmanr, kendalltau

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

COCO_DIR    = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR     = BASE_DIR / "results" / "coco_pickscore_mono"

N_SCORED    = 2000   # images scorées avec PickScore
N_ANCHOR    = 1000   # images d'ancrage (disjointes des 2000)
N_GROUPS    = 100    # nombre de groupes
GROUP_SIZE  = 20     # taille de chaque groupe (100 × 20 = 2000)
SEED        = 42

PICKSCORE_PROC  = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL = "yuvalkirstain/PickScore_v1"
PICKSCORE_PROMPT = "a high quality realistic photograph"
BATCH_PICK  = 16

# Backbones Gram-MMD
SELECTED_CONFIGS = [
    {"backbone": "sd_vae",        "layer": 11, "gamma": 1e-4,   "batch_size": 16},
    {"backbone": "dinov2_vitb14", "layer": 3,  "gamma": 1e-6,   "batch_size": 8},
]

ENABLE_CMMD   = True
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005
BATCH_CLIP    = 32


# ═══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES IMAGES COCO
# ═══════════════════════════════════════════════════════════════════════

def load_all_coco_paths() -> list:
    paths = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if len(paths) < N_SCORED + N_ANCHOR:
        raise FileNotFoundError(
            f"Besoin d'au moins {N_SCORED + N_ANCHOR} images dans {COCO_DIR}, "
            f"trouvé {len(paths)}"
        )
    return paths


def split_coco_paths(all_paths: list) -> tuple:
    """Retourne (scored_paths, anchor_paths) en indices disjoints, seed fixée."""
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(all_paths), size=N_SCORED + N_ANCHOR, replace=False)
    chosen_sorted = sorted(chosen.tolist())
    scored_idx = chosen_sorted[:N_SCORED]
    anchor_idx = chosen_sorted[N_SCORED:]
    scored_paths = [all_paths[i] for i in scored_idx]
    anchor_paths = [all_paths[i] for i in anchor_idx]
    logger.info(f"Images scorées : {len(scored_paths)}")
    logger.info(f"Images ancrage : {len(anchor_paths)}")
    return scored_paths, anchor_paths


# ═══════════════════════════════════════════════════════════════════════
#  PICKSCORE — SCORING DES IMAGES
# ═══════════════════════════════════════════════════════════════════════

def compute_pickscore(paths: list, device: str) -> np.ndarray:
    """
    Score chaque image avec PickScore contre le prompt PICKSCORE_PROMPT.
    Retourne un vecteur de scores float32 de longueur len(paths).
    """
    from transformers import AutoProcessor, AutoModel

    logger.info(f"Chargement PickScore : {PICKSCORE_MODEL}")
    processor = AutoProcessor.from_pretrained(PICKSCORE_PROC)
    model = AutoModel.from_pretrained(PICKSCORE_MODEL).eval().to(device)

    scores = []
    n = len(paths)
    for start in range(0, n, BATCH_PICK):
        batch_paths = paths[start:start + BATCH_PICK]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(
            text=[PICKSCORE_PROMPT] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"].to(device),
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
            image_embs = outputs.image_embeds
            text_embs  = outputs.text_embeds
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            text_embs  = text_embs  / text_embs.norm(dim=-1, keepdim=True)
            batch_scores = (image_embs * text_embs).sum(dim=-1)
            scores.append(batch_scores.cpu().float().numpy())

        if (start // BATCH_PICK + 1) % 20 == 0:
            logger.info(f"  PickScore : {start + len(batch_paths)}/{n} images")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    result = np.concatenate(scores, axis=0)
    logger.info(
        f"PickScore — min={result.min():.4f}  max={result.max():.4f}  "
        f"mean={result.mean():.4f}  std={result.std():.4f}"
    )
    return result


def save_scores_csv(paths: list, scores: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "pickscore"])
        w.writeheader()
        for p, s in zip(paths, scores):
            w.writerow({"path": p, "pickscore": float(s)})
    logger.info(f"Scores sauvegardés : {path}")


def load_scores_csv(path: str) -> tuple:
    paths, scores = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            scores.append(float(row["pickscore"]))
    return paths, np.array(scores, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  CONSTITUTION DES GROUPES
# ═══════════════════════════════════════════════════════════════════════

def make_groups(paths: list, scores: np.ndarray) -> list:
    """
    Trie par PickScore croissant, puis découpe en N_GROUPS × GROUP_SIZE.
    Retourne une liste de dicts {paths, score_mean, score_min, score_max}.
    """
    order = np.argsort(scores)
    sorted_paths  = [paths[i]  for i in order]
    sorted_scores = scores[order]

    total = N_GROUPS * GROUP_SIZE
    sorted_paths  = sorted_paths[:total]
    sorted_scores = sorted_scores[:total]

    groups = []
    for i in range(N_GROUPS):
        sl = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
        groups.append({
            "paths":       sorted_paths[sl],
            "score_mean":  float(np.mean(sorted_scores[sl])),
            "score_min":   float(sorted_scores[sl][0]),
            "score_max":   float(sorted_scores[sl][-1]),
        })
    logger.info(
        f"Groupes : {N_GROUPS} × {GROUP_SIZE}  "
        f"score [{groups[0]['score_mean']:.4f} … {groups[-1]['score_mean']:.4f}]"
    )
    return groups, sorted_paths


# ═══════════════════════════════════════════════════════════════════════
#  MMD² non-biaisé
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
#  STATISTIQUES DE MONOTONICITÉ
# ═══════════════════════════════════════════════════════════════════════

def strict_monotonicity(scores: list) -> float:
    """Fraction de paires consécutives correctement ordonnées (score croissant)."""
    s = np.array(scores)
    diffs = np.diff(s)
    return float(np.mean(diffs > 0))


def compute_stats(mmd_scores: list, groups: list, label: str) -> dict:
    s   = np.array(mmd_scores)
    mos = np.array([g["score_mean"] for g in groups])

    rho,  p_rho = spearmanr(s, mos)
    tau,  p_tau = kendalltau(s, mos)
    strict = strict_monotonicity(mmd_scores)

    logger.info(
        f"  {label:55s}  "
        f"ρ={rho:+.4f} (p={p_rho:.2e})  "
        f"τ={tau:+.4f} (p={p_tau:.2e})  "
        f"strict={strict:.3f}"
    )
    return {
        "label":          label,
        "n_groups":       len(mmd_scores),
        "spearman_rho":   float(rho),
        "p_spearman":     float(p_rho),
        "kendall_tau":    float(tau),
        "p_kendall":      float(p_tau),
        "strict_mono":    strict,
    }


# ═══════════════════════════════════════════════════════════════════════
#  SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════

DETAIL_FIELDS = [
    "label", "group_idx",
    "score_mean", "score_min", "score_max",
    "mmd_score",
]
SUMMARY_FIELDS = [
    "label", "n_groups",
    "spearman_rho", "p_spearman",
    "kendall_tau",  "p_kendall",
    "strict_mono",
]


def save_csv(rows: list, path: Path, fields: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  Sauvegardé : {path}")


def make_detail_rows(mmd_scores: list, groups: list, label: str) -> list:
    return [
        {
            "label":      label,
            "group_idx":  i + 1,
            "score_mean": g["score_mean"],
            "score_min":  g["score_min"],
            "score_max":  g["score_max"],
            "mmd_score":  score,
        }
        for i, (score, g) in enumerate(zip(mmd_scores, groups))
    ]


# ═══════════════════════════════════════════════════════════════════════
#  ÉVALUATION GRAM-MMD
# ═══════════════════════════════════════════════════════════════════════

def evaluate_gram_backbone(cfg: dict, anchor_paths: list,
                            sorted_paths: list, groups: list,
                            detail_rows: list, summary_rows: list):
    backbone  = cfg["backbone"]
    layer     = cfg["layer"]
    gamma     = cfg["gamma"]
    bs        = cfg["batch_size"]
    label     = f"GramMMD_{backbone}_L{layer}_g{gamma:.1e}"

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

    logger.info(f"  Extraction ancrage ({N_ANCHOR} images)...")
    anchor_feat = extractor.extract(anchor_paths, fit_transform=True, batch_size=bs)
    logger.info(f"  anchor_feat : {anchor_feat.shape}")

    term1_rr = compute_term1_rr(anchor_feat, gamma)
    logger.info(f"  gamma={gamma:.4e}  term1_rr={term1_rr:.6f}")

    logger.info(f"  Extraction {N_GROUPS * GROUP_SIZE} images COCO scorées...")
    all_images = [Image.open(p).convert("RGB") for p in sorted_paths]
    all_feat = extractor.extract(all_images, fit_transform=False, batch_size=bs)
    logger.info(f"  all_feat : {all_feat.shape}")

    mmd_scores = []
    for i, grp in enumerate(groups):
        sl = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
        gf = all_feat[sl]
        score = mmd2_unbiased(anchor_feat, gf, gamma, term1_rr)
        mmd_scores.append(score)
        if (i + 1) % 25 == 0 or i == 0:
            logger.info(
                f"    Groupe {i+1}/{N_GROUPS} "
                f"MMD²={score:.5f}  pick={grp['score_mean']:.4f}"
            )

    stats = compute_stats(mmd_scores, groups, label)
    detail_rows += make_detail_rows(mmd_scores, groups, label)
    summary_rows.append(stats)

    extractor.cleanup()
    del extractor, anchor_feat, all_feat, all_images
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  ÉVALUATION CMMD
# ═══════════════════════════════════════════════════════════════════════

def evaluate_cmmd(anchor_paths: list, sorted_paths: list, groups: list,
                  device: str, detail_rows: list, summary_rows: list):
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    label = "CMMD_CLIP-ViT-L14-336"
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

    logger.info(f"  Embeddings ancrage ({N_ANCHOR} images)...")
    anc_emb = embed_paths(anchor_paths)
    D_rr    = sq_dist_matrix(anc_emb, anc_emb)
    K_rr    = np.exp(-CMMD_GAMMA * D_rr)
    n_r     = anc_emb.shape[0]
    term1   = float((np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1)))
    logger.info(f"  gamma={CMMD_GAMMA:.4e}  term1_rr={term1:.6f}")

    logger.info(f"  Embeddings {N_GROUPS * GROUP_SIZE} images COCO scorées...")
    all_emb = embed_paths(sorted_paths)
    logger.info(f"  all_emb : {all_emb.shape}")

    mmd_scores = []
    for i, grp in enumerate(groups):
        sl = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
        gf = all_emb[sl]
        score = mmd2_unbiased(anc_emb, gf, CMMD_GAMMA, term1)
        mmd_scores.append(score)

    stats = compute_stats(mmd_scores, groups, label)
    detail_rows += make_detail_rows(mmd_scores, groups, label)
    summary_rows.append(stats)

    del model, proc, anc_emb, all_emb
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cmmd",     action="store_true",
                   help="Ne pas évaluer CMMD")
    p.add_argument("--scores-cache",  default=None,
                   help="Chemin vers un CSV de scores PickScore déjà calculés")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("COCO PickScore Monotonicity")
    logger.info(f"  Images scorées : {N_SCORED}  |  Ancrage : {N_ANCHOR}")
    logger.info(f"  Groupes : {N_GROUPS} × {GROUP_SIZE}")
    logger.info(f"  Backbones : {[c['backbone'] for c in SELECTED_CONFIGS]}")
    logger.info(f"  CMMD : {ENABLE_CMMD and not args.skip_cmmd}")
    logger.info(f"  Device : {args.device}")
    logger.info("=" * 70)

    # ── 1. Chargement et partitionnement des images ──────────────────────
    all_paths = load_all_coco_paths()
    scored_paths, anchor_paths = split_coco_paths(all_paths)

    # ── 2. Scores PickScore ──────────────────────────────────────────────
    if args.scores_cache:
        logger.info(f"Chargement scores depuis : {args.scores_cache}")
        cached_paths, pickscore_scores = load_scores_csv(args.scores_cache)
        # Réaligner les chemins si nécessaire
        if len(cached_paths) == len(scored_paths):
            pickscore_scores = pickscore_scores
        else:
            raise ValueError(
                f"Cache : {len(cached_paths)} entrées, "
                f"attendu {len(scored_paths)}"
            )
    else:
        pickscore_scores = compute_pickscore(scored_paths, args.device)
        scores_path = OUT_DIR / f"coco_pickscore_scores_{ts}.csv"
        save_scores_csv(scored_paths, pickscore_scores, scores_path)

    # ── 3. Constitution des groupes ──────────────────────────────────────
    groups, sorted_paths = make_groups(scored_paths, pickscore_scores)

    # ── 4. Évaluation ────────────────────────────────────────────────────
    detail_rows  = []
    summary_rows = []

    for cfg in SELECTED_CONFIGS:
        try:
            evaluate_gram_backbone(
                cfg, anchor_paths, sorted_paths, groups,
                detail_rows, summary_rows
            )
        except Exception as exc:
            logger.error(f"ERREUR {cfg['backbone']} : {exc}", exc_info=True)

    if ENABLE_CMMD and not args.skip_cmmd:
        try:
            evaluate_cmmd(
                anchor_paths, sorted_paths, groups, args.device,
                detail_rows, summary_rows
            )
        except Exception as exc:
            logger.error(f"ERREUR CMMD : {exc}", exc_info=True)

    # ── 5. Sauvegarde ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(detail_rows,  OUT_DIR / f"coco_pickscore_mono_detailed_{ts}.csv",  DETAIL_FIELDS)
    save_csv(summary_rows, OUT_DIR / f"coco_pickscore_mono_summary_{ts}.csv",   SUMMARY_FIELDS)

    # ── 6. Résumé terminal ────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ FINAL")
    logger.info("=" * 80)
    for r in sorted(summary_rows, key=lambda x: -x["spearman_rho"]):
        logger.info(
            f"  {r['label']:55s}  "
            f"ρ={r['spearman_rho']:+.4f}  "
            f"τ={r['kendall_tau']:+.4f}  "
            f"strict={r['strict_mono']:.3f}"
        )
    logger.info(f"\nRésultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
