"""
visualize_kadid_scatter.py
==========================
Visualisation MMD² global vs MOS/niveau de dégradation — KADID-10k.

Pour chaque backbone :
  - Calcule les scores MMD² biaisés (N=1) sur les 10 125 images KADID
    (ancrage = 81 images de référence KADID)
  - Cache les scores dans results/kadid_scatter/cache_<backbone>_l<n>.npy
  - Génère :
    (a) Fig. principale  — violin + strip par niveau de dégradation (1→5)
        avec ρ Spearman et taux de monotonie par backbone
    (b) Fig. scatter     — nuage de points (dmos, mmd) par backbone,
        colorié par niveau, avec droite de tendance et ρ

Usage :
    python experiments/visualize_kadid_scatter.py
    python experiments/visualize_kadid_scatter.py --skip-cmmd
    python experiments/visualize_kadid_scatter.py --device cpu
    python experiments/visualize_kadid_scatter.py --no-cache   # recompute
"""

import argparse
import csv
import gc
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
OUT_DIR    = BASE_DIR / "results" / "kadid_scatter"
CACHE_DIR  = OUT_DIR / "cache"

# (backbone, layer_idx, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",        7,  "SD-VAE l7",    16),
    ("dc_ae",        10,  "DC-AE l10",     4),
    ("dinov2_vitb14", 5,  "DINOv2-B l5",   8),
]

GAMMA = 2.8e-5

# CMMD baseline
ENABLE_CMMD     = True
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32

# Couleurs par niveau de dégradation (1=léger … 5=sévère)
LEVEL_COLORS = {1: "#2166ac", 2: "#74add1", 3: "#fdae61",
                4: "#f46d43", 5: "#d73027"}
LEVELS = [1, 2, 3, 4, 5]


# ============================================================
# Chargement KADID
# ============================================================

def load_kadid_csv():
    rows = []
    with open(DMOS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dist = r["dist_img"]
            parts = dist.replace(".png", "").split("_")
            rows.append({
                "dist_img":  dist,
                "ref_img":   r["ref_img"],
                "dmos":      float(r["dmos"]),
                "ref_id":    int(parts[0][1:]),
                "dist_type": int(parts[1]),
                "level":     int(parts[2]),
            })
    logger.info(f"KADID : {len(rows)} images dégradées")
    return rows


# ============================================================
# Calcul MMD global
# ============================================================

def compute_scores_gram(backbone, layer_idx, batch_size,
                        anchor_paths, dist_paths, device):
    """
    Extrait les features Gram + calcule MMD² N=1 biaisé global
    pour les 10 125 images KADID.
    Retourne np.ndarray float64 (10125,).
    """
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
    ref = ext.extract(anchor_paths, fit_transform=True, batch_size=batch_size)
    logger.info(f"  Ancrage {ref.shape}  ({time.time()-t0:.1f}s)")
    ref    = ref.astype(np.float32)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    K_rr   = np.exp(-GAMMA * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)))
    mean_Krr = float(K_rr.mean())
    logger.info(f"  mean_Krr={mean_Krr:.6f}  gamma={GAMMA:.2e}")

    n      = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    chunk  = batch_size

    dataset = ImageDataset(dist_paths, transform=ext.image_transform)
    loader  = DataLoader(dataset, batch_size=chunk, shuffle=False,
                         num_workers=4, pin_memory=(device == "cuda"),
                         persistent_workers=True)
    idx = 0
    t1  = time.time()
    for imgs, _ in loader:
        feats = ext.extract_batch(imgs)
        feats = ext.transform_features(feats, fit_pca=False).astype(np.float32)
        t_sq  = np.einsum("ij,ij->i", feats, feats)
        K_rt  = np.exp(-GAMMA * (ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ feats.T)))
        s     = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        scores[idx:idx + len(s)] = s
        idx  += len(s)
        if idx % 2000 < chunk or idx == n:
            logger.info(f"    {idx}/{n} ({idx/n:.0%})  {time.time()-t1:.1f}s")
        del feats, K_rt
        gc.collect()

    ext.cleanup()
    torch.cuda.empty_cache()
    gc.collect()
    return scores


def compute_scores_cmmd(anchor_paths, dist_paths, device):
    """CMMD (CLIP) global — retourne np.ndarray float64 (10125,)."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    logger.info(f"  Chargement {CLIP_MODEL_ID}…")
    model = (CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
             .to(device).eval())
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def encode(paths):
        out = []
        for i in range(0, len(paths), BATCH_SIZE_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i+BATCH_SIZE_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                out.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    ref    = encode(anchor_paths)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    K_rr   = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + ref_sq[None, :] - 2.0*(ref @ ref.T)))
    mean_Krr = float(K_rr.mean())
    logger.info(f"  CLIP ancrage {ref.shape}  mean_Krr={mean_Krr:.4f}")

    n      = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    chunk  = BATCH_SIZE_CLIP * 4
    t0     = time.time()

    for s in range(0, n, chunk):
        e     = min(s + chunk, n)
        emb   = encode(dist_paths[s:e])
        t_sq  = np.einsum("ij,ij->i", emb, emb)
        K_rt  = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + t_sq[None, :] - 2.0*(ref @ emb.T)))
        scores[s:e] = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        if s % 2000 < chunk or e == n:
            logger.info(f"    CLIP {e}/{n} ({e/n:.0%})  {time.time()-t0:.1f}s")
        del emb, K_rt

    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    return scores


def get_or_compute_scores(backbone, layer_idx, display, batch_size,
                          anchor_paths, dist_paths, device, use_cache):
    """Charge depuis le cache ou calcule, puis met en cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag   = display.replace(" ", "_").replace("/", "-")
    cache = CACHE_DIR / f"{tag}.npy"

    if use_cache and cache.exists():
        logger.info(f"  Cache trouvé : {cache}")
        return np.load(cache)

    logger.info(f"  Calcul des scores {display}…")
    if backbone == "cmmd":
        scores = compute_scores_cmmd(anchor_paths, dist_paths, device)
    else:
        scores = compute_scores_gram(backbone, layer_idx, batch_size,
                                     anchor_paths, dist_paths, device)
    np.save(cache, scores)
    logger.info(f"  Cache sauvegardé : {cache}")
    return scores


# ============================================================
# Statistiques rapides
# ============================================================

def stats(rows, scores):
    levels   = np.array([r["level"] for r in rows])
    dmos     = np.array([r["dmos"]  for r in rows])
    by_level = {lv: scores[levels == lv] for lv in LEVELS}

    # level ↑ = dégradation ↑ → bonne métrique si rho > 0
    rho, _   = spearmanr(scores, levels)

    groups: dict = defaultdict(dict)
    for i, row in enumerate(rows):
        groups[(row["ref_id"], row["dist_type"])][row["level"]] = scores[i]
    n_total = n_strict = 0
    for g in groups.values():
        if len(g) < 5:
            continue
        seq = [g[lv] for lv in LEVELS]
        n_total += 1
        if all(seq[i] < seq[i+1] for i in range(4)):
            n_strict += 1

    return {
        "scores":    scores,    # (10125,) dans l'ordre exact de rows — ne pas réordonner
        "by_level":  by_level,
        "dmos":      dmos,
        "levels":    levels,
        "rho":       float(rho),
        "mono_rate": n_strict / n_total if n_total else 0.0,
        "n_total":   n_total,
    }


# ============================================================
# Figures
# ============================================================

def fig_violin_strip(all_stats: dict, out_path: Path):
    """
    Figure principale : un subplot par metric.
    Chaque subplot = violin + strip par niveau (1→5),
    ligne des moyennes, ρ Spearman et taux de monotonie.
    """
    n  = len(all_stats)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (display, st) in zip(axes, all_stats.items()):
        by_level = st["by_level"]
        data     = [by_level[lv] for lv in LEVELS]

        # --- Violin ---
        vp = ax.violinplot(data, positions=LEVELS, widths=0.65,
                           showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            lv = LEVELS[i]
            body.set_facecolor(LEVEL_COLORS[lv])
            body.set_alpha(0.55)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

        # --- Strip (sous-échantillonné pour lisibilité) ---
        rng = np.random.default_rng(42)
        for lv in LEVELS:
            pts = by_level[lv]
            n_show = min(len(pts), 500)
            idx    = rng.choice(len(pts), n_show, replace=False)
            jitter = rng.normal(0, 0.06, n_show)
            ax.scatter(lv + jitter, pts[idx],
                       s=3, alpha=0.25, color=LEVEL_COLORS[lv],
                       linewidths=0, zorder=2)

        # --- Ligne des moyennes ---
        means = [float(np.mean(by_level[lv])) for lv in LEVELS]
        ax.plot(LEVELS, means, "k-o", linewidth=2, markersize=6, zorder=5,
                label="Moyenne")

        # --- Annotation ρ + mono ---
        ax.text(0.04, 0.97,
                f"ρ (Spearman) = {st['rho']:+.3f}\n"
                f"Mono. stricte = {st['mono_rate']:.1%}\n"
                f"({st['n_total']} groupes)",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="gray"))

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_xlabel("Niveau de dégradation  (1 = léger → 5 = sévère)", fontsize=10)
        ax.set_ylabel("Score MMD²", fontsize=10)
        ax.set_xticks(LEVELS)
        ax.grid(True, axis="y", alpha=0.3)

    # Légende niveaux
    handles = [mpatches.Patch(color=LEVEL_COLORS[lv], label=f"Niveau {lv}")
               for lv in LEVELS]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Distribution MMD² par niveau de dégradation — KADID-10k\n"
                 "(ancrage = 81 images de référence, γ = 2.8×10⁻⁵)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Violin plot sauvegardé : {out_path}")


def fig_scatter_dmos(all_stats: dict, out_path: Path):
    """
    Figure secondaire : scatter (dmos, mmd_score) par metric, colorié par niveau.
    Affiche droite de régression + ρ Spearman.
    """
    n  = len(all_stats)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, (display, st) in zip(axes, all_stats.items()):
        dmos       = st["dmos"]    # (10125,) — ordre rows
        scores     = st["scores"]  # (10125,) — même ordre rows → aligné avec dmos
        levels_arr = st["levels"]

        # Sous-échantillonnage pour lisibilité (2000 pts max)
        n_show = min(len(dmos), 2000)
        idx    = rng.choice(len(dmos), n_show, replace=False)
        x_sub  = dmos[idx]
        y_sub  = scores[idx]
        lv_sub = levels_arr[idx]

        for lv in LEVELS:
            mask = lv_sub == lv
            ax.scatter(x_sub[mask], y_sub[mask],
                       s=6, alpha=0.4, color=LEVEL_COLORS[lv],
                       linewidths=0, label=f"niv. {lv}")

        # Droite de tendance (régression linéaire sur tous les points)
        m, b = np.polyfit(dmos, scores, 1)
        x_fit = np.linspace(dmos.min(), dmos.max(), 200)
        ax.plot(x_fit, m * x_fit + b, "k--", linewidth=1.5, zorder=10,
                label="tendance")

        rho, _ = spearmanr(dmos, scores)
        ax.text(0.04, 0.97, f"ρ (Spearman) = {rho:+.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="gray"))

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_xlabel("dmos  (valeur haute = bonne qualité = peu dégradé)", fontsize=10)
        ax.set_ylabel("Score MMD²", fontsize=10)
        ax.legend(fontsize=8, markerscale=2, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Nuage de points : Score MMD² vs MOS (dmos) — KADID-10k\n"
                 "(2 000 points / metric, colorés par niveau de dégradation)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Scatter dmos sauvegardé : {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualisation scatter + violin MMD² KADID-10k"
    )
    p.add_argument("--device", default=None,
                   help="cuda / cpu (auto-detect si omis)")
    p.add_argument("--skip-cmmd", action="store_true",
                   help="Ne pas évaluer CMMD")
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute scores même si le cache existe")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_cache = not args.no_cache

    logger.info("=" * 70)
    logger.info("Visualisation scatter MMD² — KADID-10k")
    logger.info(f"  Gamma   = {GAMMA:.2e}")
    logger.info(f"  Device  = {device}")
    logger.info(f"  Cache   = {use_cache}")
    logger.info("=" * 70)

    rows       = load_kadid_csv()
    ref_names  = sorted({r["ref_img"] for r in rows})
    anchor_paths = [str(IMAGES_DIR / n)             for n in ref_names]
    dist_paths   = [str(IMAGES_DIR / r["dist_img"]) for r in rows]

    all_stats = {}

    # ── Gram backbones ────────────────────────────────────────────────
    for backbone, layer_idx, display, batch_size in BACKBONE_CONFIGS:
        logger.info(f"\n── {display} ──")
        try:
            scores = get_or_compute_scores(
                backbone, layer_idx, display, batch_size,
                anchor_paths, dist_paths, device, use_cache,
            )
            all_stats[display] = stats(rows, scores)
            st = all_stats[display]
            logger.info(
                f"  ρ={st['rho']:+.4f}  mono={st['mono_rate']:.1%}"
            )
        except Exception as exc:
            logger.error(f"  ERREUR {display} : {exc}", exc_info=True)

    # ── CMMD ─────────────────────────────────────────────────────────
    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info("\n── CMMD (CLIP ViT-L/14@336) ──")
        try:
            scores = get_or_compute_scores(
                "cmmd", 0, "CMMD", BATCH_SIZE_CLIP,
                anchor_paths, dist_paths, device, use_cache,
            )
            all_stats["CMMD"] = stats(rows, scores)
            st = all_stats["CMMD"]
            logger.info(f"  ρ={st['rho']:+.4f}  mono={st['mono_rate']:.1%}")
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)

    if not all_stats:
        logger.error("Aucun résultat.")
        return

    # dmos et levels sont déjà stockés dans all_stats par stats()
    # (dans l'ordre de rows, alignés avec scores)

    # ── Figures ──────────────────────────────────────────────────────
    fig_violin_strip(
        all_stats,
        OUT_DIR / f"violin_par_niveau_{ts}.png",
    )
    fig_scatter_dmos(
        all_stats,
        OUT_DIR / f"scatter_dmos_{ts}.png",
    )

    logger.info(f"\nTous les résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
