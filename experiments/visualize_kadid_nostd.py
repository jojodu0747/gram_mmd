"""
visualize_kadid_nostd.py
========================
Meme protocole que visualize_kadid_scatter.py mais SANS standardisation
des vecteurs Gram.

Differences vs visualize_kadid_scatter.py :
  - fit_transform=False (aucun StandardScaler)
  - Gamma adaptatif par backbone (heuristique mediane sur les ancres)
  - Teste aussi gamma=2.8e-5 (reference) + 1e-5 + 1e-6 pour comparaison
  - Cache separe dans results/kadid_nostd/cache/

Usage :
    python experiments/visualize_kadid_nostd.py
    python experiments/visualize_kadid_nostd.py --skip-cmmd
    python experiments/visualize_kadid_nostd.py --no-cache
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
OUT_DIR    = BASE_DIR / "results" / "kadid_nostd"
CACHE_DIR  = OUT_DIR / "cache"

# (backbone, layer_idx, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",        7,  "SD-VAE l7 (nostd)",    16),
    ("dc_ae",        10,  "DC-AE l10 (nostd)",     4),
    ("dinov2_vitb14", 5,  "DINOv2-B l5 (nostd)",   8),
]

# Gammas fixes a tester en plus de la mediane
EXTRA_GAMMAS = [2.8e-5, 1e-5, 1e-6]

# CMMD baseline (inchange — pas de standardisation dans CMMD)
ENABLE_CMMD     = True
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32

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
            dist  = r["dist_img"]
            parts = dist.replace(".png", "").split("_")
            rows.append({
                "dist_img":  dist,
                "ref_img":   r["ref_img"],
                "dmos":      float(r["dmos"]),
                "ref_id":    int(parts[0][1:]),
                "dist_type": int(parts[1]),
                "level":     int(parts[2]),
            })
    logger.info(f"KADID : {len(rows)} images")
    return rows


# ============================================================
# Extraction ancrage (sans std) + calcul gamma mediane
# ============================================================

def extract_anchor_nostd(backbone, layer_idx, batch_size, anchor_paths, device):
    """Extrait les features d'ancrage SANS standardisation.
    Retourne (extractor, ref, ref_sq, gamma_median).
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
    t0  = time.time()
    # fit_transform=False -> aucun StandardScaler
    ref = ext.extract(anchor_paths, fit_transform=False, batch_size=batch_size)
    logger.info(f"  Ancrage {ref.shape}  ({time.time()-t0:.1f}s)  "
                f"mean={ref.mean():.4e}  std={ref.std():.4e}")

    ref    = ref.astype(np.float32)
    ref_sq = np.einsum("ij,ij->i", ref, ref)

    # Heuristique mediane : gamma = 1 / (2 * median(||xi - xj||^2))
    D_rr      = np.maximum(ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T), 0.0)
    triu_vals = D_rr[np.triu_indices(len(ref), k=1)]
    med_dist  = float(np.median(triu_vals))
    gamma_med = 1.0 / (2.0 * med_dist + 1e-8)
    logger.info(f"  Distance mediane ancrage = {med_dist:.6e}  "
                f"=> gamma_mediane = {gamma_med:.6e}")

    return ext, ref, ref_sq, gamma_med


# ============================================================
# Calcul scores per-image (MMD² N=1 biaise) — sans std
# ============================================================

def compute_scores_nostd(backbone, layer_idx, batch_size,
                          anchor_paths, dist_paths, device, gamma):
    """
    Calcule MMD² N=1 biaise pour 10125 images KADID — sans standardisation.
    gamma peut etre un float fixe ou None (=> heuristique mediane).
    Retourne (scores: np.ndarray, gamma_used: float).
    """
    ext, ref, ref_sq, gamma_med = extract_anchor_nostd(
        backbone, layer_idx, batch_size, anchor_paths, device
    )
    gamma_used = gamma_med if gamma is None else gamma

    K_rr     = np.exp(-gamma_used * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)))
    mean_Krr = float(K_rr.mean())
    logger.info(f"  gamma_used={gamma_used:.6e}  mean_Krr={mean_Krr:.6f}")

    if mean_Krr < 1e-6:
        logger.warning(f"  ATTENTION : mean_Krr tres faible ({mean_Krr:.2e}) "
                       f"— gamma trop grand pour cette echelle")
    if mean_Krr > 0.9999:
        logger.warning(f"  ATTENTION : mean_Krr proche de 1 ({mean_Krr:.6f}) "
                       f"— gamma trop petit, discriminabilite faible")

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
        # extract_batch -> raw gram sans std
        feats = ext.extract_batch(imgs)
        # transform_features avec fit_pca=False -> pas de scaler
        feats = ext.transform_features(feats, fit_pca=False).astype(np.float32)
        t_sq  = np.einsum("ij,ij->i", feats, feats)
        K_rt  = np.exp(-gamma_used * (ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ feats.T)))
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
    return scores, gamma_used


def compute_scores_cmmd(anchor_paths, dist_paths, device):
    """CMMD (CLIP) global — inchange."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    logger.info(f"  Chargement {CLIP_MODEL_ID}...")
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
        e    = min(s + chunk, n)
        emb  = encode(dist_paths[s:e])
        t_sq = np.einsum("ij,ij->i", emb, emb)
        K_rt = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + t_sq[None, :] - 2.0*(ref @ emb.T)))
        scores[s:e] = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        if s % 2000 < chunk or e == n:
            logger.info(f"    CLIP {e}/{n} ({e/n:.0%})  {time.time()-t0:.1f}s")
        del emb, K_rt

    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    return scores


def get_or_compute_nostd(backbone, layer_idx, display, batch_size,
                          anchor_paths, dist_paths, device, use_cache,
                          gamma=None):
    """Cache-aware wrapper. gamma=None => mediane."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag   = display.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    g_tag = "median" if gamma is None else f"{gamma:.2e}"
    cache = CACHE_DIR / f"{tag}_g{g_tag}.npy"

    if use_cache and cache.exists():
        logger.info(f"  Cache trouve : {cache}")
        # gamma_used n'est pas stocke en cache, on le recalcule si besoin
        return np.load(cache), gamma  # gamma peut etre None ici, c'est ok pour le label

    logger.info(f"  Calcul scores {display} gamma={g_tag}...")
    scores, gamma_used = compute_scores_nostd(
        backbone, layer_idx, batch_size, anchor_paths, dist_paths, device, gamma
    )
    np.save(cache, scores)
    logger.info(f"  Cache sauvegarde : {cache}")
    return scores, gamma_used


# ============================================================
# Statistiques
# ============================================================

def stats(rows, scores):
    levels   = np.array([r["level"] for r in rows])
    dmos     = np.array([r["dmos"]  for r in rows])
    by_level = {lv: scores[levels == lv] for lv in LEVELS}

    rho, _ = spearmanr(scores, levels)

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
        "scores":    scores,
        "by_level":  by_level,
        "dmos":      dmos,
        "levels":    levels,
        "rho":       float(rho),
        "mono_rate": n_strict / n_total if n_total else 0.0,
        "n_total":   n_total,
    }


# ============================================================
# Figures (identiques a visualize_kadid_scatter.py)
# ============================================================

def fig_violin_strip(all_stats: dict, out_path: Path, gamma_labels: dict):
    n  = len(all_stats)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (display, st) in zip(axes, all_stats.items()):
        by_level = st["by_level"]
        data     = [by_level[lv] for lv in LEVELS]

        vp = ax.violinplot(data, positions=LEVELS, widths=0.65,
                           showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(LEVEL_COLORS[LEVELS[i]])
            body.set_alpha(0.55)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

        rng = np.random.default_rng(42)
        for lv in LEVELS:
            pts    = by_level[lv]
            n_show = min(len(pts), 500)
            idx    = rng.choice(len(pts), n_show, replace=False)
            jitter = rng.normal(0, 0.06, n_show)
            ax.scatter(lv + jitter, pts[idx],
                       s=3, alpha=0.25, color=LEVEL_COLORS[lv],
                       linewidths=0, zorder=2)

        means = [float(np.mean(by_level[lv])) for lv in LEVELS]
        ax.plot(LEVELS, means, "k-o", linewidth=2, markersize=6, zorder=5)

        g_label = gamma_labels.get(display, "?")
        ax.text(0.04, 0.97,
                f"rho (Spearman) = {st['rho']:+.3f}\n"
                f"Mono. stricte  = {st['mono_rate']:.1%}\n"
                f"gamma = {g_label}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="gray"))

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_xlabel("Niveau de degradation  (1=leger -> 5=severe)", fontsize=10)
        ax.set_ylabel("Score MMD^2 (sans std)", fontsize=10)
        ax.set_xticks(LEVELS)
        ax.grid(True, axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=LEVEL_COLORS[lv], label=f"Niveau {lv}")
               for lv in LEVELS]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Distribution MMD^2 par niveau — KADID-10k  [SANS standardisation]\n"
                 "(ancrage = 81 images de reference, gamma adaptatif par mediane)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Violin sauvegarde : {out_path}")


def fig_scatter_dmos(all_stats: dict, out_path: Path):
    n  = len(all_stats)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, (display, st) in zip(axes, all_stats.items()):
        dmos       = st["dmos"]
        scores     = st["scores"]
        levels_arr = st["levels"]

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

        m, b  = np.polyfit(dmos, scores, 1)
        x_fit = np.linspace(dmos.min(), dmos.max(), 200)
        ax.plot(x_fit, m * x_fit + b, "k--", linewidth=1.5, zorder=10)

        rho, _ = spearmanr(dmos, scores)
        ax.text(0.04, 0.97, f"rho (Spearman) = {rho:+.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="gray"))

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_xlabel("dmos  (valeur haute = bonne qualite)", fontsize=10)
        ax.set_ylabel("Score MMD^2 (sans std)", fontsize=10)
        ax.legend(fontsize=8, markerscale=2, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scatter MMD^2 vs dmos — KADID-10k  [SANS standardisation]\n"
                 "(2000 points / metric, colories par niveau)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Scatter sauvegarde : {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Visualisation MMD nostd KADID-10k")
    p.add_argument("--device",    default=None)
    p.add_argument("--skip-cmmd", action="store_true")
    p.add_argument("--no-cache",  action="store_true")
    return p.parse_args()


def main():
    args      = parse_args()
    device    = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_cache = not args.no_cache

    logger.info("=" * 70)
    logger.info("Visualisation MMD^2 KADID-10k — SANS STANDARDISATION")
    logger.info(f"  Gamma   = heuristique mediane par backbone")
    logger.info(f"  Device  = {device}")
    logger.info(f"  Cache   = {use_cache}")
    logger.info("=" * 70)

    rows        = load_kadid_csv()
    ref_names   = sorted({r["ref_img"] for r in rows})
    anchor_paths = [str(IMAGES_DIR / n)             for n in ref_names]
    dist_paths   = [str(IMAGES_DIR / r["dist_img"]) for r in rows]

    all_stats   = {}
    gamma_labels = {}  # display -> gamma string pour les figures

    # ── Gram backbones (gamma mediane) ───────────────────────────────
    for backbone, layer_idx, display, batch_size in BACKBONE_CONFIGS:
        logger.info(f"\n-- {display} --")
        try:
            scores, gamma_used = get_or_compute_nostd(
                backbone, layer_idx, display, batch_size,
                anchor_paths, dist_paths, device, use_cache,
                gamma=None,  # mediane
            )
            all_stats[display]    = stats(rows, scores)
            gamma_labels[display] = f"{gamma_used:.2e}" if isinstance(gamma_used, float) else "median"
            st = all_stats[display]
            logger.info(f"  rho={st['rho']:+.4f}  mono={st['mono_rate']:.1%}")
        except Exception as exc:
            logger.error(f"  ERREUR {display} : {exc}", exc_info=True)

    # ── Extra gammas (report only, pas de figure separee) ────────────
    logger.info("\n-- Gammas fixes (comparaison) --")
    summary_extra = []
    for backbone, layer_idx, display, batch_size in BACKBONE_CONFIGS:
        for g in EXTRA_GAMMAS:
            tag  = display.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
            g_tag = f"{g:.2e}"
            cache = CACHE_DIR / f"{tag}_g{g_tag}.npy"
            try:
                if use_cache and cache.exists():
                    sc = np.load(cache)
                    gu = g
                else:
                    sc, gu = compute_scores_nostd(
                        backbone, layer_idx, batch_size,
                        anchor_paths, dist_paths, device, g
                    )
                    np.save(cache, sc)
                rho, _ = spearmanr(sc, [r["level"] for r in rows])
                rho_d,_ = spearmanr(sc, [r["dmos"] for r in rows])
                logger.info(f"  {display} gamma={g:.2e}  "
                            f"rho(level)={rho:+.4f}  rho(dmos)={rho_d:+.4f}")
                summary_extra.append((display, g, rho, rho_d))
            except Exception as exc:
                logger.error(f"  ERREUR {display} gamma={g:.2e} : {exc}", exc_info=True)

    # ── CMMD ─────────────────────────────────────────────────────────
    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info("\n-- CMMD (CLIP ViT-L/14@336) --")
        try:
            cache = CACHE_DIR / "CMMD.npy"
            if use_cache and cache.exists():
                scores = np.load(cache)
            else:
                scores = compute_scores_cmmd(anchor_paths, dist_paths, device)
                np.save(cache, scores)
            all_stats["CMMD"]    = stats(rows, scores)
            gamma_labels["CMMD"] = f"{CMMD_GAMMA:.3e}"
            st = all_stats["CMMD"]
            logger.info(f"  rho={st['rho']:+.4f}  mono={st['mono_rate']:.1%}")
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)

    if not all_stats:
        logger.error("Aucun resultat.")
        return

    # ── Tableau comparatif ───────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("COMPARAISON — KADID-10k sans standardisation")
    logger.info(f"  {'Metrique':40s}  {'gamma':>12s}  {'rho(level)':>10s}  {'mono%':>7s}")
    logger.info("-" * 80)
    for display, st in all_stats.items():
        g_lbl = gamma_labels.get(display, "?")
        logger.info(f"  {display:40s}  {g_lbl:>12s}  {st['rho']:>+10.4f}  "
                    f"{st['mono_rate']:>6.1%}")
    logger.info("-" * 80)
    if summary_extra:
        logger.info("  Gammas fixes (report supplementaire) :")
        for (disp, g, rho, rho_d) in summary_extra:
            logger.info(f"    {disp:38s} g={g:.2e}  rho(level)={rho:+.4f}  rho(dmos)={rho_d:+.4f}")
    logger.info("=" * 80)

    # ── Figures ──────────────────────────────────────────────────────
    fig_violin_strip(
        all_stats,
        OUT_DIR / f"violin_nostd_{ts}.png",
        gamma_labels,
    )
    fig_scatter_dmos(
        all_stats,
        OUT_DIR / f"scatter_nostd_{ts}.png",
    )

    logger.info(f"\nResultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
