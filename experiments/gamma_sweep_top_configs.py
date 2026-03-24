"""
gamma_sweep_top_configs.py
===========================
Sweep gamma sur [γ_med/100 … 100×γ_med] pour les top-10 Spearman + top-10 Kendall
configs du sweep KADID.

Anchor  : 1000 KITTI
Query A : 1000 VKITTI2 variées
Query B : 1000 Stanford Cars

Produit 20 plots + 1 CMMD + JSON résultats.
Coupe la partie basse (zone d'inversion dégénérée Gram-MMD) pour
mieux visualiser la zone utile autour de γ_med.
"""

import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KITTI_DIR    = BASE_DIR / "dataset" / "kitti" / "images"
VKITTI_DIR   = BASE_DIR / "dataset" / "vkitti_2.0 _rgb"
STANFORD_DIR = BASE_DIR / "dataset" / "stanford_cars"
RESULTS_DIR  = BASE_DIR / "experiments" / "results" / "gamma_sweep_top_configs"
SWEEP_CSV    = BASE_DIR / "results" / "synthetic_kadid_sweep" / "summary_full_sweep.csv"

N_ANCHOR   = 1000
N_QUERY    = 1000
SEED       = 42
BATCH_SIZE = 16
CMMD_GAMMA = 0.005

# Sweep : γ_med/1000 à 100×γ_med
N_GAMMA_STEPS = 40
GAMMA_MIN_MUL = 1e-3
GAMMA_MAX_MUL = 1e2


# ── Utilitaires ──────────────────────────────────────────────────

def sample_paths(paths, n, seed):
    rng = np.random.default_rng(seed)
    paths = sorted(paths)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(idx)]


def median_gamma(feats, max_samples=300, max_dims=512):
    rng = np.random.default_rng(0)
    n, d = feats.shape
    idx = rng.choice(n, size=min(n, max_samples), replace=False)
    sample = feats[idx]
    if d > max_dims:
        sample = sample[:, rng.choice(d, size=max_dims, replace=False)]
    sq = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1)
    return float(1.0 / (2 * np.median(sq[sq > 0]) + 1e-8))


def sq_dists(A, B):
    return (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * (A @ B.T)


def mmd2_from_dists(D_xx, D_yy, D_xy, gamma):
    n, m = D_xx.shape[0], D_yy.shape[0]
    Kxx = np.exp(-gamma * D_xx); np.fill_diagonal(Kxx, 0.0)
    Kyy = np.exp(-gamma * D_yy); np.fill_diagonal(Kyy, 0.0)
    Kxy = np.exp(-gamma * D_xy)
    return float(Kxx.sum() / (n * (n - 1)) +
                 Kyy.sum() / (m * (m - 1)) -
                 2 * Kxy.mean())


def get_gram(paths, ext, fit_transform=False):
    return ext.extract([str(p) for p in paths],
                       fit_transform=fit_transform,
                       batch_size=BATCH_SIZE).astype(np.float32)


def get_clip_feats(paths, model, preprocess, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [preprocess(Image.open(str(p)).convert("RGB"))
                 for p in paths[i:i + batch_size]]
        with torch.no_grad():
            f = model.encode_image(torch.stack(batch).to(device)).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


# ── Plot ─────────────────────────────────────────────────────────

def plot_single(gammas_gram, mmd_vk_gram, mmd_sc_gram,
                gammas_clip, mmd_vk_clip, mmd_sc_clip,
                cmmd_fixed_vk, cmmd_fixed_sc,
                g_med_gram, g_med_clip,
                config_label, ranking_label, out_path):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"MMD² sweep — KITTI 5k anchor | VKITTI2 vs Stanford Cars\n"
                 f"{config_label}  ({ranking_label})",
                 fontsize=12, fontweight="bold")

    for ax, gammas, mmd_vk, mmd_sc, g_med, label, extra in [
        (axes[0], gammas_gram, mmd_vk_gram, mmd_sc_gram, g_med_gram,
         f"Gram-MMD ({config_label})", None),
        (axes[1], gammas_clip, mmd_vk_clip, mmd_sc_clip, g_med_clip,
         "CMMD (CLIP ViT-L/14)",
         (CMMD_GAMMA, cmmd_fixed_vk, cmmd_fixed_sc)),
    ]:
        ax.semilogx(gammas, mmd_vk, "b-o", markersize=3, label="VKITTI2 (synthetic)")
        ax.semilogx(gammas, mmd_sc, "r-s", markersize=3, label="Stanford Cars (real)")

        # Zone d'inversion : VKITTI2 < Stanford (synth plus proche que réel)
        inv_mask = mmd_vk < mmd_sc
        if inv_mask.any():
            inv_gs = gammas[inv_mask]
            ax.axvspan(inv_gs.min(), inv_gs.max(), alpha=0.12, color="red",
                       label="zone inversion")

        ax.fill_between(gammas,
                        np.minimum(mmd_vk, mmd_sc),
                        np.maximum(mmd_vk, mmd_sc),
                        alpha=0.08,
                        color="green" if not inv_mask.any() else "orange")

        # Point γ_med
        ax.axvline(g_med, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.annotate(f"γ_med={g_med:.1e}", xy=(g_med, ax.get_ylim()[1] * 0.95),
                    fontsize=7, color="darkgreen", rotation=90, va="top", ha="right")

        # γ/10
        g10 = g_med * 0.1
        if g10 >= gammas.min():
            ax.axvline(g10, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.annotate("γ/10", xy=(g10, 0), fontsize=7, color="gray", va="bottom")

        # CMMD fixe γ=0.005
        if extra is not None:
            g_fixed, vk_f, sc_f = extra
            if gammas.min() <= g_fixed <= gammas.max():
                ax.axvline(g_fixed, color="purple", linestyle=":", linewidth=1.2,
                           label=f"γ=0.005 (CMMD fixe)")
                ax.plot(g_fixed, vk_f, "b*", markersize=10)
                ax.plot(g_fixed, sc_f, "r*", markersize=10)

        ax.set_xlabel("γ  (convention: exp(-γ ||x-y||²))  — log scale")
        ax.set_ylabel("MMD² (unbiased)")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load sweep results ────────────────────────────────────────
    df = pd.read_csv(SWEEP_CSV)
    best_sp = df.loc[df.groupby(["backbone", "layer"])["spearman"].idxmax()]
    top10_sp = best_sp.nlargest(10, "spearman")
    best_kt = df.loc[df.groupby(["backbone", "layer"])["kendall"].idxmax()]
    top10_kt = best_kt.nlargest(10, "kendall")

    logger.info("=== TOP 10 SPEARMAN ===")
    for i, (_, r) in enumerate(top10_sp.iterrows()):
        logger.info(f"  {i+1:2d}. {r.backbone:20s} L{int(r.layer):2d}  ρ={r.spearman:.3f}")
    logger.info("=== TOP 10 KENDALL ===")
    for i, (_, r) in enumerate(top10_kt.iterrows()):
        logger.info(f"  {i+1:2d}. {r.backbone:20s} L{int(r.layer):2d}  τ={r.kendall:.3f}")

    # Collect unique configs
    configs = {}
    for i, (_, r) in enumerate(top10_sp.iterrows()):
        key = (r.backbone, int(r.layer))
        if key not in configs:
            configs[key] = {"rankings": []}
        configs[key]["rankings"].append(f"Spearman #{i+1} (ρ={r.spearman:.3f})")
    for i, (_, r) in enumerate(top10_kt.iterrows()):
        key = (r.backbone, int(r.layer))
        if key not in configs:
            configs[key] = {"rankings": []}
        configs[key]["rankings"].append(f"Kendall #{i+1} (τ={r.kendall:.3f})")

    logger.info(f"\n{len(configs)} unique configs to process")

    # ── Paths ────────────────────────────────────────────────────
    anchor_paths  = sample_paths(list(KITTI_DIR.glob("*.png")), N_ANCHOR, SEED)
    vkitti_all    = list(VKITTI_DIR.rglob("*.jpg"))
    vkitti_paths  = sample_paths(vkitti_all, N_QUERY, SEED + 1)
    stanford_all  = list(STANFORD_DIR.glob("*.jpg")) + list(STANFORD_DIR.glob("*.png"))
    stanford_paths = sample_paths(stanford_all, N_QUERY, SEED + 2)

    logger.info(f"Anchor : {len(anchor_paths)} KITTI")
    logger.info(f"QueryA : {len(vkitti_paths)} VKITTI2")
    logger.info(f"QueryB : {len(stanford_paths)} Stanford Cars")

    # ── CLIP features (une seule fois) ────────────────────────────
    logger.info("\nChargement CLIP...")
    import clip
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()

    logger.info("Extraction CLIP anchor...")
    anc_clip = get_clip_feats(anchor_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP VKITTI2...")
    vk_clip = get_clip_feats(vkitti_paths, clip_model, clip_preprocess, device)
    logger.info("Extraction CLIP Stanford Cars...")
    sc_clip = get_clip_feats(stanford_paths, clip_model, clip_preprocess, device)

    g_med_clip = median_gamma(anc_clip)
    logger.info(f"γ_med CLIP = {g_med_clip:.4e}")

    # CMMD fixe
    D_xx_c, D_yy_vk_c, D_xy_vk_c = sq_dists(anc_clip, anc_clip), sq_dists(vk_clip, vk_clip), sq_dists(anc_clip, vk_clip)
    D_yy_sc_c, D_xy_sc_c = sq_dists(sc_clip, sc_clip), sq_dists(anc_clip, sc_clip)
    cmmd_fixed_vk = mmd2_from_dists(D_xx_c, D_yy_vk_c, D_xy_vk_c, CMMD_GAMMA)
    cmmd_fixed_sc = mmd2_from_dists(D_xx_c, D_yy_sc_c, D_xy_sc_c, CMMD_GAMMA)
    logger.info(f"CMMD fixe γ=0.005 : VKITTI2={cmmd_fixed_vk:.4e}  Stanford={cmmd_fixed_sc:.4e}")

    # Sweep CLIP
    multipliers = np.logspace(np.log10(GAMMA_MIN_MUL), np.log10(GAMMA_MAX_MUL), N_GAMMA_STEPS)
    gammas_clip = g_med_clip * multipliers
    mmd_vk_clip = np.array([mmd2_from_dists(D_xx_c, D_yy_vk_c, D_xy_vk_c, g) for g in gammas_clip])
    mmd_sc_clip = np.array([mmd2_from_dists(D_xx_c, D_yy_sc_c, D_xy_sc_c, g) for g in gammas_clip])

    del anc_clip, vk_clip, sc_clip, D_xx_c, D_yy_vk_c, D_xy_vk_c, D_yy_sc_c, D_xy_sc_c
    del clip_model, clip_preprocess
    gc.collect()
    torch.cuda.empty_cache()

    # ── Process each Gram config ──────────────────────────────────
    transform_cfg = {"name": "gram_spatial", "use_gram": True,
                     "use_pca": False, "gram_patches": False, "gram_averaging": "spatial"}

    all_results = {}

    # Group configs by backbone for efficiency
    from collections import defaultdict
    backbone_groups = defaultdict(list)
    for (backbone, layer) in configs:
        backbone_groups[backbone].append(layer)

    for backbone, layers in backbone_groups.items():
        layers = sorted(layers)
        logger.info(f"\n{'='*60}")
        logger.info(f"Backbone: {backbone}  |  Layers: {layers}")
        logger.info(f"{'='*60}")

        for layer in layers:
            key = (backbone, layer)
            config_label = f"{backbone} L{layer}"
            logger.info(f"\n--- {config_label} ---")

            ext = FeatureExtractor(backbone=backbone, layer=layer,
                                   transform_config=transform_cfg, device=device)

            logger.info(f"  Extraction anchor (fit)...")
            anc_gram = get_gram(anchor_paths, ext, fit_transform=True)
            logger.info(f"  Extraction VKITTI2...")
            vk_gram = get_gram(vkitti_paths, ext)
            logger.info(f"  Extraction Stanford Cars...")
            sc_gram = get_gram(stanford_paths, ext)

            g_med_gram = median_gamma(anc_gram)
            logger.info(f"  γ_med = {g_med_gram:.4e}  dim={anc_gram.shape[1]}")

            # Precompute distances
            D_xx_g = sq_dists(anc_gram, anc_gram)
            D_yy_vk_g = sq_dists(vk_gram, vk_gram)
            D_xy_vk_g = sq_dists(anc_gram, vk_gram)
            D_yy_sc_g = sq_dists(sc_gram, sc_gram)
            D_xy_sc_g = sq_dists(anc_gram, sc_gram)

            del anc_gram, vk_gram, sc_gram; gc.collect()

            # Sweep
            gammas_gram = g_med_gram * multipliers
            mmd_vk_gram = np.array([mmd2_from_dists(D_xx_g, D_yy_vk_g, D_xy_vk_g, g)
                                    for g in gammas_gram])
            mmd_sc_gram = np.array([mmd2_from_dists(D_xx_g, D_yy_sc_g, D_xy_sc_g, g)
                                    for g in gammas_gram])

            del D_xx_g, D_yy_vk_g, D_xy_vk_g, D_yy_sc_g, D_xy_sc_g; gc.collect()

            # Inversion summary
            inv_count = int((mmd_vk_gram < mmd_sc_gram).sum())
            logger.info(f"  Inversion Gram: {inv_count}/{N_GAMMA_STEPS} gammas")

            # Store results
            all_results[key] = {
                "g_med_gram": g_med_gram,
                "gammas_gram": gammas_gram,
                "mmd_vk_gram": mmd_vk_gram,
                "mmd_sc_gram": mmd_sc_gram,
            }

            # Generate plots for each ranking
            for ranking in configs[key]["rankings"]:
                safe_name = ranking.replace(" ", "_").replace("#", "").replace("(", "").replace(")", "").replace("=", "")
                out_path = RESULTS_DIR / f"{safe_name}_{backbone}_L{layer}.png"
                plot_single(gammas_gram, mmd_vk_gram, mmd_sc_gram,
                            gammas_clip, mmd_vk_clip, mmd_sc_clip,
                            cmmd_fixed_vk, cmmd_fixed_sc,
                            g_med_gram, g_med_clip,
                            config_label, ranking, out_path)
                logger.info(f"  Plot saved → {out_path}")

            # Free extractor
            del ext; gc.collect()
            torch.cuda.empty_cache()

    # ── Save JSON ─────────────────────────────────────────────────
    out_json = RESULTS_DIR / f"gamma_sweep_top_configs_{ts}.json"
    json_data = {
        "experiment": "gamma_sweep_top_configs",
        "timestamp": ts,
        "config": {
            "n_anchor": N_ANCHOR, "n_query": N_QUERY,
            "seed": SEED, "cmmd_gamma": CMMD_GAMMA,
            "gamma_min_mul": GAMMA_MIN_MUL, "gamma_max_mul": GAMMA_MAX_MUL,
            "n_gamma_steps": N_GAMMA_STEPS,
        },
        "cmmd_fixed_vkitti": cmmd_fixed_vk,
        "cmmd_fixed_stanford": cmmd_fixed_sc,
        "gamma_med_clip": g_med_clip,
        "clip_sweep": {
            "gammas": gammas_clip.tolist(),
            "mmd_vkitti": mmd_vk_clip.tolist(),
            "mmd_stanford": mmd_sc_clip.tolist(),
        },
        "gram_configs": {},
    }
    for (backbone, layer), data in all_results.items():
        key_str = f"{backbone}_L{layer}"
        json_data["gram_configs"][key_str] = {
            "backbone": backbone,
            "layer": layer,
            "rankings": configs[(backbone, layer)]["rankings"],
            "gamma_med": data["g_med_gram"],
            "gammas": data["gammas_gram"].tolist(),
            "mmd_vkitti": data["mmd_vk_gram"].tolist(),
            "mmd_stanford": data["mmd_sc_gram"].tolist(),
        }
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"\nResults saved → {out_json}")

    # ── Summary ───────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"CMMD fixe γ=0.005: VKITTI2={cmmd_fixed_vk:.4e}  Stanford={cmmd_fixed_sc:.4e}  "
                f"ratio={cmmd_fixed_vk/cmmd_fixed_sc:.3f}  "
                f"{'INV ★' if cmmd_fixed_vk < cmmd_fixed_sc else 'ok'}")
    logger.info(f"{'Config':<30} {'γ_med':>10}  {'Inv(γ/10)':>10}  {'Inv(γ_med)':>10}  {'Total inv':>10}")
    for (backbone, layer), data in sorted(all_results.items()):
        g = data["g_med_gram"]
        gammas = data["gammas_gram"]
        vk = data["mmd_vk_gram"]
        sc = data["mmd_sc_gram"]

        # Find closest gamma to γ/10 and γ_med
        idx_g10 = np.argmin(np.abs(gammas - g * 0.1))
        idx_gmed = np.argmin(np.abs(gammas - g))
        inv_g10 = "INV" if vk[idx_g10] < sc[idx_g10] else "ok"
        inv_gmed = "INV" if vk[idx_gmed] < sc[idx_gmed] else "ok"
        total_inv = int((vk < sc).sum())

        logger.info(f"  {backbone} L{layer:<3d}  {' '*10}  "
                     f"{g:>10.2e}  {inv_g10:>10}  {inv_gmed:>10}  {total_inv:>10}/{N_GAMMA_STEPS}")
    logger.info(f"{'='*70}")
    logger.info(f"Total plots generated: {sum(len(c['rankings']) for c in configs.values())}")


if __name__ == "__main__":
    main()
