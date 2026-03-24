"""
evaluate_kadid_coco_anchor.py
==============================
KADID-10k — ancrage 200 images COCO (≠ références KADID).

3 backbones × 3 gammas heuristiques + CMMD = 10 configs.

  Backbones :
    SD-VAE  L11   batch=16  input=256
    DC-AE   L8    batch= 4  input=512
    DINOv2  L3    batch= 8  input=518

  Gammas (par backbone, médiane sur les 200 anchors) :
    γ/100  —  très petit  (kernel très large)
    γ_med  —  médiane heuristique   1/(2·median(d²))
    10×γ   —  10× médiane

  CMMD : CLIP ViT-L/14@336, γ fixe = 0.005

Sorties dans results/kadid_coco_anchor/ :
  summary_{ts}.csv
  plots/
    01_scatter_spearman_vs_mono.png
    02_tripanel_metrics.png

Usage :
    .venv/bin/python experiments/evaluate_kadid_coco_anchor.py
    .venv/bin/python experiments/evaluate_kadid_coco_anchor.py --skip-cmmd
"""

import argparse
import csv
import gc
import glob as glob_module
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.features import FeatureExtractor, ImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════

KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
COCO_DIR   = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR    = BASE_DIR / "results" / "kadid_coco_anchor"

N_ANCHOR    = 200
ANCHOR_SEED = 42

# (backbone, layer, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",        11, "SD-VAE L11",  16),
    ("dc_ae",          8, "DC-AE L8",     4),
    ("dinov2_vitb14",  3, "DINOv2 L3",    8),
]

GAMMA_MULTIPLIERS = {
    "gamma_div100": 0.01,   # très petit → kernel très large
    "gamma_med":    1.0,    # médiane heuristique
    "10x_gamma":    10.0,   # 10× médiane
}

ENABLE_CMMD     = True
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32

# ── Visuel ─────────────────────────────────────────────────
BACKBONE_COLORS = {
    "sd_vae":         "#2196F3",   # bleu
    "dc_ae":          "#9C27B0",   # violet
    "dinov2_vitb14":  "#FF9800",   # orange
    "clip_vit_large": "#4CAF50",   # vert
}
GAMMA_MARKERS = {
    "gamma_div100": "v",   # ▽ très petit γ
    "gamma_med":    "o",   # ● médiane
    "10x_gamma":    "s",   # ■ 10×
    "fixed":        "D",   # ◆ CMMD
}
GAMMA_SHORT = {
    "gamma_div100": "γ/100",
    "gamma_med":    "γ_med",
    "10x_gamma":    "10×γ",
    "fixed":        "γ=0.005",
}
BACKBONE_SHORT = {
    "sd_vae":         "SD-VAE L11",
    "dc_ae":          "DC-AE L8",
    "dinov2_vitb14":  "DINOv2 L3",
    "clip_vit_large": "CMMD",
}


def short_label(r):
    b = BACKBONE_SHORT.get(r["backbone"], r["backbone"])
    g = GAMMA_SHORT.get(r["gamma_tag"], r["gamma_tag"])
    if r["backbone"] == "clip_vit_large":
        return b
    return f"{b}  {g}"


def get_color(r):
    return BACKBONE_COLORS.get(r["backbone"], "#888888")


def get_marker(r):
    return GAMMA_MARKERS.get(r["gamma_tag"], "o")


# ════════════════════════════════════════════════════════════
#  CHARGEMENT KADID + COCO ANCHOR
# ════════════════════════════════════════════════════════════

def load_kadid_csv():
    rows = []
    with open(DMOS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dist_img = r["dist_img"]
            parts = dist_img.replace(".png", "").split("_")
            rows.append({
                "dist_img":  dist_img,
                "ref_img":   r["ref_img"],
                "dmos":      float(r["dmos"]),
                "var":       float(r["var"]),
                "ref_id":    int(parts[0][1:]),
                "dist_type": int(parts[1]),
                "level":     int(parts[2]),
            })
    logger.info(f"KADID CSV : {len(rows)} images dégradées")
    return rows


def load_coco_anchor() -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    if not all_imgs:
        raise FileNotFoundError(f"Pas d'images dans {COCO_DIR}")
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = sorted(rng.choice(len(all_imgs), size=N_ANCHOR, replace=False).tolist())
    paths = [all_imgs[i] for i in idx]
    logger.info(f"Ancrage COCO : {len(paths)} images (seed={ANCHOR_SEED})")
    return paths


# ════════════════════════════════════════════════════════════
#  EXTRACTION + SCORING (single-pass, 3 gammas simultanés)
# ════════════════════════════════════════════════════════════

def extract_anchor(backbone, layer, batch_size, anchor_paths, device):
    ext = FeatureExtractor(
        backbone=backbone, layer=layer,
        transform_config={
            "name": "gram_spatial", "use_gram": True,
            "use_pca": False, "gram_patches": False, "gram_averaging": "spatial",
        },
        device=device,
    )
    t0 = time.time()
    ref = ext.extract(anchor_paths, fit_transform=True,
                      batch_size=batch_size).astype(np.float32)
    logger.info(f"  Ancrage COCO ({N_ANCHOR} imgs) : {ref.shape}  ({time.time()-t0:.1f}s)")
    return ext, ref


def compute_gammas(ref: np.ndarray) -> dict:
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    D = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
    triu = D[np.triu_indices(len(ref), k=1)]
    median_d2 = float(np.median(triu))
    g_med = 1.0 / (2.0 * median_d2 + 1e-8)
    logger.info(f"  median(d²) = {median_d2:.4e}  →  γ_med = {g_med:.4e}")
    gammas = {tag: mult * g_med for tag, mult in GAMMA_MULTIPLIERS.items()}
    for tag, g in gammas.items():
        logger.info(f"    {tag:<15s}: γ = {g:.4e}")
    return gammas


def score_all_gammas_single_pass(ext, ref, gammas: dict, dist_paths, device, batch_size):
    """Une seule passe sur les ~10 000 images KADID, 3 kernels simultanés."""
    n = len(dist_paths)
    ref_sq = np.einsum("ij,ij->i", ref, ref)

    mean_Krr  = {}
    cross_sum = {}
    for tag, g in gammas.items():
        D_rr = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
        K_rr = np.exp(-g * D_rr)
        mean_Krr[tag]  = float(np.mean(K_rr))
        cross_sum[tag] = np.zeros(n, dtype=np.float64)
        logger.info(f"  [{tag:<15s}] mean_Krr = {mean_Krr[tag]:.6f}")

    dataset = ImageDataset(dist_paths, transform=ext.image_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"),
                        persistent_workers=True)

    t0 = time.time()
    idx = 0
    for batch_imgs, _ in loader:
        feats = ext.extract_batch(batch_imgs)
        feats = ext.transform_features(feats, fit_pca=False).astype(np.float32)
        B = feats.shape[0]

        t_sq = np.einsum("ij,ij->i", feats, feats)
        D_rt = ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ feats.T)  # (n_ref, B)

        for tag, g in gammas.items():
            K_rt = np.exp(-g * D_rt)
            cross_sum[tag][idx:idx + B] = np.sum(K_rt, axis=0)

        idx += B
        if idx % 2000 < batch_size or idx >= n:
            logger.info(f"    {idx}/{n} ({idx/n:.0%})  {time.time()-t0:.0f}s")

        del feats, D_rt
        gc.collect()

    logger.info(f"  Passe unique : {time.time()-t0:.1f}s")

    n_ref = len(ref)
    return {tag: mean_Krr[tag] + 1.0 - 2.0 / n_ref * cross_sum[tag]
            for tag in gammas}


# ════════════════════════════════════════════════════════════
#  CMMD — CLIP ViT-L/14@336
# ════════════════════════════════════════════════════════════

def run_cmmd(anchor_paths, dist_paths, device):
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    logger.info(f"  Chargement {CLIP_MODEL_ID}…")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def embed_list(paths):
        embs = []
        for i in range(0, len(paths), BATCH_SIZE_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_SIZE_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32)

    logger.info(f"  Embeddings anchor ({len(anchor_paths)} imgs)…")
    ref = embed_list(anchor_paths)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    K_rr = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)))
    mean_Krr = float(np.mean(K_rr))
    logger.info(f"  Ancrage CLIP : {ref.shape}  mean_Krr={mean_Krr:.6f}")

    n = len(dist_paths)
    cross_sum = np.zeros(n, dtype=np.float64)
    chunk = BATCH_SIZE_CLIP * 4
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        emb = embed_list(dist_paths[s:e])
        t_sq = np.einsum("ij,ij->i", emb, emb)
        K_rt = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ emb.T)))
        cross_sum[s:e] = np.sum(K_rt, axis=0)
        if s % 2000 < chunk or e == n:
            logger.info(f"    CLIP : {e}/{n} ({e/n:.0%})  {time.time()-t0:.0f}s")
        del emb, K_rt

    scores = mean_Krr + 1.0 - 2.0 / len(ref) * cross_sum
    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    return scores


# ════════════════════════════════════════════════════════════
#  STATISTIQUES
# ════════════════════════════════════════════════════════════

def compute_stats(rows, scores):
    degradation = np.array([5.0 - r["dmos"] for r in rows])
    sp,  sp_p  = spearmanr(scores, degradation)
    kt,  kt_p  = kendalltau(scores, degradation)
    sp_lv, _   = spearmanr(scores, [r["level"] for r in rows])

    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        groups[(row["ref_id"], row["dist_type"])][row["level"]] = scores[i]

    n_total = n_mono = 0
    for lvl_scores in groups.values():
        if len(lvl_scores) < 5:
            continue
        seq = [lvl_scores[lv] for lv in sorted(lvl_scores)]
        n_total += 1
        if all(seq[i] < seq[i + 1] for i in range(4)):
            n_mono += 1

    by_level = defaultdict(list)
    for r, s in zip(rows, scores):
        by_level[r["level"]].append(s)
    by_level = {lv: float(np.mean(v)) for lv, v in by_level.items()}

    return {
        "spearman":    float(sp),
        "spearman_p":  float(sp_p),
        "kendall":     float(kt),
        "kendall_p":   float(kt_p),
        "spearman_lv": float(sp_lv),
        "mono_strict": n_mono / n_total if n_total else 0.0,
        "n_total":     n_total,
        "by_level":    by_level,
    }


# ════════════════════════════════════════════════════════════
#  SAUVEGARDE CSV
# ════════════════════════════════════════════════════════════

def save_summary(all_results, ts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"summary_{ts}.csv"
    fields = ["key", "backbone", "layer", "gamma_tag", "gamma_val", "label",
              "spearman", "spearman_p", "kendall", "kendall_p", "spearman_lv",
              "mono_strict", "n_total",
              "score_lv1", "score_lv2", "score_lv3", "score_lv4", "score_lv5"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in all_results:
            st = e["stats"]
            lv = st["by_level"]
            w.writerow({
                "key": e["key"], "backbone": e["backbone"],
                "layer": e["layer"], "gamma_tag": e["gamma_tag"],
                "gamma_val": f"{e['gamma_val']:.4e}", "label": e["label"],
                "spearman":   f"{st['spearman']:.6f}",
                "spearman_p": f"{st['spearman_p']:.4e}",
                "kendall":    f"{st['kendall']:.6f}",
                "kendall_p":  f"{st['kendall_p']:.4e}",
                "spearman_lv": f"{st['spearman_lv']:.6f}",
                "mono_strict": f"{st['mono_strict']:.6f}",
                "n_total": st["n_total"],
                **{f"score_lv{lv_k}": f"{lv[lv_k]:.6e}" if lv_k in lv else ""
                   for lv_k in [1, 2, 3, 4, 5]},
            })
    logger.info(f"CSV résumé : {path}")
    return path


# ════════════════════════════════════════════════════════════
#  FIGURE 1 — Scatter Spearman vs Mono strict
# ════════════════════════════════════════════════════════════

def plot_scatter(all_results, plots_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#f9f9f9")
    ax.set_title(
        "KADID-10k — Spearman ρ vs Strict monotonicity rate\n"
        f"3 backbones × 3 gammas + CMMD  ·  anchor = {N_ANCHOR} COCO images",
        fontsize=12, fontweight="bold",
    )

    for r in all_results:
        st = r["stats"]
        ax.scatter(
            st["spearman"], st["mono_strict"],
            color=get_color(r), marker=get_marker(r),
            s=130, zorder=5, edgecolors="black", linewidths=0.8,
        )
        ax.annotate(
            short_label(r),
            (st["spearman"], st["mono_strict"]),
            fontsize=7.5, xytext=(7, 4), textcoords="offset points",
            color=get_color(r),
        )

    ax.set_xlabel("Spearman ρ(score, 5−dmos)  [↑ better]", fontsize=11)
    ax.set_ylabel("Strict monotonicity rate  [↑ better]", fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    backbone_leg = [
        mpatches.Patch(color=BACKBONE_COLORS["sd_vae"],         label="SD-VAE L11"),
        mpatches.Patch(color=BACKBONE_COLORS["dc_ae"],          label="DC-AE L8"),
        mpatches.Patch(color=BACKBONE_COLORS["dinov2_vitb14"],  label="DINOv2 L3"),
        mpatches.Patch(color=BACKBONE_COLORS["clip_vit_large"], label="CMMD"),
    ]
    gamma_leg = [
        Line2D([0],[0], marker="v", color="gray", linestyle="None",
               markersize=8, markeredgecolor="black", label="γ/100"),
        Line2D([0],[0], marker="o", color="gray", linestyle="None",
               markersize=8, markeredgecolor="black", label="γ_median"),
        Line2D([0],[0], marker="s", color="gray", linestyle="None",
               markersize=8, markeredgecolor="black", label="10 × γ_median"),
        Line2D([0],[0], marker="D", color=BACKBONE_COLORS["clip_vit_large"],
               linestyle="None", markersize=8, markeredgecolor="black",
               label="CMMD (γ fixe)"),
    ]
    l1 = ax.legend(handles=backbone_leg, fontsize=9, loc="lower left",
                   title="Backbone", title_fontsize=9)
    ax.add_artist(l1)
    ax.legend(handles=gamma_leg, fontsize=9, loc="upper left",
              title="Gamma", title_fontsize=9)

    plt.tight_layout()
    out = plots_dir / "01_scatter_spearman_vs_mono.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════
#  FIGURE 2 — 3-panel : Spearman / Mono strict / Kendall τ
# ════════════════════════════════════════════════════════════

def plot_tripanel(all_results, plots_dir):
    # Tri par Spearman ρ décroissant — ordre commun aux 3 panneaux
    sorted_r = sorted(all_results, key=lambda e: -e["stats"]["spearman"])
    labels     = [short_label(r) for r in sorted_r]
    colors     = [get_color(r) for r in sorted_r]
    spearman   = [r["stats"]["spearman"]   for r in sorted_r]
    mono       = [r["stats"]["mono_strict"] for r in sorted_r]
    kendall    = [r["stats"]["kendall"]     for r in sorted_r]

    x = np.arange(len(labels))
    w = 0.72

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        f"KADID-10k — 3 backbones × 3 gammas + CMMD\n"
        f"anchor = {N_ANCHOR} COCO images  ·  sorted by Spearman ρ",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1 : Spearman ρ ─────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor("#f9f9f9")
    bars = ax0.bar(x, spearman, color=colors, edgecolor="white",
                   linewidth=0.5, zorder=3, width=w)
    ax0.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    for bar, v in zip(bars, spearman):
        txt_c = "white" if abs(v) > 0.35 else "black"
        ax0.text(bar.get_x() + w / 2, bar.get_height() / 2, f"{v:+.3f}",
                 ha="center", va="center", fontsize=8.5, color=txt_c, fontweight="bold")
    ax0.set_ylabel("Spearman ρ  ↑", fontsize=10)
    ax0.set_ylim(-0.15, 1.0)
    ax0.yaxis.grid(True, alpha=0.3)
    ax0.set_axisbelow(True)

    # ── Panel 2 : Strict monotonicity rate ────────────────────
    ax1 = axes[1]
    ax1.set_facecolor("#f9f9f9")
    bars1 = ax1.bar(x, mono, color=colors, edgecolor="white",
                    linewidth=0.5, zorder=3, width=w)
    for bar, v in zip(bars1, mono):
        txt_c = "white" if v > 0.45 else "black"
        ax1.text(bar.get_x() + w / 2, max(bar.get_height() / 2, 0.01), f"{v:.1%}",
                 ha="center", va="center", fontsize=8.5, color=txt_c, fontweight="bold")
    ax1.set_ylabel("Strict monotonicity rate  ↑", fontsize=10)
    ax1.set_ylim(0, 1.08)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # ── Panel 3 : Kendall τ ────────────────────────────────────
    ax2 = axes[2]
    ax2.set_facecolor("#f9f9f9")
    bars2 = ax2.bar(x, kendall, color=colors, edgecolor="white",
                    linewidth=0.5, zorder=3, width=w)
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    for bar, v in zip(bars2, kendall):
        txt_c = "white" if abs(v) > 0.25 else "black"
        ax2.text(bar.get_x() + w / 2, bar.get_height() / 2, f"{v:+.3f}",
                 ha="center", va="center", fontsize=8.5, color=txt_c, fontweight="bold")
    ax2.set_ylabel("Kendall τ  ↑", fontsize=10)
    ax2.set_ylim(-0.15, 0.85)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # X ticks
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    # Légende backbone (sur le panel 1)
    backbone_leg = [
        mpatches.Patch(color=BACKBONE_COLORS["sd_vae"],         label="SD-VAE L11"),
        mpatches.Patch(color=BACKBONE_COLORS["dc_ae"],          label="DC-AE L8"),
        mpatches.Patch(color=BACKBONE_COLORS["dinov2_vitb14"],  label="DINOv2 L3"),
        mpatches.Patch(color=BACKBONE_COLORS["clip_vit_large"], label="CMMD"),
    ]
    axes[0].legend(handles=backbone_leg, fontsize=9, loc="upper right")

    plt.tight_layout()
    out = plots_dir / "02_tripanel_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════
#  RÉSUMÉ TERMINAL
# ════════════════════════════════════════════════════════════

def print_summary(all_results):
    print("\n" + "=" * 90)
    print("KADID-10k — anchor = 200 COCO images")
    print(f"  {'Label':<50}  {'Spearman':>9}  {'Kendall':>8}  {'Mono.str':>9}")
    print("  " + "-" * 76)
    for e in sorted(all_results, key=lambda x: -x["stats"]["spearman"]):
        st = e["stats"]
        print(
            f"  {e['label']:<50}  {st['spearman']:>+9.4f}  "
            f"{st['kendall']:>+8.4f}  {st['mono_strict']:>9.1%}"
        )
    print("=" * 90)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-cmmd", action="store_true")
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    t_total = time.time()
    logger.info("=" * 90)
    logger.info("KADID-10k — anchor COCO 200 imgs  ·  3 backbones × 3 gammas + CMMD")
    logger.info(f"  Device : {args.device}")
    logger.info("=" * 90)

    rows = load_kadid_csv()
    dist_paths = [str(IMAGES_DIR / r["dist_img"]) for r in rows]
    logger.info(f"  {len(dist_paths)} images dégradées KADID")

    anchor_paths = load_coco_anchor()

    all_results = []

    # ── Backbones Gram-MMD ───────────────────────────────────
    for backbone, layer, display, batch_size in BACKBONE_CONFIGS:
        logger.info(f"\n{'='*90}")
        logger.info(f"{display}  —  ancrage COCO {N_ANCHOR} imgs  ·  3 gammas (single-pass)")
        logger.info("=" * 90)

        try:
            ext, ref = extract_anchor(backbone, layer, batch_size, anchor_paths, args.device)
            gammas   = compute_gammas(ref)

            all_scores = score_all_gammas_single_pass(
                ext, ref, gammas, dist_paths, args.device, batch_size
            )

            ext.cleanup()
            del ext, ref
            torch.cuda.empty_cache()
            gc.collect()

            for gamma_tag, scores in all_scores.items():
                gamma_val = gammas[gamma_tag]
                stats = compute_stats(rows, scores)
                key   = f"{backbone}_l{layer}_{gamma_tag}"
                label = f"{display}  {GAMMA_SHORT[gamma_tag]}  γ={gamma_val:.2e}"
                all_results.append({
                    "key": key, "backbone": backbone, "layer": layer,
                    "gamma_tag": gamma_tag, "gamma_val": gamma_val,
                    "label": label, "stats": stats,
                })
                logger.info(
                    f"  [{gamma_tag:<15s}] Spearman={stats['spearman']:+.4f}  "
                    f"Kendall={stats['kendall']:+.4f}  Mono.strict={stats['mono_strict']:.1%}"
                )

        except Exception as exc:
            logger.error(f"  ERREUR {display} : {exc}", exc_info=True)
            torch.cuda.empty_cache()
            gc.collect()

    # ── CMMD ────────────────────────────────────────────────
    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info(f"\n{'='*90}")
        logger.info(f"CMMD — {CLIP_MODEL_ID}  γ={CMMD_GAMMA}")
        logger.info("=" * 90)
        try:
            scores_clip = run_cmmd(anchor_paths, dist_paths, args.device)
            stats = compute_stats(rows, scores_clip)
            all_results.append({
                "key": "CMMD", "backbone": "clip_vit_large", "layer": 0,
                "gamma_tag": "fixed", "gamma_val": CMMD_GAMMA,
                "label": f"CMMD CLIP-L/14  γ={CMMD_GAMMA:.3e}",
                "stats": stats,
            })
            logger.info(
                f"  Spearman={stats['spearman']:+.4f}  Kendall={stats['kendall']:+.4f}  "
                f"Mono.strict={stats['mono_strict']:.1%}"
            )
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)

    logger.info(f"\nTemps total : {(time.time()-t_total)/60:.1f} min")

    if not all_results:
        logger.error("Aucun résultat — arrêt.")
        return

    # ── Sauvegarde + figures ─────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    save_summary(all_results, ts)
    print_summary(all_results)

    logger.info("\nFigures…")
    plot_scatter(all_results, plots_dir)
    plot_tripanel(all_results, plots_dir)

    logger.info(f"\nRésultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
