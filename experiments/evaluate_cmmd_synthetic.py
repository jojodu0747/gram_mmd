#!/usr/bin/env python3
"""
CMMD baseline on the synthetic KADID-COCO experiment.

Same setup as sweep_synthetic_kadid.py:
  - 20 COCO reference images (seed=42) → degraded 125 ways (25 types × 5 levels)
  - 1000 COCO anchor images
  - MMD²_u dist-vs-dist, fixed γ = 0.005 (CMMD convention)
  - Backbone: CLIP ViT-L/14@336 (openai/clip-vit-large-patch14-336)

Metrics: Spearman ρ, Kendall τ, strict monotonicity

Output: results/synthetic_kadid_sweep/cmmd_result.csv
        (can be appended to summary_all.csv for joint plots)
"""

import csv
import gc
import logging
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.stats import kendalltau, spearmanr

BASE_DIR  = Path(__file__).parent.parent
COCO_DIR  = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR   = BASE_DIR / "results" / "synthetic_kadid_sweep"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_SEED    = 42
N_ANCHOR       = 1000
N_REFS         = 20
CMMD_GAMMA     = 0.005
CLIP_MODEL_ID  = "openai/clip-vit-large-patch14-336"
BATCH_CLIP     = 32

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KADID DMOS
# ─────────────────────────────────────────────────────────────────────────────

_DISTORTION_SEQUENCE = [
    (dt, lv)
    for lv in range(1, 6)
    for dt in range(1, 26)
]

def build_groups():
    """125 cumulative groups: group k applies the first k+1 distortions in sequence.
    Rank = k+1 (unique, no ties). Each group is strictly worse than the previous.
    """
    return [
        {"operations": _DISTORTION_SEQUENCE[:k + 1], "rank": k + 1}
        for k in range(125)
    ]


def apply_distortion_sequence(img_pil: Image.Image, operations: list) -> Image.Image:
    img = img_pil
    for dt, lv in operations:
        img = apply_distortion(img, dt, lv)
    return img

def get_coco_split():
    all_paths = sorted(list(COCO_DIR.glob("*.jpg")) + list(COCO_DIR.glob("*.png")))
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = rng.permutation(len(all_paths))
    ref_paths    = [all_paths[i] for i in idx[:N_REFS]]
    anchor_paths = [all_paths[i] for i in idx[N_REFS:N_REFS + N_ANCHOR]]
    return ref_paths, anchor_paths

# ─────────────────────────────────────────────────────────────────────────────
# Distortions (identical to sweep_synthetic_kadid.py)
# ─────────────────────────────────────────────────────────────────────────────

def apply_distortion(img_pil: Image.Image, dist_type: int, level: int) -> Image.Image:
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    lv = level - 1

    if dist_type == 1:
        sig = [0.02, 0.06, 0.12, 0.20, 0.30][lv]
        arr = arr + np.random.default_rng().normal(0, sig, arr.shape).astype(np.float32)
    elif dist_type == 2:
        sig = [0.02, 0.06, 0.12, 0.20, 0.30][lv]
        yuv = np.array(img_pil.convert("YCbCr")).astype(np.float32) / 255.0
        yuv[:, :, 0] += np.random.default_rng().normal(0, sig, (H, W)).astype(np.float32)
        yuv = np.clip(yuv, 0, 1)
        arr = np.array(Image.fromarray((yuv * 255).astype(np.uint8), "YCbCr").convert("RGB")).astype(np.float32) / 255.0
    elif dist_type == 3:
        sig = [0.02, 0.06, 0.12, 0.20, 0.30][lv]
        arr = arr * (1 + np.random.default_rng().normal(0, sig, arr.shape).astype(np.float32))
    elif dist_type == 4:
        nsig = [0.04, 0.08, 0.16, 0.24, 0.32][lv]
        bsig = [1.5, 2.0, 3.0, 4.0, 5.0][lv]
        noisy = arr + np.random.default_rng().normal(0, nsig, arr.shape).astype(np.float32)
        arr = ndimage.gaussian_filter(noisy, sigma=[bsig, bsig, 0])
    elif dist_type == 5:
        arr = arr ** [0.80, 0.65, 0.50, 0.35, 0.20][lv]
    elif dist_type == 6:
        arr = arr ** [1.5, 2.0, 3.0, 4.0, 5.0][lv]
    elif dist_type == 7:
        arr = arr + [0.08, 0.16, 0.25, 0.35, 0.45][lv]
    elif dist_type == 8:
        amp = [1, 2, 4, 8, 16][lv]
        rng = np.random.default_rng()
        dy = rng.uniform(-amp, amp, (H, W)).astype(np.float32)
        dx = rng.uniform(-amp, amp, (H, W)).astype(np.float32)
        gy, gx = np.mgrid[0:H, 0:W]
        coords = [np.clip(gy + dy, 0, H - 1), np.clip(gx + dx, 0, W - 1)]
        arr = np.stack([ndimage.map_coordinates(arr[:, :, c], coords, order=1)
                        for c in range(3)], axis=2).astype(np.float32)
    elif dist_type == 9:
        ps_list = [(8, 5), (16, 10), (24, 20), (32, 30), (48, 40)]
        ps, np_ = ps_list[lv]
        rng = np.random.default_rng()
        mu = arr.mean(axis=(0, 1))
        for _ in range(np_):
            y = rng.integers(0, H - ps)
            x = rng.integers(0, W - ps)
            arr[y:y+ps, x:x+ps] = mu
    elif dist_type == 10:
        f = [2, 4, 8, 12, 16][lv]
        small = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).resize(
            (max(1, W // f), max(1, H // f)), Image.NEAREST)
        arr = np.array(small.resize((W, H), Image.NEAREST)).astype(np.float32) / 255.0
    elif dist_type == 11:
        bits = [6, 5, 4, 3, 2][lv]
        levels = 2 ** bits
        arr = np.round(arr * (levels - 1)) / (levels - 1)
    elif dist_type == 12:
        mult = [1.5, 2.0, 3.0, 4.0, 5.0][lv]
        hsv = np.array(Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert("HSV")).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * mult, 0, 255)
        arr = np.array(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")).astype(np.float32) / 255.0
    elif dist_type == 13:
        mult = [0.7, 0.5, 0.3, 0.15, 0.0][lv]
        hsv = np.array(Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert("HSV")).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * mult, 0, 255)
        arr = np.array(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")).astype(np.float32) / 255.0
    elif dist_type == 14:
        density = [0.01, 0.03, 0.07, 0.12, 0.20][lv]
        rng = np.random.default_rng()
        n = int(density * H * W)
        ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
        col = rng.random((n, 3)).astype(np.float32)
        size = rng.integers(1, max(2, int(density * 20)), n)
        for i in range(n):
            s = int(size[i])
            arr[max(0, ys[i]-s):ys[i]+s+1, max(0, xs[i]-s):xs[i]+s+1] = col[i]
    elif dist_type == 15:
        sig = [0.5, 1.0, 2.0, 4.0, 6.0][lv]
        arr = np.stack([ndimage.gaussian_filter(arr[:, :, c], sig) for c in range(3)], axis=2).astype(np.float32)
    elif dist_type == 16:
        from PIL.ImageFilter import SHARPEN
        factor = [1.2, 1.5, 2.0, 3.0, 5.0][lv]
        base = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        sharp = base.filter(SHARPEN)
        blend = Image.blend(base, sharp, factor)
        arr = np.array(blend).astype(np.float32) / 255.0
    elif dist_type == 17:
        q = [75, 50, 30, 15, 5][lv]
        buf = BytesIO()
        Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=q)
        arr = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
    elif dist_type == 18:
        q = [75, 50, 30, 15, 5][lv]
        for _ in range(2):
            buf = BytesIO()
            Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=q)
            arr = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
    elif dist_type == 19:
        scale = [2, 3, 4, 6, 8][lv]
        base = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        down = base.resize((max(1, W // scale), max(1, H // scale)), Image.BICUBIC)
        arr = np.array(down.resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0
    elif dist_type == 20:
        scale = [2, 3, 4, 6, 8][lv]
        base = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        down = base.resize((max(1, W // scale), max(1, H // scale)), Image.NEAREST)
        arr = np.array(down.resize((W, H), Image.NEAREST)).astype(np.float32) / 255.0
    elif dist_type == 21:
        sig = [0.5, 1.0, 2.0, 4.0, 6.0][lv]
        kernel = np.outer(np.ones(int(sig * 3) | 1), np.ones(1))
        kernel = kernel / kernel.sum()
        arr = np.stack([ndimage.convolve(arr[:, :, c], kernel) for c in range(3)], axis=2).astype(np.float32)
    elif dist_type == 22:
        sig = [0.5, 1.0, 2.0, 4.0, 6.0][lv]
        kernel = np.outer(np.ones(1), np.ones(int(sig * 3) | 1))
        kernel = kernel / kernel.sum()
        arr = np.stack([ndimage.convolve(arr[:, :, c], kernel) for c in range(3)], axis=2).astype(np.float32)
    elif dist_type == 23:
        k = [3, 5, 9, 13, 19][lv]
        arr = np.stack([ndimage.uniform_filter(arr[:, :, c], k) for c in range(3)], axis=2).astype(np.float32)
    elif dist_type == 24:
        g = [0.90, 0.75, 0.60, 0.45, 0.30][lv]
        arr = arr * g
    elif dist_type == 25:
        g = [0.20, 0.35, 0.50, 0.65, 0.80][lv]
        alpha = [0.10, 0.15, 0.20, 0.30, 0.50][lv]
        arr = arr * (1 - alpha) + (arr ** g) * alpha

    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

# ─────────────────────────────────────────────────────────────────────────────
# CLIP embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_pil_list(model, processor, pil_list, device):
    embs = []
    for i in range(0, len(pil_list), BATCH_CLIP):
        batch = pil_list[i:i + BATCH_CLIP]
        pv = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
        if (i // BATCH_CLIP + 1) % 10 == 0:
            logger.info(f"    {min(i + BATCH_CLIP, len(pil_list))}/{len(pil_list)}")
    return np.concatenate(embs, axis=0).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# MMD²_u dist-vs-dist
# ─────────────────────────────────────────────────────────────────────────────

def rbf_kernel(D2, gamma):
    return np.exp(-gamma * D2)

def mmd2_unbiased(X, Y, gamma):
    """X: (n, D), Y: (m, D)"""
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    Dxx = torch.cdist(X, X).pow(2)
    Dyy = torch.cdist(Y, Y).pow(2)
    Dxy = torch.cdist(X, Y).pow(2)
    n, m = X.shape[0], Y.shape[0]
    mask_xx = ~torch.eye(n, dtype=torch.bool)
    mask_yy = ~torch.eye(m, dtype=torch.bool)
    Kxx = torch.exp(-gamma * Dxx)[mask_xx].mean().item()
    Kyy = torch.exp(-gamma * Dyy)[mask_yy].mean().item()
    Kxy = torch.exp(-gamma * Dxy).mean().item()
    return Kxx + Kyy - 2 * Kxy

def compute_stats(mmd2_scores, groups):
    """Correlate MMD² with distortion level (1=best, 5=worst).
    Strict mono: mean MMD² per level must be increasing over 5 levels.
    """
    quality_ranks = np.array([g["rank"] for g in groups], dtype=float)
    sp, sp_p = spearmanr(quality_ranks, mmd2_scores)
    kt, kt_p = kendalltau(quality_ranks, mmd2_scores)
    mono = float((np.diff(mmd2_scores) > 0).sum()) / (len(mmd2_scores) - 1)
    return {"spearman": sp, "spearman_p": sp_p, "kendall": kt, "kendall_p": kt_p, "mono_strict": mono}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    cache_anc = CACHE_DIR / "cmmd_anchor.npy"
    cache_dist = CACHE_DIR / "cmmd_dist.npy"

    groups = build_groups()
    ref_paths, anchor_paths = get_coco_split()
    logger.info(f"Groups: {len(groups)}, Refs: {len(ref_paths)}, Anchor: {len(anchor_paths)}")

    # ── Load CLIP ────────────────────────────────────────────────────────────
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    logger.info(f"Loading CLIP: {CLIP_MODEL_ID}")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    # ── Anchor embeddings ────────────────────────────────────────────────────
    if cache_anc.exists():
        logger.info("Loading anchor cache...")
        emb_anc = np.load(cache_anc)
    else:
        logger.info(f"Embedding anchor ({N_ANCHOR} images)...")
        anc_pil = [Image.open(p).convert("RGB") for p in anchor_paths]
        emb_anc = embed_pil_list(model, proc, anc_pil, device)
        np.save(cache_anc, emb_anc)
        logger.info(f"Anchor saved → {cache_anc}  shape: {emb_anc.shape}")

    # ── Dist embeddings (125 groups × 20 degraded images) ────────────────────
    if cache_dist.exists():
        logger.info("Loading dist cache...")
        emb_dist = np.load(cache_dist)  # (125, 20, D)
    else:
        logger.info("Embedding dist groups (125 × 20 images)...")
        ref_pil_base = [Image.open(p).convert("RGB") for p in ref_paths]
        emb_dist_list = []
        for k, grp in enumerate(groups):
            if k % 25 == 0:
                dt_last, lv_last = grp["operations"][-1]
                logger.info(f"  Group {k}/124  +type={dt_last} level={lv_last}  (total ops={len(grp['operations'])})")
            dist_pil = [apply_distortion_sequence(img, grp["operations"]) for img in ref_pil_base]
            emb_k = embed_pil_list(model, proc, dist_pil, device)
            emb_dist_list.append(emb_k)  # (20, D)
        emb_dist = np.stack(emb_dist_list, axis=0)  # (125, 20, D)
        np.save(cache_dist, emb_dist)
        logger.info(f"Dist saved → {cache_dist}  shape: {emb_dist.shape}")

    del model, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Compute MMD² for each group ──────────────────────────────────────────
    logger.info(f"Computing MMD² (γ={CMMD_GAMMA}) for {len(groups)} groups...")
    mmd2_scores = []
    for k in range(len(groups)):
        score = mmd2_unbiased(emb_anc, emb_dist[k], CMMD_GAMMA)
        mmd2_scores.append(score)
        if k % 25 == 0:
            logger.info(f"  Group {k}/124  MMD²={score:.6f}")
    mmd2_scores = np.array(mmd2_scores)

    # ── Statistics ───────────────────────────────────────────────────────────
    stats = compute_stats(mmd2_scores, groups)
    logger.info(
        f"\nCMMD results:\n"
        f"  Spearman ρ  = {stats['spearman']:+.4f}  (p={stats['spearman_p']:.2e})\n"
        f"  Kendall τ   = {stats['kendall']:+.4f}  (p={stats['kendall_p']:.2e})\n"
        f"  Strict mono = {stats['mono_strict']:.1%}"
    )

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_csv = OUT_DIR / "cmmd_result.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "layer", "gamma_mult", "gamma_val",
                                           "spearman", "spearman_p", "kendall", "kendall_p", "mono_strict"])
        w.writeheader()
        w.writerow({
            "backbone":    "CMMD",
            "layer":       "",
            "gamma_mult":  "",
            "gamma_val":   f"{CMMD_GAMMA:.4e}",
            "spearman":    f"{stats['spearman']:.6f}",
            "spearman_p":  f"{stats['spearman_p']:.4e}",
            "kendall":     f"{stats['kendall']:.6f}",
            "kendall_p":   f"{stats['kendall_p']:.4e}",
            "mono_strict": f"{stats['mono_strict']:.6f}",
        })
    logger.info(f"Saved: {out_csv}")
    print(f"\n--- CMMD Synthetic KADID Result ---")
    print(f"Spearman ρ  : {stats['spearman']:+.4f}")
    print(f"Kendall τ   : {stats['kendall']:+.4f}")
    print(f"Strict mono : {stats['mono_strict']:.1%}")

if __name__ == "__main__":
    main()
