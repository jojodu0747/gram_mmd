#!/usr/bin/env python3
"""
Synthetic KADID degradation on 20 COCO refs + 1000 COCO anchor.
Backbone: SD-VAE layer 11. γ = median heuristic.

For each of the 125 distortion combinations (25 types × 5 levels), we:
  1. Parse ground-truth quality ordering from KADID DMOS (mean DMOS across refs).
  2. Apply each distortion to 20 COCO "reference" images on-the-fly.
  3. Extract SD-VAE L11 features, cache to disk (mmap).
  4. Compute MMD² between each distorted group (20 images) and 1000 anchor images.
  5. Report Spearman ρ, Kendall τ, and strict monotonicity vs. ground-truth rank.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.stats import kendalltau, spearmanr

# Make sure the src/ tree is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gram_mmd.features import FeatureExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
COCO_DIR = BASE_DIR / "dataset/coco_5000"
KADID_DMOS_CSV = BASE_DIR / "dataset/kadid10k/dmos.csv"
RESULTS_DIR = BASE_DIR / "results/synthetic_kadid_coco"
CACHE_DIR = BASE_DIR / "experiments/cache"

N_REFS = 20          # COCO images to degrade
N_ANCHOR = 1000      # COCO images used as anchor
N_DIST_TYPES = 25
N_LEVELS = 5
N_GROUPS = N_DIST_TYPES * N_LEVELS  # 125

GAMMA_SUBSAMPLE = 500  # anchor subsamples for median heuristic


# ===========================================================================
# 1. DISTORTION IMPLEMENTATIONS
# ===========================================================================

def apply_distortion(img_pil: Image.Image, dist_type: int, level: int) -> Image.Image:
    """
    Apply one of 25 KADID-inspired distortions to a PIL image.

    Args:
        img_pil:   RGB PIL image.
        dist_type: Integer in 1..25.
        level:     Integer in 1..5 (1 = mildest, 5 = strongest).

    Returns:
        Distorted RGB PIL image.
    """
    assert 1 <= dist_type <= 25, f"dist_type must be 1..25, got {dist_type}"
    assert 1 <= level <= 5,      f"level must be 1..5, got {level}"

    img = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
    li = level - 1  # 0-based index

    rng = np.random.default_rng(seed=dist_type * 100 + level)

    # -----------------------------------------------------------------------
    # Type 1 – Gaussian noise (all channels)
    # -----------------------------------------------------------------------
    if dist_type == 1:
        sigmas = [0.02, 0.06, 0.12, 0.20, 0.30]
        img = img + rng.normal(0, sigmas[li], img.shape).astype(np.float32)

    # -----------------------------------------------------------------------
    # Type 2 – Luminance noise (Y channel in YCbCr)
    # -----------------------------------------------------------------------
    elif dist_type == 2:
        sigmas = [0.02, 0.06, 0.12, 0.20, 0.30]
        # Convert to YCbCr float
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        ycbcr = np.array(Image.fromarray(img_uint8).convert("YCbCr")).astype(np.float32) / 255.0
        ycbcr[..., 0] += rng.normal(0, sigmas[li], ycbcr[..., 0].shape).astype(np.float32)
        ycbcr = ycbcr.clip(0, 1)
        ycbcr_uint8 = (ycbcr * 255).clip(0, 255).astype(np.uint8)
        img = np.array(Image.fromarray(ycbcr_uint8, mode="YCbCr").convert("RGB")).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 3 – Multiplicative noise
    # -----------------------------------------------------------------------
    elif dist_type == 3:
        sigmas = [0.02, 0.06, 0.12, 0.20, 0.30]
        noise = rng.normal(0, sigmas[li], img.shape).astype(np.float32)
        img = img * (1.0 + noise)

    # -----------------------------------------------------------------------
    # Type 4 – Denoise simulation (add noise then Gaussian blur)
    # -----------------------------------------------------------------------
    elif dist_type == 4:
        noise_sigmas = [0.04, 0.08, 0.16, 0.24, 0.32]
        blur_sigmas  = [1.5,  2.0,  3.0,  4.0,  5.0]
        img = img + rng.normal(0, noise_sigmas[li], img.shape).astype(np.float32)
        img = ndimage.gaussian_filter(img, sigma=[blur_sigmas[li], blur_sigmas[li], 0])

    # -----------------------------------------------------------------------
    # Type 5 – Brighten (gamma < 1)
    # -----------------------------------------------------------------------
    elif dist_type == 5:
        gammas = [0.80, 0.65, 0.50, 0.35, 0.20]
        img = img.clip(0, 1) ** gammas[li]

    # -----------------------------------------------------------------------
    # Type 6 – Darken (gamma > 1)
    # -----------------------------------------------------------------------
    elif dist_type == 6:
        gammas = [1.5, 2.0, 3.0, 4.0, 5.0]
        img = img.clip(0, 1) ** gammas[li]

    # -----------------------------------------------------------------------
    # Type 7 – Mean shift (add constant to all channels)
    # -----------------------------------------------------------------------
    elif dist_type == 7:
        shifts = [0.08, 0.16, 0.25, 0.35, 0.45]
        img = img + shifts[li]

    # -----------------------------------------------------------------------
    # Type 8 – Jitter (random pixel displacement via map_coordinates)
    # -----------------------------------------------------------------------
    elif dist_type == 8:
        amplitudes = [1, 2, 4, 8, 16]
        H, W, C = img.shape
        amp = amplitudes[li]
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        dr = rng.uniform(-amp, amp, (H, W)).astype(np.float32)
        dc = rng.uniform(-amp, amp, (H, W)).astype(np.float32)
        new_rows = (rows + dr).clip(0, H - 1)
        new_cols = (cols + dc).clip(0, W - 1)
        out = np.stack([
            ndimage.map_coordinates(img[..., c], [new_rows, new_cols], order=1, mode="reflect")
            for c in range(C)
        ], axis=-1).astype(np.float32)
        img = out

    # -----------------------------------------------------------------------
    # Type 9 – Non-eccentricity patch (mean-color blotches)
    # -----------------------------------------------------------------------
    elif dist_type == 9:
        params = [(8, 5), (16, 10), (24, 20), (32, 30), (48, 40)]
        patch_size, n_patches = params[li]
        H, W, _ = img.shape
        mean_color = img.mean(axis=(0, 1))
        for _ in range(n_patches):
            r = rng.integers(0, max(1, H - patch_size))
            c = rng.integers(0, max(1, W - patch_size))
            img[r:r + patch_size, c:c + patch_size] = mean_color

    # -----------------------------------------------------------------------
    # Type 10 – Pixelate (nearest-neighbor down/up scale)
    # -----------------------------------------------------------------------
    elif dist_type == 10:
        factors = [2, 4, 8, 12, 16]
        f = factors[li]
        H, W, _ = img.shape
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        small = Image.fromarray(img_uint8).resize(
            (max(1, W // f), max(1, H // f)), Image.NEAREST
        )
        img = np.array(small.resize((W, H), Image.NEAREST)).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 11 – Quantization (reduce to N bits per channel)
    # -----------------------------------------------------------------------
    elif dist_type == 11:
        bits = [6, 5, 4, 3, 2]
        n = bits[li]
        levels_q = 2 ** n
        img = np.floor(img * levels_q) / levels_q

    # -----------------------------------------------------------------------
    # Type 12 – Color saturation increase (HSV)
    # -----------------------------------------------------------------------
    elif dist_type == 12:
        sat_mults = [1.5, 2.0, 3.0, 4.0, 5.0]
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        hsv = np.array(Image.fromarray(img_uint8).convert("HSV")).astype(np.float32)
        hsv[..., 1] = (hsv[..., 1] * sat_mults[li]).clip(0, 255)
        img = np.array(
            Image.fromarray(hsv.clip(0, 255).astype(np.uint8), mode="HSV").convert("RGB")
        ).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 13 – Color saturation decrease (HSV)
    # -----------------------------------------------------------------------
    elif dist_type == 13:
        sat_mults = [0.7, 0.5, 0.3, 0.15, 0.0]
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        hsv = np.array(Image.fromarray(img_uint8).convert("HSV")).astype(np.float32)
        hsv[..., 1] = (hsv[..., 1] * sat_mults[li]).clip(0, 255)
        img = np.array(
            Image.fromarray(hsv.clip(0, 255).astype(np.uint8), mode="HSV").convert("RGB")
        ).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 14 – Spatter (random dark blobs)
    # -----------------------------------------------------------------------
    elif dist_type == 14:
        params = [(10, 3), (20, 5), (40, 7), (60, 10), (100, 14)]
        n_spots, radius = params[li]
        H, W, _ = img.shape
        for _ in range(n_spots):
            cy = rng.integers(0, H)
            cx = rng.integers(0, W)
            y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            blob = img[y0:y1, x0:x1]
            blob[mask] *= rng.uniform(0.0, 0.3)

    # -----------------------------------------------------------------------
    # Type 15 – Chromatic aberration (shift R right, B left)
    # -----------------------------------------------------------------------
    elif dist_type == 15:
        shifts_px = [1, 2, 4, 6, 10]
        s = shifts_px[li]
        r_ch = np.roll(img[..., 0], s, axis=1)
        b_ch = np.roll(img[..., 2], -s, axis=1)
        img = np.stack([r_ch, img[..., 1], b_ch], axis=-1)

    # -----------------------------------------------------------------------
    # Type 16 – Sparse sampling (zero out fraction then bilinear inpaint)
    # -----------------------------------------------------------------------
    elif dist_type == 16:
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8]
        frac = fractions[li]
        H, W, C = img.shape
        mask = rng.random((H, W)) < frac  # True = pixel removed
        out = img.copy()
        for c in range(C):
            ch = out[..., c].copy()
            ch[mask] = 0.0
            # Use Gaussian blur as inpainting proxy
            weight = (~mask).astype(np.float32)
            blurred_ch = ndimage.gaussian_filter(ch, sigma=3)
            blurred_w  = ndimage.gaussian_filter(weight, sigma=3)
            filled = np.where(blurred_w > 1e-6, blurred_ch / blurred_w.clip(1e-6), 0.5)
            ch[mask] = filled[mask]
            out[..., c] = ch
        img = out

    # -----------------------------------------------------------------------
    # Type 17 – JPEG compression
    # -----------------------------------------------------------------------
    elif dist_type == 17:
        qualities = [80, 60, 40, 25, 10]
        q = qualities[li]
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(img_uint8).save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 18 – Gaussian blur
    # -----------------------------------------------------------------------
    elif dist_type == 18:
        sigmas = [0.5, 1.0, 2.0, 3.5, 5.0]
        img = ndimage.gaussian_filter(img, sigma=[sigmas[li], sigmas[li], 0])

    # -----------------------------------------------------------------------
    # Type 19 – Lens blur (disk / pillbox kernel)
    # -----------------------------------------------------------------------
    elif dist_type == 19:
        radii = [1, 2, 3, 5, 8]
        r = radii[li]
        d = 2 * r + 1
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        kernel = ((yy ** 2 + xx ** 2) <= r ** 2).astype(np.float32)
        kernel /= kernel.sum()
        out = np.stack([
            ndimage.convolve(img[..., c], kernel, mode="reflect")
            for c in range(3)
        ], axis=-1)
        img = out.astype(np.float32)

    # -----------------------------------------------------------------------
    # Type 20 – Motion blur (horizontal kernel)
    # -----------------------------------------------------------------------
    elif dist_type == 20:
        lengths = [3, 7, 15, 25, 35]
        L = lengths[li]
        kernel = np.ones((1, L), dtype=np.float32) / L
        out = np.stack([
            ndimage.convolve(img[..., c], kernel, mode="reflect")
            for c in range(3)
        ], axis=-1)
        img = out.astype(np.float32)

    # -----------------------------------------------------------------------
    # Type 21 – Diffuse (repeated box filter)
    # -----------------------------------------------------------------------
    elif dist_type == 21:
        params = [(3, 2), (5, 3), (7, 4), (9, 6), (11, 8)]
        ksize, repeats = params[li]
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
        out = img.copy()
        for _ in range(repeats):
            out = np.stack([
                ndimage.convolve(out[..., c], kernel, mode="reflect")
                for c in range(3)
            ], axis=-1)
        img = out.astype(np.float32)

    # -----------------------------------------------------------------------
    # Type 22 – Tilt-stretch (horizontal affine stretch)
    # -----------------------------------------------------------------------
    elif dist_type == 22:
        stretch_xs = [0.85, 0.75, 0.65, 0.55, 0.45]
        sx = stretch_xs[li]
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8)
        W, H = pil.size
        new_W = max(1, int(W * sx))
        stretched = pil.resize((new_W, H), Image.BILINEAR)
        # Paste into original-size canvas (center crop / pad)
        canvas = Image.new("RGB", (W, H), (128, 128, 128))
        offset = (W - new_W) // 2
        canvas.paste(stretched, (offset, 0))
        img = np.array(canvas).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 23 – Affine transform (rotate + shear)
    # -----------------------------------------------------------------------
    elif dist_type == 23:
        params = [(3, 0.02), (6, 0.05), (10, 0.10), (15, 0.18), (20, 0.25)]
        angle_deg, shear = params[li]
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8)
        W, H = pil.size
        # PIL affine data: (a, b, c, d, e, f) where x' = ax + by + c, y' = dx + ey + f
        import math
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        # Combined rotate + shear
        a = cos_a + shear * sin_a
        b = -sin_a + shear * cos_a
        c = (1 - cos_a) * W / 2 + sin_a * H / 2
        d = sin_a
        e = cos_a
        f = -sin_a * W / 2 + (1 - cos_a) * H / 2
        distorted = pil.transform(
            (W, H), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR
        )
        img = np.array(distorted).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 24 – Perspective (trapezoid distortion)
    # -----------------------------------------------------------------------
    elif dist_type == 24:
        strengths = [0.02, 0.05, 0.08, 0.12, 0.18]
        s = strengths[li]
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8)
        W, H = pil.size
        dx = int(s * W)
        # Source quad (original corners)
        src = [
            (0,   0),
            (W-1, 0),
            (W-1, H-1),
            (0,   H-1),
        ]
        # Destination quad (slightly narrower at top = perspective effect)
        dst = [
            (dx,    0),
            (W-1-dx, 0),
            (W-1,   H-1),
            (0,     H-1),
        ]
        # PIL perspective_coeffs: 8 coefficients from src→dst mapping
        coeffs = _find_coeffs(dst, src)
        distorted = pil.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR)
        img = np.array(distorted).astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Type 25 – Non-uniform blur (spatially varying σ per block)
    # -----------------------------------------------------------------------
    elif dist_type == 25:
        sigma_ranges = [(0.5, 2.0), (0.5, 4.0), (0.5, 6.0), (1.0, 8.0), (1.0, 12.0)]
        sigma_lo, sigma_hi = sigma_ranges[li]
        H, W, C = img.shape
        n_blocks = 8
        bH = max(1, H // n_blocks)
        bW = max(1, W // n_blocks)
        out = img.copy()
        for bi in range(n_blocks):
            for bj in range(n_blocks):
                sigma_b = rng.uniform(sigma_lo, sigma_hi)
                r0, r1 = bi * bH, min(H, (bi + 1) * bH)
                c0, c1 = bj * bW, min(W, (bj + 1) * bW)
                block = img[r0:r1, c0:c1]
                out[r0:r1, c0:c1] = ndimage.gaussian_filter(block, sigma=[sigma_b, sigma_b, 0])
        img = out.astype(np.float32)

    return Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))


def _find_coeffs(source_coords, target_coords):
    """
    Solve for the 8 perspective coefficients for PIL.Image.transform(PERSPECTIVE).
    source_coords: destination quad points, target_coords: source quad points.
    The equation system maps each destination (x2,y2) to source (x1,y1).
    """
    matrix = []
    for (x1, y1), (x2, y2) in zip(source_coords, target_coords):
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
    A = np.array(matrix, dtype=np.float64)
    # B is the vector of destination x,y coordinates interleaved
    B = np.array([coord for pair in source_coords for coord in pair], dtype=np.float64)
    result = np.linalg.solve(A, B)
    return tuple(result)


# ===========================================================================
# 2. KADID DMOS PARSING
# ===========================================================================

def parse_kadid_dmos(csv_path: Path):
    """
    Parse KADID DMOS CSV and compute mean DMOS per (dist_type, level).

    Returns a list of 125 dicts sorted by mean_dmos DESCENDING:
        [{"dist_type": int, "level": int, "mean_dmos": float}, ...]
    Index 0 = highest DMOS = best quality (rank 0).
    Index 124 = lowest DMOS = worst quality (rank 124).
    """
    import csv

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_img = row["dist_img"].strip()
            dmos = float(row["dmos"])
            # Format: I{ref:02d}_{type:02d}_{level:02d}.png
            # e.g. I01_01_01.png
            parts = dist_img.replace(".png", "").split("_")
            dist_type = int(parts[1])
            level = int(parts[2])
            rows.append((dist_type, level, dmos))

    # Aggregate mean DMOS per (dist_type, level)
    from collections import defaultdict
    agg = defaultdict(list)
    for dist_type, level, dmos in rows:
        agg[(dist_type, level)].append(dmos)

    groups = []
    for (dist_type, level), dmos_list in agg.items():
        groups.append({
            "dist_type": dist_type,
            "level": level,
            "mean_dmos": float(np.mean(dmos_list)),
        })

    # Sort descending by mean_dmos (best quality first = rank 0)
    groups.sort(key=lambda x: x["mean_dmos"], reverse=True)
    return groups


# ===========================================================================
# 3. FEATURE EXTRACTION HELPERS
# ===========================================================================

def extract_pil_list(extractor: FeatureExtractor, pil_images: list, batch_size: int = 8) -> np.ndarray:
    """
    Extract features from a list of PIL images in mini-batches.
    Returns ndarray of shape (N, D) using GAP (no Gram, no PCA).
    """
    all_feats = []
    transform = extractor.image_transform
    for i in range(0, len(pil_images), batch_size):
        batch_pil = pil_images[i:i + batch_size]
        tensors = torch.stack([transform(img) for img in batch_pil])
        feats = extractor.extract_batch(tensors)  # (B, D)
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


# ===========================================================================
# 4. MMD² COMPUTATION
# ===========================================================================

def rbf_kernel_matrix(X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute RBF kernel matrix K(X, Y) = exp(-||x-y||²/(2γ))."""
    # cdist computes ||x-y||² when p=2 then squared
    dists = torch.cdist(X, Y, p=2) ** 2
    return torch.exp(-dists / (2.0 * gamma))


def compute_mmd2_unbiased(
    X: torch.Tensor, Y: torch.Tensor, gamma: float
) -> float:
    """
    Compute unbiased MMD² between samples X (n×D) and Y (m×D).
    MMD²_u = E[K(x,x')] + E[K(y,y')] - 2·E[K(x,y)]
    where expectations exclude diagonal (unbiased).
    """
    n = X.shape[0]
    m = Y.shape[0]

    Kxx = rbf_kernel_matrix(X, X, gamma)
    Kyy = rbf_kernel_matrix(Y, Y, gamma)
    Kxy = rbf_kernel_matrix(X, Y, gamma)

    # Exclude diagonal for within-set kernels
    mask_xx = ~torch.eye(n, dtype=torch.bool, device=X.device)
    mask_yy = ~torch.eye(m, dtype=torch.bool, device=Y.device)

    mmd2 = (
        Kxx[mask_xx].mean()
        + Kyy[mask_yy].mean()
        - 2.0 * Kxy.mean()
    )
    return mmd2.item()


def compute_gamma_median(anchor_feats: np.ndarray, n_sub: int = 500) -> float:
    """
    Compute γ = median of pairwise squared L2 distances on a subsample
    of anchor features.
    """
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(anchor_feats), size=min(n_sub, len(anchor_feats)), replace=False)
    sub = torch.tensor(anchor_feats[idx], dtype=torch.float32)
    dists = torch.cdist(sub, sub, p=2) ** 2
    # Upper-triangle (exclude diagonal)
    n = sub.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    gamma = dists[mask].median().item()
    logger.info(f"Gamma (median heuristic, n_sub={len(idx)}): {gamma:.4f}")
    return float(gamma)


# ===========================================================================
# 5. MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic KADID degradation experiment on COCO images"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for feature extraction (default: 8)"
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore cached features and recompute everything"
    )
    args = parser.parse_args()

    logger.info(f"Device: {args.device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Parse KADID DMOS → group ordering
    # ------------------------------------------------------------------
    group_order_path = CACHE_DIR / "group_order.json"
    if group_order_path.exists() and not args.force_recompute:
        logger.info(f"Loading group order from {group_order_path}")
        with open(group_order_path) as f:
            group_order = json.load(f)
    else:
        logger.info("Parsing KADID DMOS CSV …")
        group_order = parse_kadid_dmos(KADID_DMOS_CSV)
        with open(group_order_path, "w") as f:
            json.dump(group_order, f, indent=2)
        logger.info(f"Saved group order to {group_order_path}")

    assert len(group_order) == N_GROUPS, (
        f"Expected {N_GROUPS} distortion groups, got {len(group_order)}"
    )
    logger.info(
        f"Group order: rank 0 = type {group_order[0]['dist_type']} "
        f"level {group_order[0]['level']} (DMOS={group_order[0]['mean_dmos']:.3f}), "
        f"rank 124 = type {group_order[124]['dist_type']} "
        f"level {group_order[124]['level']} (DMOS={group_order[124]['mean_dmos']:.3f})"
    )

    # ------------------------------------------------------------------
    # Step 2: COCO split (seed=42)
    # ------------------------------------------------------------------
    coco_paths = sorted(COCO_DIR.glob("*.jpg"))
    assert len(coco_paths) >= N_REFS + N_ANCHOR, (
        f"Need at least {N_REFS + N_ANCHOR} COCO images, found {len(coco_paths)}"
    )
    rng = np.random.default_rng(seed=42)
    shuffled_idx = rng.permutation(len(coco_paths))
    ref_paths    = [coco_paths[i] for i in shuffled_idx[:N_REFS]]
    anchor_paths = [coco_paths[i] for i in shuffled_idx[N_REFS: N_REFS + N_ANCHOR]]
    logger.info(f"COCO split: {len(ref_paths)} refs, {len(anchor_paths)} anchors")

    # ------------------------------------------------------------------
    # Step 3: Build FeatureExtractor (SD-VAE L11, GAP, no Gram)
    # ------------------------------------------------------------------
    logger.info("Initialising FeatureExtractor (sd_vae, layer=11) …")
    extractor = FeatureExtractor(
        backbone="sd_vae",
        layer=11,
        transform_config={
            "name": "gap_only",
            "use_gram": False,
            "use_pca": False,
        },
        device=args.device,
    )

    # ------------------------------------------------------------------
    # Step 4: Extract / load anchor features
    # ------------------------------------------------------------------
    anc_cache = CACHE_DIR / "anc_sd_vae_L11.npy"
    if anc_cache.exists() and not args.force_recompute:
        logger.info(f"Loading anchor features from {anc_cache}")
        anchor_feats = np.load(str(anc_cache))
    else:
        logger.info(f"Extracting anchor features for {len(anchor_paths)} images …")
        # Load images lazily batch-by-batch to avoid loading 1000 images at once
        all_feats = []
        bs = args.batch_size
        transform = extractor.image_transform
        for i in range(0, len(anchor_paths), bs):
            batch_pil = [Image.open(p).convert("RGB") for p in anchor_paths[i:i+bs]]
            tensors = torch.stack([transform(img) for img in batch_pil])
            feats = extractor.extract_batch(tensors)
            all_feats.append(feats)
            if (i // bs) % 10 == 0:
                logger.info(f"  Anchor: {i+len(batch_pil)}/{len(anchor_paths)}")
        anchor_feats = np.concatenate(all_feats, axis=0)
        np.save(str(anc_cache), anchor_feats)
        logger.info(f"Anchor features saved: {anchor_feats.shape} → {anc_cache}")

    D = anchor_feats.shape[1]
    logger.info(f"Anchor features shape: {anchor_feats.shape}  (D={D})")

    # ------------------------------------------------------------------
    # Step 5: Compute gamma
    # ------------------------------------------------------------------
    gamma = compute_gamma_median(anchor_feats, n_sub=GAMMA_SUBSAMPLE)

    # ------------------------------------------------------------------
    # Step 6: Extract / load distorted features
    # ------------------------------------------------------------------
    dist_cache = CACHE_DIR / "dist_sd_vae_L11.npy"
    dist_shape = (N_GROUPS, N_REFS, D)

    if dist_cache.exists() and not args.force_recompute:
        logger.info(f"Loading distorted features from {dist_cache}")
        dist_feats = np.load(str(dist_cache))
        assert dist_feats.shape == dist_shape, (
            f"Cached dist features shape {dist_feats.shape} != expected {dist_shape}"
        )
    else:
        logger.info(f"Extracting distorted features (shape {dist_shape}) …")
        # Create mmap file
        dist_feats = np.lib.format.open_memmap(
            str(dist_cache), mode="w+", dtype=np.float32, shape=dist_shape
        )

        # Load ref images once (PIL)
        ref_pil_list = [Image.open(p).convert("RGB") for p in ref_paths]

        for k, group in enumerate(group_order):
            dt = group["dist_type"]
            lv = group["level"]
            logger.info(
                f"Group {k:3d}/124 – distortion type={dt:2d}, level={lv}  "
                f"(mean DMOS={group['mean_dmos']:.3f})"
            )
            degraded_pil = [apply_distortion(img, dt, lv) for img in ref_pil_list]
            feats = extract_pil_list(extractor, degraded_pil, batch_size=args.batch_size)
            dist_feats[k, :, :] = feats
            # Flush periodically
            if k % 10 == 0:
                dist_feats.flush()

        dist_feats.flush()
        logger.info(f"Distorted features saved to {dist_cache}")
        # Reload as normal array for computations
        dist_feats = np.array(dist_feats)

    # ------------------------------------------------------------------
    # Step 7: Compute MMD² for each group
    # ------------------------------------------------------------------
    logger.info("Computing MMD² for each of the 125 groups …")

    anchor_t = torch.tensor(anchor_feats, dtype=torch.float32, device=args.device)
    mmd2_values = np.zeros(N_GROUPS, dtype=np.float64)

    for k in range(N_GROUPS):
        group_t = torch.tensor(dist_feats[k], dtype=torch.float32, device=args.device)
        mmd2_values[k] = compute_mmd2_unbiased(group_t, anchor_t, gamma)

    logger.info("MMD² computation done.")

    # ------------------------------------------------------------------
    # Step 8: Metrics
    # ------------------------------------------------------------------
    gt_ranks = np.arange(N_GROUPS)  # 0..124 from best to worst quality

    spearman_rho, spearman_p = spearmanr(gt_ranks, mmd2_values)
    kendall_tau, kendall_p    = kendalltau(gt_ranks, mmd2_values)

    # Strict monotonicity: consecutive pairs where MMD²[k] < MMD²[k+1]
    consecutive_increasing = np.sum(np.diff(mmd2_values) > 0)
    strict_mono = consecutive_increasing / (N_GROUPS - 1)

    # ------------------------------------------------------------------
    # Step 9: Print results table
    # ------------------------------------------------------------------
    header = f"{'Rank':>4}  {'Type':>4}  {'Level':>5}  {'MeanDMOS':>9}  {'MMD2':>14}"
    separator = "-" * len(header)
    print()
    print(header)
    print(separator)
    for k in range(N_GROUPS):
        g = group_order[k]
        print(
            f"{k:>4}  {g['dist_type']:>4}  {g['level']:>5}  "
            f"{g['mean_dmos']:>9.4f}  {mmd2_values[k]:>14.6f}"
        )

    print()
    print(f"Spearman ρ = {spearman_rho:.4f} (p={spearman_p:.2e})")
    print(f"Kendall τ  = {kendall_tau:.4f} (p={kendall_p:.2e})")
    print(f"Strict mono = {strict_mono * 100:.1f}%  "
          f"({consecutive_increasing}/{N_GROUPS - 1} consecutive pairs increasing)")
    print()

    # ------------------------------------------------------------------
    # Step 10: Save results CSV
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"summary_{timestamp}.csv"

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "dist_type", "level", "mean_dmos", "mmd2",
            "spearman_rho", "spearman_p", "kendall_tau", "kendall_p", "strict_mono"
        ])
        for k in range(N_GROUPS):
            g = group_order[k]
            writer.writerow([
                k,
                g["dist_type"],
                g["level"],
                f"{g['mean_dmos']:.6f}",
                f"{mmd2_values[k]:.8f}",
                f"{spearman_rho:.6f}" if k == 0 else "",
                f"{spearman_p:.4e}"   if k == 0 else "",
                f"{kendall_tau:.6f}"  if k == 0 else "",
                f"{kendall_p:.4e}"    if k == 0 else "",
                f"{strict_mono:.4f}"  if k == 0 else "",
            ])

    logger.info(f"Results saved to {csv_path}")

    # ------------------------------------------------------------------
    # Step 11: Scatter plot (rank vs MMD²)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(gt_ranks, mmd2_values, s=20, alpha=0.7, color="steelblue")
        ax.set_xlabel("Ground-truth rank (0=best quality, 124=worst)")
        ax.set_ylabel("MMD²")
        ax.set_title(
            f"MMD² vs quality rank — SD-VAE L11\n"
            f"Spearman ρ={spearman_rho:.3f}  Kendall τ={kendall_tau:.3f}  "
            f"Mono={strict_mono * 100:.1f}%"
        )
        ax.grid(True, alpha=0.3)
        plot_path = RESULTS_DIR / f"scatter_{timestamp}.png"
        fig.savefig(str(plot_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Scatter plot saved to {plot_path}")
    except ImportError:
        logger.warning("matplotlib not available — skipping scatter plot")

    return {
        "spearman_rho": spearman_rho,
        "kendall_tau": kendall_tau,
        "strict_mono": strict_mono,
    }


if __name__ == "__main__":
    main()
