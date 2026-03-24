#!/usr/bin/env python3
"""
Sweep all layers on the synthetic KADID-COCO experiment.

Structure: 20 distortion types × 10 severity levels = 200 groups.
Redondances éliminées vs les 25 originaux :
  - T2 Luminance noise  (proche T1 Gaussian noise)
  - T4 Denoise sim      (composite T1+T18)
  - T7 Mean shift       (proche T5 Brighten)
  - T14 Spatter         (proche T16 Sparse sampling)
  - T21 Diffuse         (approxime T18 Gaussian blur)

Evaluation (option B): pour chaque type, Spearman ρ sur ses 10 niveaux.
Métrique reportée : mean ρ sur 20 types (par backbone/layer/gamma).

Outputs:
  results/synthetic_kadid_sweep/summary_{ts}.csv
  results/synthetic_kadid_sweep/cache/{backbone}_anc.npz
  results/synthetic_kadid_sweep/cache/{backbone}_dist_20types.npz
"""

import argparse
import csv
import gc
import logging
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gram_mmd.config import BACKBONE_CONFIGS
from gram_mmd.features import FeatureExtractor, ImageDataset

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
COCO_DIR   = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR    = BASE_DIR / "results" / "synthetic_kadid_sweep"
CACHE_DIR  = OUT_DIR / "cache"

ANCHOR_SEED     = 42
N_ANCHOR        = 1000
N_REFS          = 50
N_DIST_TYPES    = 20
N_LEVELS        = 10
N_GROUPS        = N_DIST_TYPES * N_LEVELS   # 200
GAMMA_MULTS     = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
GAMMA_SUBSAMPLE = 500

BACKBONE_SWEEP = {
    "sd_vae":         list(range(17)),
    "dc_ae":          list(range(22)),
    "lpips_vgg":      list(range(13)),
    "vgg19":          list(range(18)),
    "dinov2_vitb14":  list(range(14)),
    "resnet50":       list(range(25)),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Groups
# ─────────────────────────────────────────────────────────────────────────────

def build_groups():
    """200 groups: type t (1..20) × level lv (1..10). rank = lv."""
    groups = []
    for t in range(1, N_DIST_TYPES + 1):
        for lv in range(1, N_LEVELS + 1):
            groups.append({"dist_type": t, "level": lv, "rank": lv})
    assert len(groups) == N_GROUPS
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Distortion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(lo, hi, level, n=N_LEVELS):
    """Linear interpolation: level 1 → lo, level n → hi."""
    t = (level - 1) / (n - 1)
    return lo + (hi - lo) * t


def _lerpf(lo, hi, level):
    return float(_lerp(lo, hi, level))


def _lerpi(lo, hi, level):
    return int(round(_lerp(lo, hi, level)))


def apply_distortion(img_pil: Image.Image, dist_type: int, level: int,
                     img_seed: int = 0) -> Image.Image:
    """Apply distortion type t (1..20) at severity level (1..10).

    Mapping (original KADID numbering → new index):
      1=T1 Gaussian noise,   2=T3 Multiplicative noise,
      3=T5 Brighten,         4=T6 Darken,
      5=T8 Jitter,           6=T9 Patches,
      7=T10 Pixelate,        8=T11 Quantization,
      9=T12 Fog,             10=T13 Color cast cool,
      11=T15 Chrom. aberr.,  12=T16 Sparse sampling,
      13=T17 JPEG,           14=T18 Gaussian blur,
      15=T19 Lens blur,      16=T20 Motion blur,
      17=T22 Tilt-stretch,   18=T23 Vignette,
      19=T24 Contrast compr, 20=T25 Non-uniform blur
    """
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    rng = np.random.default_rng(img_seed + dist_type * 1000 + level * 100)

    # 1 — Gaussian noise  σ: 0.002 → 0.022
    if dist_type == 1:
        sig = _lerpf(0.002, 0.022, level)
        arr = arr + rng.normal(0, sig, arr.shape).astype(np.float32)

    # 2 — Multiplicative noise  σ: 0.002 → 0.030
    elif dist_type == 2:
        sig = _lerpf(0.002, 0.030, level)
        arr = arr * (1 + rng.normal(0, sig, arr.shape).astype(np.float32))

    # 3 — Brighten  γ: 0.96 → 0.945
    elif dist_type == 3:
        g = _lerpf(0.96, 0.945, level)
        arr = arr.clip(0, 1) ** g

    # 4 — Darken  γ: 1.10 → 1.13
    elif dist_type == 4:
        g = _lerpf(1.10, 1.13, level)
        arr = arr.clip(0, 1) ** g

    # 5 — Jitter  amp: 1 → 5 px
    elif dist_type == 5:
        amp = _lerpf(1.0, 5.0, level)
        dy = rng.uniform(-amp, amp, (H, W)).astype(np.float32)
        dx = rng.uniform(-amp, amp, (H, W)).astype(np.float32)
        gy, gx = np.mgrid[0:H, 0:W]
        coords = [np.clip(gy + dy, 0, H - 1), np.clip(gx + dx, 0, W - 1)]
        arr = np.stack([ndimage.map_coordinates(arr[:, :, c], coords, order=1)
                        for c in range(3)], axis=2).astype(np.float32)

    # 6 — Non-eccentricity patches  ps: 4→10, n: 1→6
    elif dist_type == 6:
        ps = _lerpi(4, 10, level)
        n_p = _lerpi(1, 6, level)
        mu = arr.mean(axis=(0, 1))
        for _ in range(n_p):
            y = rng.integers(0, max(1, H - ps))
            x = rng.integers(0, max(1, W - ps))
            arr[y:y+ps, x:x+ps] = mu

    # 7 — Pixelate  f: 2 → 3
    elif dist_type == 7:
        f = max(2, _lerpi(2, 3, level))
        small = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).resize(
            (max(1, W // f), max(1, H // f)), Image.NEAREST)
        arr = np.array(small.resize((W, H), Image.NEAREST)).astype(np.float32) / 255.0

    # 8 — Quantization  bits: 8 → 5
    elif dist_type == 8:
        bits = max(5, _lerpi(8, 5, level))
        levels_q = 2 ** bits
        arr = np.round(arr * (levels_q - 1)) / (levels_q - 1)

    # 9 — Fog  α: 0.02 → 0.11
    elif dist_type == 9:
        alpha = _lerpf(0.02, 0.11, level)
        arr = arr * (1 - alpha) + alpha

    # 10 — Color cast cool  s: 0.02 → 0.08
    elif dist_type == 10:
        s = _lerpf(0.02, 0.08, level)
        arr[:, :, 2] = np.clip(arr[:, :, 2] + s, 0, 1)
        arr[:, :, 0] = np.clip(arr[:, :, 0] - s * 0.7, 0, 1)

    # 11 — Chromatic aberration  shift: 1 → 3 px
    elif dist_type == 11:
        s = max(1, _lerpi(1, 3, level))
        out = arr.copy()
        out[:, s:, 0]   = arr[:, :W-s, 0]
        out[:, :W-s, 2] = arr[:, s:, 2]
        arr = out

    # 12 — Sparse sampling  frac: 0.01 → 0.08
    elif dist_type == 12:
        frac = _lerpf(0.01, 0.08, level)
        mask = rng.random((H, W)) < frac
        for c in range(3):
            ch = arr[:, :, c].copy()
            ch[mask] = 0.0
            arr[:, :, c] = ch

    # 13 — JPEG compression  q: 95 → 72
    elif dist_type == 13:
        q = max(1, _lerpi(95, 72, level))
        buf = BytesIO()
        Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(
            buf, format="JPEG", quality=q)
        buf.seek(0)
        arr = np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0

    # 14 — Gaussian blur  σ: 0.2 → 0.40
    elif dist_type == 14:
        sig = _lerpf(0.2, 0.40, level)
        arr = ndimage.gaussian_filter(arr, sigma=[sig, sig, 0])

    # 15 — Lens blur (disk)  r: 1 → 2
    elif dist_type == 15:
        r = max(1, _lerpi(1, 2, level))
        y_, x_ = np.ogrid[-r:r+1, -r:r+1]
        kernel = ((x_**2 + y_**2) <= r**2).astype(np.float32)
        kernel /= kernel.sum()
        arr = np.stack([ndimage.convolve(arr[:, :, c], kernel) for c in range(3)], axis=2)

    # 16 — Motion blur (horizontal)  L: 3 → 4
    elif dist_type == 16:
        L = max(3, _lerpi(3, 4, level))
        kernel = np.zeros((L, L), dtype=np.float32)
        kernel[L // 2, :] = 1.0 / L
        arr = np.stack([ndimage.convolve(arr[:, :, c], kernel) for c in range(3)], axis=2)

    # 17 — Tilt-stretch  sx: 0.97 → 0.95
    elif dist_type == 17:
        sx = _lerpf(0.97, 0.95, level)
        new_w = max(1, int(W * sx))
        stretched = img_pil.resize((new_w, H), Image.BILINEAR)
        canvas = Image.new("RGB", (W, H), (128, 128, 128))
        canvas.paste(stretched, ((W - new_w) // 2, 0))
        arr = np.array(canvas).astype(np.float32) / 255.0

    # 18 — Vignette  strength: 0.08 → 0.18
    elif dist_type == 18:
        strength = _lerpf(0.08, 0.18, level)
        Y_g, X_g = np.ogrid[:H, :W]
        r_v = np.sqrt(((Y_g - H / 2) / (H / 2)) ** 2 + ((X_g - W / 2) / (W / 2)) ** 2)
        mask = np.clip(1 - strength * np.clip(r_v, 0, 1) ** 1.5, 0, 1)[:, :, None]
        arr = arr * mask

    # 19 — Contrast compress  α: 0.94 → 0.88
    elif dist_type == 19:
        alpha = _lerpf(0.94, 0.88, level)
        arr = alpha * arr + (1 - alpha) * 0.5

    # 20 — Non-uniform blur  lo: 0.2→0.8, hi: 0.4→2.5
    elif dist_type == 20:
        lo = _lerpf(0.2, 0.8, level)
        hi = _lerpf(0.4, 2.5, level)
        BLOCK = 64
        out = arr.copy()
        for y in range(0, H, BLOCK):
            for x in range(0, W, BLOCK):
                sig = rng.uniform(lo, hi)
                patch = arr[y:y+BLOCK, x:x+BLOCK]
                out[y:y+BLOCK, x:x+BLOCK] = ndimage.gaussian_filter(patch, sigma=[sig, sig, 0])
        arr = out

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# COCO split
# ─────────────────────────────────────────────────────────────────────────────

def get_coco_split():
    all_paths = sorted(
        list(COCO_DIR.glob("*.jpg")) + list(COCO_DIR.glob("*.png"))
    )
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = rng.permutation(len(all_paths))
    ref_paths    = [all_paths[i] for i in idx[:N_REFS]]
    anchor_paths = [all_paths[i] for i in idx[N_REFS:N_REFS + N_ANCHOR]]
    return ref_paths, anchor_paths


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer extractor — Gram matrices
# ─────────────────────────────────────────────────────────────────────────────

def gram_spatial(t: torch.Tensor) -> np.ndarray:
    t = t.float()
    if t.dim() == 4:
        B, C, H, W = t.shape
        F = t.reshape(B, C, H * W)
    elif t.dim() == 3:
        B, N, D = t.shape
        F = t.permute(0, 2, 1)
        C = D
    elif t.dim() == 2:
        B, D = t.shape
        F = t.unsqueeze(2)
        C = D
    else:
        return t.cpu().numpy()
    S = F.shape[2]
    G = torch.bmm(F, F.transpose(1, 2)) / S
    mask = torch.triu(torch.ones(C, C, device=t.device, dtype=torch.bool))
    return G[:, mask].cpu().numpy()


class MultiLayerExtractor:
    def __init__(self, backbone: str, layer_indices: list, device: str):
        self.backbone = backbone
        self.layer_indices = layer_indices
        self.device = device
        self._fe = FeatureExtractor(
            backbone=backbone,
            layer_config={"name": "all", "layers": layer_indices},
            transform_config={"name": "gram_spatial", "use_gram": True, "use_pca": False},
            device=device,
        )
        self.transform = self._fe.image_transform

    def extract(self, pil_images: list, batch_size: int = 16) -> dict:
        accum = {l: [] for l in self.layer_indices}
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            tensor = torch.stack([self.transform(img) for img in batch]).to(self.device)
            self._forward(tensor)
            for l in self.layer_indices:
                hook = self._fe.hooks.get(l)
                if hook is not None and hook.features is not None:
                    accum[l].append(gram_spatial(hook.features))
        return {l: np.concatenate(accum[l], axis=0) for l in self.layer_indices if accum[l]}

    def _forward(self, tensor: torch.Tensor):
        for h in self._fe.hooks.values():
            h.clear()
        with torch.no_grad():
            name = self.backbone
            if name in ("sd_vae", "flux_vae", "dc_ae"):
                self._fe.model.encoder(tensor)
            elif name == "lpips_vgg":
                scaled = self._fe.model.scaling_layer(tensor)
                self._fe.model.net(scaled)
            else:
                self._fe.model(tensor)


def extract_lazy(extractor: MultiLayerExtractor, paths: list, batch_size: int = 16) -> dict:
    from torch.utils.data import DataLoader
    str_paths = [str(p) for p in paths]
    dataset = ImageDataset(str_paths, transform=extractor.transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                        pin_memory=False, shuffle=False)
    accum = {l: [] for l in extractor.layer_indices}
    for tensor_batch, _ in loader:
        tensor_batch = tensor_batch.to(extractor.device, non_blocking=True)
        with torch.no_grad():
            extractor._forward(tensor_batch)
        for l in extractor.layer_indices:
            hook = extractor._fe.hooks.get(l)
            if hook is not None and hook.features is not None:
                accum[l].append(gram_spatial(hook.features))
                hook.clear()  # free VRAM immediately after gram computation
        torch.cuda.empty_cache()
    return {l: np.concatenate(accum[l], axis=0) for l in extractor.layer_indices if accum[l]}


# ─────────────────────────────────────────────────────────────────────────────
# γ median heuristic
# ─────────────────────────────────────────────────────────────────────────────

def compute_gamma_med(feats: np.ndarray, subsample: int = GAMMA_SUBSAMPLE,
                      device: str = "cpu") -> float:
    n = min(subsample, feats.shape[0])
    idx = np.random.default_rng(0).choice(feats.shape[0], n, replace=False)
    X = torch.tensor(feats[idx], dtype=torch.float32, device=device)
    D2 = torch.cdist(X, X, p=2).pow(2)
    triu = D2[torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)]
    return float(triu.median())


# ─────────────────────────────────────────────────────────────────────────────
# MMD²
# ─────────────────────────────────────────────────────────────────────────────

def precompute_anchor_dists(Y: torch.Tensor):
    """Pre-compute anchor pairwise distances (done once per layer, reused 200× for groups)."""
    n = Y.shape[0]
    D_rr = torch.cdist(Y, Y, p=2).pow(2)
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=Y.device), diagonal=1)
    triu_rr = D_rr[triu_mask]
    del D_rr
    return triu_rr


def mmd2_all_gammas(X: torch.Tensor, Y: torch.Tensor, gammas: list,
                    triu_rr: torch.Tensor = None) -> np.ndarray:
    m, n = X.shape[0], Y.shape[0]
    D_qq = torch.cdist(X, X, p=2).pow(2)
    D_rq = torch.cdist(Y, X, p=2).pow(2)
    triu_qq = D_qq[torch.triu(torch.ones(m, m, dtype=torch.bool, device=X.device), diagonal=1)]
    del D_qq
    if triu_rr is None:
        D_rr = torch.cdist(Y, Y, p=2).pow(2)
        triu_rr = D_rr[torch.triu(torch.ones(n, n, dtype=torch.bool, device=X.device), diagonal=1)]
        del D_rr
    results = []
    for g in gammas:
        two_g = 2.0 * g
        k_qq = torch.exp(-triu_qq / two_g).mean()
        k_rr = torch.exp(-triu_rr / two_g).mean()
        k_rq = torch.exp(-D_rq / two_g).mean()
        results.append((k_qq + k_rr - 2 * k_rq).item())
    del D_rq, triu_qq
    return np.array(results)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics: per-type ρ → mean over 20 types
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_per_type(mmd2_matrix: np.ndarray, groups: list) -> tuple:
    """
    mmd2_matrix: (200, n_gammas)
    Returns: mean_rho (n_gammas,), mean_mono (n_gammas,), per_type_rho (20, n_gammas)
    """
    n_gammas = mmd2_matrix.shape[1]
    per_type_rho  = np.zeros((N_DIST_TYPES, n_gammas))
    per_type_mono = np.zeros((N_DIST_TYPES, n_gammas))

    for ti, t in enumerate(range(1, N_DIST_TYPES + 1)):
        idx    = [k for k, g in enumerate(groups) if g["dist_type"] == t]
        levels = np.array([groups[k]["level"] for k in idx], dtype=float)
        mmd2_t = mmd2_matrix[idx]   # (N_LEVELS, n_gammas)
        for gi in range(n_gammas):
            rho, _ = spearmanr(levels, mmd2_t[:, gi])
            per_type_rho[ti, gi]  = rho if not np.isnan(rho) else 0.0
            per_type_mono[ti, gi] = float((np.diff(mmd2_t[:, gi]) > 0).sum()) / (N_LEVELS - 1)

    return per_type_rho.mean(axis=0), per_type_mono.mean(axis=0), per_type_rho


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 = auto based on input size and 12GB VRAM")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONE_SWEEP.keys()),
                        choices=list(BACKBONE_SWEEP.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 80)
    logger.info(f"Synthetic KADID sweep — {N_DIST_TYPES} types × {N_LEVELS} levels = {N_GROUPS} groups")
    logger.info(f"  Backbones : {args.backbones}")
    logger.info(f"  Device    : {args.device}")

    groups = build_groups()
    ref_paths, anchor_paths = get_coco_split()
    logger.info(f"  COCO: {N_REFS} refs, {N_ANCHOR} anchor")
    clean_imgs = [Image.open(p).convert("RGB") for p in ref_paths]

    all_results = []

    for backbone in args.backbones:
        t_bb = time.time()
        layers = BACKBONE_SWEEP[backbone]
        if args.layers is not None:
            layers = [l for l in layers if l in args.layers]
        anc_cache  = CACHE_DIR / f"{backbone}_anc.npz"
        dist_cache = CACHE_DIR / f"{backbone}_dist_20types.npz"

        logger.info("")
        logger.info(f"{'─'*60}")
        logger.info(f"Backbone: {backbone}  ({len(layers)} layers)")

        bb_input_size = BACKBONE_CONFIGS.get(backbone, {}).get("input_size", 224)
        sequential_layers = True  # always sequential: 1 mmap at a time avoids RAM explosion
        if args.batch_size > 0:
            bs = args.batch_size
        elif bb_input_size >= 512:
            bs = 8   # large input: conservative batch
        elif bb_input_size >= 256:
            bs = 16
        else:
            bs = 16
        logger.info(f"  Input size: {bb_input_size}px  →  batch_size={bs}"
                    + (" [sequential layers]" if sequential_layers else ""))

        # ── Per-layer cache directory ──
        layer_cache_dir = CACHE_DIR / backbone
        layer_cache_dir.mkdir(parents=True, exist_ok=True)

        # Migrate old monolithic cache → per-layer files (one-time)
        has_any_anc = any((layer_cache_dir / f"anc_L{l}.npz").exists() for l in layers)
        if anc_cache.exists() and not has_any_anc:
            logger.info("  Migrating monolithic anchor cache → per-layer files…")
            _npz = np.load(str(anc_cache))
            for k in _npz.files:
                np.savez_compressed(str(layer_cache_dir / f"anc_{k}.npz"), data=_npz[k])
            _npz.close(); del _npz; gc.collect()
        has_any_dist = any((layer_cache_dir / f"dist_L{l}.npz").exists() for l in layers)
        if dist_cache.exists() and not has_any_dist:
            logger.info("  Migrating monolithic dist cache → per-layer files…")
            _npz = np.load(str(dist_cache))
            for k in _npz.files:
                np.savez_compressed(str(layer_cache_dir / f"dist_{k}.npz"), data=_npz[k])
            _npz.close(); del _npz; gc.collect()

        # ── Check which layers need extraction ──
        layers_to_extract = []
        for l in layers:
            a_path = layer_cache_dir / f"anc_L{l}.npz"
            d_npy  = layer_cache_dir / f"dist_L{l}.npy"
            d_npz  = layer_cache_dir / f"dist_L{l}.npz"
            if args.force_recompute or not a_path.exists() or (not d_npy.exists() and not d_npz.exists()):
                layers_to_extract.append(l)

        if layers_to_extract:
            logger.info(f"  Extracting {len(layers_to_extract)} layers: {layers_to_extract}"
                        + (" [sequential mode]" if sequential_layers else ""))

            if sequential_layers:
                # Pre-generate all distorted images once (avoids 22× CPU regeneration)
                layers_need_dist = [l for l in layers_to_extract
                                    if not (layer_cache_dir / f"dist_L{l}.npy").exists()
                                    and not (layer_cache_dir / f"dist_L{l}.npz").exists()
                                    and not args.force_recompute]
                layers_need_dist_forced = layers_to_extract if args.force_recompute else layers_need_dist

                if layers_need_dist_forced:
                    logger.info(f"  Pre-generating {N_GROUPS} groups × {N_REFS} distorted images…")
                    all_dist_pils = []  # list[list[PIL]] — shape (N_GROUPS, N_REFS)
                    for k, grp in enumerate(groups):
                        dt, lv = grp["dist_type"], grp["level"]
                        if lv == 1:
                            logger.info(f"    Type {dt:02d}/{N_DIST_TYPES} …")
                        all_dist_pils.append([apply_distortion(img, dt, lv, img_seed=i)
                                              for i, img in enumerate(clean_imgs)])
                    logger.info(f"  Pre-generation done ({len(all_dist_pils)} groups in RAM)")
                else:
                    all_dist_pils = None

                feat_dims = {}
                for l in layers_to_extract:
                    anc_path = layer_cache_dir / f"anc_L{l}.npz"
                    dist_npy = layer_cache_dir / f"dist_L{l}.npy"
                    dist_npz = layer_cache_dir / f"dist_L{l}.npz"
                    if not args.force_recompute and anc_path.exists() and (dist_npy.exists() or dist_npz.exists()):
                        _a = np.load(str(anc_path))["data"]
                        feat_dims[l] = _a.shape[1]; del _a
                        continue
                    logger.info(f"    Layer {l} …")
                    try:
                        ext = MultiLayerExtractor(backbone, [l], args.device)
                    except Exception as e:
                        logger.warning(f"    SKIP layer {l}: {e}"); continue

                    if not anc_path.exists() or args.force_recompute:
                        anc_l = extract_lazy(ext, anchor_paths, bs)
                        if l not in anc_l:
                            del ext; gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            continue
                        np.savez_compressed(str(anc_path), data=anc_l[l])
                        feat_dims[l] = anc_l[l].shape[1]
                        del anc_l; gc.collect()
                    else:
                        _a = np.load(str(anc_path))["data"]
                        feat_dims[l] = _a.shape[1]; del _a

                    tmp_path = layer_cache_dir / f"dist_L{l}.npy.tmp"
                    mmap_l = np.lib.format.open_memmap(
                        str(tmp_path), mode='w+', dtype=np.float32,
                        shape=(N_GROUPS, N_REFS, feat_dims[l]))
                    src = all_dist_pils if all_dist_pils is not None else None
                    for k, grp in enumerate(groups):
                        dt, lv = grp["dist_type"], grp["level"]
                        if lv == 1:
                            logger.info(f"      Type {dt:02d}/{N_DIST_TYPES} …")
                        if src is not None:
                            imgs_k = src[k]
                        else:
                            imgs_k = [apply_distortion(img, dt, lv, img_seed=i)
                                      for i, img in enumerate(clean_imgs)]
                        feats_k = ext.extract(imgs_k, batch_size=bs)
                        if l in feats_k:
                            mmap_l[k] = feats_k[l]
                        del feats_k; gc.collect()
                    mmap_l.flush()
                    tmp_path.rename(layer_cache_dir / f"dist_L{l}.npy")
                    del mmap_l, ext; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                del all_dist_pils; gc.collect()
                logger.info(f"  Sequential extraction done ({len(feat_dims)} layers)")

            else:
                try:
                    ext = MultiLayerExtractor(backbone, layers_to_extract, args.device)
                except Exception as e:
                    logger.warning(f"  SKIP {backbone}: {e}")
                    continue

                # Anchor features
                logger.info(f"  Extracting anchor ({N_ANCHOR} images)…")
                anc_feats = extract_lazy(ext, anchor_paths, bs)
                feat_dims = {}
                for l in anc_feats:
                    np.savez_compressed(str(layer_cache_dir / f"anc_L{l}.npz"), data=anc_feats[l])
                    feat_dims[l] = anc_feats[l].shape[1]
                logger.info(f"  Anchor saved ({len(anc_feats)} layers)")
                del anc_feats; gc.collect()

                # Dist features — mmap
                logger.info(f"  Extracting dist ({N_GROUPS} groups × {N_REFS} imgs)…")
                dist_mmaps = {}
                for l in layers_to_extract:
                    if l not in feat_dims:
                        continue
                    tmp_path = layer_cache_dir / f"dist_L{l}.npy.tmp"
                    dist_mmaps[l] = np.lib.format.open_memmap(
                        str(tmp_path), mode='w+', dtype=np.float32,
                        shape=(N_GROUPS, N_REFS, feat_dims[l]))

                for k, grp in enumerate(groups):
                    dt, lv = grp["dist_type"], grp["level"]
                    if lv == 1:
                        logger.info(f"    Type {dt:02d}/{N_DIST_TYPES} …")
                    dist_imgs = [apply_distortion(img, dt, lv, img_seed=i)
                                 for i, img in enumerate(clean_imgs)]
                    feats_k = ext.extract(dist_imgs, batch_size=bs)
                    for l in dist_mmaps:
                        if l in feats_k:
                            dist_mmaps[l][k] = feats_k[l]
                    del dist_imgs, feats_k; gc.collect()

                for l in dist_mmaps:
                    dist_mmaps[l].flush()
                    tmp_path = layer_cache_dir / f"dist_L{l}.npy.tmp"
                    final_path = layer_cache_dir / f"dist_L{l}.npy"
                    tmp_path.rename(final_path)
                del dist_mmaps; gc.collect()
                logger.info(f"  Dist saved ({len(feat_dims)} layers)")

                del ext; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.info(f"  All {len(layers)} layers cached — skipping extraction")

        # ── Scoring: one layer at a time, memory-mapped dist ──
        from sklearn.preprocessing import StandardScaler
        logger.info(f"  Scoring {len(layers)} layers × {len(GAMMA_MULTS)} gammas…")

        for l in layers:
            anc_path = layer_cache_dir / f"anc_L{l}.npz"
            dist_npy = layer_cache_dir / f"dist_L{l}.npy"
            dist_npz = layer_cache_dir / f"dist_L{l}.npz"

            if not anc_path.exists():
                logger.warning(f"    L{l}: anchor cache missing, skipping")
                continue

            # Load dist features (prefer npy/mmap, fallback to npz)
            if dist_npy.exists():
                dist_data = np.load(str(dist_npy), mmap_mode='r')
                logger.info(f"    L{l}: dist mmap ({dist_data.shape}, ~0 RAM)")
            elif dist_npz.exists():
                dist_data = np.load(str(dist_npz))["data"]
                logger.info(f"    L{l}: dist loaded ({dist_data.shape}, {dist_data.nbytes/1e9:.1f} GB)")
            else:
                logger.warning(f"    L{l}: dist cache missing, skipping")
                continue

            # Load anchor (~500 MB)
            anc_l = np.load(str(anc_path))["data"]
            scaler = StandardScaler()
            anc_scaled = scaler.fit_transform(anc_l).astype(np.float32)
            del anc_l; gc.collect()
            D = anc_scaled.shape[1]

            # Compute gamma_med BEFORE GPU transfer (avoids extra CPU copy)
            gamma_med = compute_gamma_med(anc_scaled, device=args.device)
            gammas = [gamma_med * m for m in GAMMA_MULTS]

            # Move anchor to GPU
            anc_t_l = torch.tensor(anc_scaled, dtype=torch.float32, device=args.device)
            del anc_scaled; gc.collect()

            # Pre-compute anchor pairwise distances (once per layer, reused 200×)
            triu_rr = precompute_anchor_dists(anc_t_l)

            mmd2_matrix = np.zeros((N_GROUPS, len(GAMMA_MULTS)), dtype=np.float32)
            for k in range(N_GROUPS):
                # Transform single group inline (~50×D, avoids 5+ GB full copy)
                x_k = scaler.transform(
                    dist_data[k].reshape(-1, D)).astype(np.float32)
                X = torch.tensor(x_k, dtype=torch.float32, device=args.device)
                mmd2_matrix[k] = mmd2_all_gammas(X, anc_t_l, gammas, triu_rr=triu_rr)
                del x_k, X

            del triu_rr, anc_t_l, dist_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mean_rho, mean_mono, per_type_rho = compute_metrics_per_type(mmd2_matrix, groups)

            best_gi = int(np.argmax(mean_rho))
            logger.info(f"    L{l} per-type ρ at γ×={GAMMA_MULTS[best_gi]:.2f} (best mean ρ={mean_rho[best_gi]:+.4f}):")
            for ti in range(N_DIST_TYPES):
                logger.info(f"      T{ti+1:02d}: ρ={per_type_rho[ti, best_gi]:+.3f}")

            for gi, mult in enumerate(GAMMA_MULTS):
                all_results.append({
                    "backbone":    backbone,
                    "layer":       l,
                    "gamma_mult":  mult,
                    "gamma_val":   gammas[gi],
                    "spearman":    mean_rho[gi],
                    "mono_strict": mean_mono[gi],
                    "rho_min":     per_type_rho[:, gi].min(),
                    "rho_max":     per_type_rho[:, gi].max(),
                })

        logger.info(f"  {backbone} done in {time.time()-t_bb:.0f}s")
        gc.collect()

    # Save CSV
    csv_path = OUT_DIR / f"summary_{ts}.csv"
    fields = ["backbone", "layer", "gamma_mult", "gamma_val",
              "spearman", "mono_strict", "rho_min", "rho_max"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fields})
    logger.info(f"\nCSV saved: {csv_path}  ({len(all_results)} rows)")

    # Top-10
    top = sorted(all_results, key=lambda x: x["spearman"], reverse=True)[:10]
    logger.info(f"\nTop-10 by mean Spearman ρ (over {N_DIST_TYPES} types):")
    logger.info(f"  {'Backbone':<18} {'Layer':>5} {'γ×':>6}  {'mean ρ':>7}  {'mono':>6}  {'ρ_min':>7}  {'ρ_max':>7}")
    for r in top:
        logger.info(f"  {r['backbone']:<18} {r['layer']:>5} {r['gamma_mult']:>6.2f}  "
                    f"{r['spearman']:>+7.4f}  {r['mono_strict']:>6.1%}  "
                    f"{r['rho_min']:>+7.4f}  {r['rho_max']:>+7.4f}")


if __name__ == "__main__":
    main()
