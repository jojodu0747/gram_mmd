"""
sweep_kadid_selected.py
=======================
Evaluate 3 selected (backbone, layer, gamma) configurations + CMMD on KADID-10k.

Selected configurations (best from COCO-degraded sweep):
  - SD-VAE   L11  γ = 1e-4    (encoder.down_blocks.3.resnets.1)
  - DC-AE    L09  γ = 5e-6    (encoder.down_blocks.2.2)
  - DINOv2   L11  γ = 2.8e-5  (blocks.10)
  - CMMD     CLIP ViT-L/14@336, γ = 0.005  (baseline)

Anchor  = 81 clean reference images from KADID-10k (I01.png ... I81.png).
Queries = 10 125 distorted images (from dmos.csv).

Scoring: biased 1-sample global MMD²
    score(q) = mean_rr(K_rr) + 1 - (2 / N_r) * sum_r K(r, q)

Metrics vs degradation *level* (1 → 5):
  - Spearman ρ           (global correlation)
  - Strict mono rate     (fraction of perfectly ordered (ref, dist_type) sequences)
  - Mean Kendall τ       (partial monotonicity)

Output : results/sweep_kadid_selected/sweep_results_<ts>.csv
         + plots via plot_sweep_results.py (same format as sweep_coco_degraded)

Usage:
    .venv/bin/python experiments/sweep_kadid_selected.py
    .venv/bin/python experiments/sweep_kadid_selected.py --skip-cmmd
    .venv/bin/python experiments/sweep_kadid_selected.py --device cpu
    .venv/bin/python experiments/sweep_kadid_selected.py --no-plots
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

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE_DIR  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.config import get_backbone_config
from gram_mmd.features import FeatureHook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
OUT_DIR    = BASE_DIR / "results" / "sweep_kadid_selected"
CACHE_DIR  = OUT_DIR / "cache"

# (backbone_name, layer_idx, gamma, batch_size)
SELECTED_CONFIGS = [
    {"backbone": "sd_vae",        "layer": 11, "gamma": 1e-4,   "batch_size": 16},
    {"backbone": "dc_ae",         "layer": 9,  "gamma": 5e-6,   "batch_size": 8},
    {"backbone": "dinov2_vitb14", "layer": 11, "gamma": 2.8e-5, "batch_size": 8},
]

MAX_CHANNELS   = 512   # cap Gram channels (memory)
CLIP_MODEL_ID  = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA     = 0.005
BATCH_CLIP     = 32


# ============================================================
# Dataset
# ============================================================

class _ImageDataset(Dataset):
    def __init__(self, paths: list, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.paths[idx]
        except Exception as e:
            logger.warning(f"Cannot load {self.paths[idx]}: {e}")
            h = w = self.transform.transforms[0].size
            if isinstance(h, (list, tuple)):
                h, w = h
            return torch.zeros(3, h, w), self.paths[idx]


# ============================================================
# KADID data loading
# ============================================================

def load_kadid() -> tuple:
    """
    Returns:
        ref_paths  : list of 81 clean reference image paths
        dist_paths : list of 10 125 distorted image paths
        levels     : int array (N,)  — degradation level 1..5
        seq_ids    : int array (N,)  — (ref_idx * 25 + dist_type_idx)
    """
    rows = []
    with open(DMOS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name  = r["dist_img"].replace(".png", "")
            parts = name.split("_")
            rows.append({
                "dist_img":  r["dist_img"],
                "ref_img":   r["ref_img"],
                "ref_id":    int(parts[0][1:]),     # 1..81
                "dist_type": int(parts[1]),          # 1..25
                "level":     int(parts[2]),          # 1..5
            })

    # Reference images (unique, sorted)
    ref_names = sorted(set(r["ref_img"] for r in rows))
    ref_to_idx = {n: i for i, n in enumerate(ref_names)}
    ref_paths  = [str(IMAGES_DIR / n) for n in ref_names]

    # Distorted images
    dist_types_sorted = sorted(set(r["dist_type"] for r in rows))
    type_to_rank = {t: i for i, t in enumerate(dist_types_sorted)}   # 0..24

    dist_paths, levels, seq_ids = [], [], []
    for r in rows:
        dist_paths.append(str(IMAGES_DIR / r["dist_img"]))
        levels.append(r["level"])
        seq_ids.append(ref_to_idx[r["ref_img"]] * 25 + type_to_rank[r["dist_type"]])

    logger.info(f"KADID: {len(ref_paths)} refs, {len(dist_paths)} distorted images")
    return (
        ref_paths,
        dist_paths,
        np.array(levels,  dtype=np.int32),
        np.array(seq_ids, dtype=np.int32),
    )


# ============================================================
# Gram-MMD extractor (single layer)
# ============================================================

class SingleLayerExtractor:
    """
    Loads one backbone and hooks a single layer.
    Extracts standardized Gram features and computes MMD² scores.
    """

    def __init__(self, backbone_name: str, layer_idx: int, device: str,
                 batch_size: int, max_channels: int = MAX_CHANNELS):
        self.backbone_name = backbone_name
        self.layer_idx     = layer_idx
        self.device        = device
        self.batch_size    = batch_size
        self.max_channels  = max_channels
        self.cfg           = get_backbone_config(backbone_name)

        self.model        = None
        self.hook         = FeatureHook()
        self.hook_handle  = None
        self.scaler       = None

        self._load_model()
        self._register_hook()
        self.transform = self._make_transform()

    # ── Model loading ────────────────────────────────────────

    def _load_model(self):
        name = self.backbone_name
        cfg  = self.cfg
        logger.info(f"  Loading {name} ...")

        if name in ("sd_vae", "flux_vae"):
            from diffusers import AutoencoderKL
            self.model = AutoencoderKL.from_pretrained(cfg["weights"])
        elif name == "dc_ae":
            from diffusers import AutoencoderDC
            self.model = AutoencoderDC.from_pretrained(cfg["weights"])
        elif name == "dinov2_vitb14":
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )
        elif name == "vgg19":
            from torchvision import models as tv
            self.model = tv.vgg19(weights=tv.VGG19_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Backbone not supported: {name}")

        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _get_module(self, dotted: str) -> nn.Module:
        m = self.model
        for part in dotted.split("."):
            try:
                m = getattr(m, part)
            except AttributeError:
                m = m[int(part)]
        return m

    def _register_hook(self):
        layer_name = self.cfg["layer_names"][self.layer_idx]
        mod = self._get_module(layer_name)
        self.hook_handle = mod.register_forward_hook(self.hook)
        logger.info(f"  Hook: L{self.layer_idx} → {layer_name}")

    def _make_transform(self):
        cfg = self.cfg
        return transforms.Compose([
            transforms.Resize((cfg["input_size"], cfg["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["normalize_mean"],
                                 std=cfg["normalize_std"]),
        ])

    # ── Forward + Gram ───────────────────────────────────────

    def _forward(self, imgs: torch.Tensor):
        imgs = imgs.to(self.device)
        self.hook.features = None
        with torch.no_grad():
            if self.backbone_name in ("sd_vae", "flux_vae", "dc_ae"):
                _ = self.model.encoder(imgs)
            else:
                _ = self.model(imgs)

    def _gram(self, feat: torch.Tensor) -> np.ndarray:
        feat = feat.float()
        if feat.dim() == 4:
            B, C, H, W = feat.shape
            if C > self.max_channels:
                gen = torch.Generator(device="cpu").manual_seed(42)
                idx = torch.randperm(C, generator=gen)[:self.max_channels]
                feat = feat[:, idx.to(feat.device)]
                C = self.max_channels
            F = feat.reshape(B, C, H * W)
        elif feat.dim() == 3:
            B, N, D = feat.shape
            if D > self.max_channels:
                gen = torch.Generator(device="cpu").manual_seed(42)
                idx = torch.randperm(D, generator=gen)[:self.max_channels]
                feat = feat[:, :, idx.to(feat.device)]
                D = self.max_channels
            F = feat.permute(0, 2, 1)
            C = D
        elif feat.dim() == 2:
            return feat.cpu().numpy().astype(np.float32)
        else:
            return None

        S = F.shape[2]
        G = torch.bmm(F, F.transpose(1, 2)) / S
        triu = torch.triu(torch.ones(C, C, device=feat.device, dtype=torch.bool))
        return G[:, triu].cpu().numpy().astype(np.float32)

    # ── Batch extraction ─────────────────────────────────────

    def _extract_batch(self, imgs: torch.Tensor) -> np.ndarray | None:
        self._forward(imgs)
        feat = self.hook.features
        if feat is None:
            return None
        return self._gram(feat)

    def extract_all(self, paths: list) -> np.ndarray:
        """Extract Gram features for all image paths → (N, D)."""
        ds     = _ImageDataset(paths, self.transform)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=(self.device == "cuda"),
            persistent_workers=True,
        )
        parts = []
        for imgs, _ in loader:
            g = self._extract_batch(imgs)
            if g is not None:
                parts.append(g)
        return np.concatenate(parts, axis=0)

    # ── MMD² scoring ─────────────────────────────────────────

    def fit_scaler(self, ref_feats: np.ndarray) -> np.ndarray:
        """Fit StandardScaler on reference features, return scaled array."""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        ref_scaled = self.scaler.fit_transform(ref_feats).astype(np.float32)
        logger.info(
            f"  Scaler fitted — mean={self.scaler.mean_.mean():.3e}  "
            f"std={self.scaler.scale_.mean():.3e}"
        )
        return ref_scaled

    def score_all(self, ref_scaled: np.ndarray, dist_paths: list,
                  gamma: float) -> np.ndarray:
        """
        Compute biased 1-sample MMD²:
          score(q) = mean_rr(K_rr) + 1 - 2/N_r * Σ_r K(r, q)
        Returns float64 array (N_dist,).
        """
        # Precompute K_rr (once, on GPU)
        r  = torch.tensor(ref_scaled, device=self.device)
        r_sq = (r * r).sum(dim=1)
        D2_rr = torch.clamp(
            r_sq[:, None] + r_sq[None, :] - 2.0 * (r @ r.T), min=0.0
        )
        term1 = float(torch.exp(-gamma * D2_rr).mean().item())
        logger.info(f"  term1 (mean K_rr) = {term1:.6f}")

        # Score distorted images streaming
        n_dist = len(dist_paths)
        scores = np.empty(n_dist, dtype=np.float64)

        ds     = _ImageDataset(dist_paths, self.transform)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=(self.device == "cuda"),
            persistent_workers=True,
        )

        idx = 0
        t0  = time.time()
        for imgs, _ in loader:
            g = self._extract_batch(imgs)
            if g is None:
                idx += imgs.shape[0]
                continue

            # Scale with anchor scaler
            g_scaled = self.scaler.transform(g).astype(np.float32)

            q    = torch.tensor(g_scaled, device=self.device)
            q_sq = (q * q).sum(dim=1)
            D2_qr = torch.clamp(
                r_sq[None, :] + q_sq[:, None] - 2.0 * (q @ r.T), min=0.0
            )   # (B, N_r)
            K_qr = torch.exp(-gamma * D2_qr)                           # (B, N_r)
            term3 = 2.0 * K_qr.mean(dim=1).cpu().numpy().astype(np.float64)
            scores[idx: idx + len(term3)] = term1 + 1.0 - term3
            idx += len(term3)

            if idx % 1000 < self.batch_size or idx >= n_dist:
                logger.info(
                    f"    Scored {idx}/{n_dist} ({idx/n_dist:.0%}) — "
                    f"{time.time()-t0:.1f}s"
                )

        del r, r_sq, D2_rr
        torch.cuda.empty_cache()
        gc.collect()
        return scores

    def get_layer_name(self) -> str:
        return self.cfg["layer_names"][self.layer_idx]

    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# CMMD (CLIP ViT-L/14@336)
# ============================================================

def run_cmmd(ref_paths: list, dist_paths: list, device: str) -> np.ndarray:
    """Returns MMD² scores (N_dist,) using CLIP embeddings."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    logger.info(f"  Loading {CLIP_MODEL_ID} ...")
    model = (
        CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
        .to(device).eval()
    )
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def _embed(paths: list) -> np.ndarray:
        parts = []
        for i in range(0, len(paths), BATCH_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                parts.append(
                    model(pixel_values=pv).image_embeds.float().cpu().numpy()
                )
        return np.concatenate(parts, axis=0).astype(np.float32)

    logger.info("  Embedding references ...")
    ref = _embed(ref_paths)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    K_rr   = np.exp(-CMMD_GAMMA * (
        ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
    ))
    term1  = float(K_rr.mean())
    logger.info(f"  CMMD term1 = {term1:.6f}")

    logger.info(f"  Scoring {len(dist_paths)} distorted images ...")
    n = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    t0 = time.time()
    chunk = BATCH_CLIP * 4

    for s in range(0, n, chunk):
        e    = min(s + chunk, n)
        emb  = _embed(dist_paths[s:e])
        t_sq = np.einsum("ij,ij->i", emb, emb)
        K_rt = np.exp(-CMMD_GAMMA * (
            ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ emb.T)
        ))
        scores[s:e] = term1 + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        if s % 1000 < chunk or e == n:
            logger.info(
                f"    CMMD {e}/{n} ({e/n:.0%}) — {time.time()-t0:.1f}s"
            )
        del emb, K_rt

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return scores, term1


# ============================================================
# Metrics
# ============================================================

def compute_metrics(scores: np.ndarray, levels: np.ndarray,
                    seq_ids: np.ndarray) -> dict:
    """
    Spearman ρ (global), strict mono rate, mean Kendall τ.
    Same logic as sweep_coco_degraded.py::compute_metrics().
    """
    rho, _ = spearmanr(scores, levels)

    unique_seqs = np.unique(seq_ids)
    strict_count = 0
    tau_values   = []

    for sid in unique_seqs:
        mask = seq_ids == sid
        if mask.sum() < 2:
            continue
        s = scores[mask]
        l = levels[mask]
        order    = np.argsort(l)
        s_sorted = s[order]

        if np.all(np.diff(s_sorted) > 0):
            strict_count += 1

        tau, _ = kendalltau(l, s)
        if np.isfinite(tau):
            tau_values.append(tau)

    return {
        "spearman_rho":     float(rho),
        "strict_mono_rate": strict_count / len(unique_seqs) if unique_seqs.size > 0 else 0.0,
        "kendall_tau_mean": float(np.mean(tau_values)) if tau_values else 0.0,
        "n_images":         int(len(scores)),
        "n_sequences":      int(len(unique_seqs)),
    }


# ============================================================
# CSV output — same columns as sweep_coco_degraded.py
# ============================================================

CSV_FIELDS = [
    "backbone", "layer", "layer_name", "gamma_label", "gamma_value",
    "spearman_rho", "strict_mono_rate", "kendall_tau_mean",
    "feat_dim", "median_dist", "k_rq_mean", "term1",
    "n_images", "n_sequences", "gamma_raw",
]


def write_csv(out_path: Path, rows: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  CSV saved → {out_path}")


# ============================================================
# Plots — reuse plot_sweep_results.py
# ============================================================

def run_plots(csv_path: Path, top_k: int, out_dir: Path):
    import subprocess, sys
    script = BASE_DIR / "experiments" / "plot_sweep_results.py"
    cmd = [
        sys.executable, str(script),
        "--csv",     str(csv_path),
        "--top-k",   str(top_k),
        "--out-dir", str(out_dir),
        "--dataset", "KADID-10k  ·  anchor = 81 clean refs",
    ]
    logger.info(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cmmd",  action="store_true")
    p.add_argument("--no-plots",   action="store_true")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("KADID-10k — selected configurations sweep")
    logger.info(f"  Device   : {args.device}")
    logger.info(f"  CMMD     : {'disabled' if args.skip_cmmd else 'enabled'}")
    logger.info("=" * 70)

    ref_paths, dist_paths, levels, seq_ids = load_kadid()

    all_rows = []

    # ── Gram-MMD configs ──────────────────────────────────────
    for cfg in SELECTED_CONFIGS:
        bb    = cfg["backbone"]
        layer = cfg["layer"]
        gamma = cfg["gamma"]
        bs    = cfg["batch_size"]
        glabel = f"{gamma:.1e}"

        logger.info(f"\n{'='*60}")
        logger.info(f"  {bb}  L{layer:02d}  γ={glabel}")
        logger.info(f"{'='*60}")

        extractor = SingleLayerExtractor(
            backbone_name=bb, layer_idx=layer,
            device=args.device, batch_size=bs,
        )

        # Extract anchor (ref) features and fit scaler
        logger.info(f"  Extracting {len(ref_paths)} reference features ...")
        ref_raw = extractor.extract_all(ref_paths)
        ref_scaled = extractor.fit_scaler(ref_raw)
        feat_dim = ref_raw.shape[1]

        # Compute median distance (informational)
        r_sq = np.einsum("ij,ij->i", ref_scaled, ref_scaled)
        D2_rr = np.maximum(
            r_sq[:, None] + r_sq[None, :] - 2.0 * (ref_scaled @ ref_scaled.T),
            0.0,
        )
        median_dist = float(np.median(D2_rr[D2_rr > 0])) if (D2_rr > 0).any() else 0.0

        # Score all distorted images
        logger.info(f"  Scoring {len(dist_paths)} distorted images ...")
        scores = extractor.score_all(ref_scaled, dist_paths, gamma)

        # Metrics
        met = compute_metrics(scores, levels, seq_ids)
        logger.info(
            f"  ρ={met['spearman_rho']:+.4f}  "
            f"mono={met['strict_mono_rate']:.3f}  "
            f"τ={met['kendall_tau_mean']:+.4f}"
        )

        k_rq_mean = float(np.mean(
            np.exp(-gamma * D2_rr)   # approximate: use D_rr as proxy
        ))

        all_rows.append({
            "backbone":          bb,
            "layer":             layer,
            "layer_name":        extractor.get_layer_name(),
            "gamma_label":       glabel,
            "gamma_value":       gamma,
            "spearman_rho":      met["spearman_rho"],
            "strict_mono_rate":  met["strict_mono_rate"],
            "kendall_tau_mean":  met["kendall_tau_mean"],
            "feat_dim":          feat_dim,
            "median_dist":       median_dist,
            "k_rq_mean":         k_rq_mean,
            "term1":             "",
            "n_images":          met["n_images"],
            "n_sequences":       met["n_sequences"],
            "gamma_raw":         gamma,
        })

        extractor.cleanup()

    # ── CMMD ─────────────────────────────────────────────────
    if not args.skip_cmmd:
        logger.info(f"\n{'='*60}")
        logger.info("  CMMD  (CLIP ViT-L/14@336)")
        logger.info(f"{'='*60}")

        scores_cmmd, term1_cmmd = run_cmmd(ref_paths, dist_paths, args.device)
        met_cmmd = compute_metrics(scores_cmmd, levels, seq_ids)

        logger.info(
            f"  CMMD  ρ={met_cmmd['spearman_rho']:+.4f}  "
            f"mono={met_cmmd['strict_mono_rate']:.3f}  "
            f"τ={met_cmmd['kendall_tau_mean']:+.4f}"
        )

        all_rows.append({
            "backbone":          "cmmd",
            "layer":             0,
            "layer_name":        "CLIP ViT-L/14@336",
            "gamma_label":       f"{CMMD_GAMMA:.1e}",
            "gamma_value":       CMMD_GAMMA,
            "spearman_rho":      met_cmmd["spearman_rho"],
            "strict_mono_rate":  met_cmmd["strict_mono_rate"],
            "kendall_tau_mean":  met_cmmd["kendall_tau_mean"],
            "feat_dim":          768,
            "median_dist":       "",
            "k_rq_mean":         "",
            "term1":             term1_cmmd,
            "n_images":          met_cmmd["n_images"],
            "n_sequences":       met_cmmd["n_sequences"],
            "gamma_raw":         CMMD_GAMMA,
        })

    # ── Summary ───────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS SUMMARY — KADID-10k")
    logger.info(f"{'Backbone':<18} {'Layer':<6} {'γ':<10} {'Spearman ρ':>10}  "
                f"{'Strict mono':>11}  {'Kendall τ':>9}")
    logger.info("-" * 70)
    for r in sorted(all_rows, key=lambda x: -x["spearman_rho"]):
        logger.info(
            f"  {r['backbone']:<16} L{r['layer']:<4} "
            f"{r['gamma_label']:<10} {r['spearman_rho']:>+10.4f}  "
            f"{r['strict_mono_rate']:>11.3f}  {r['kendall_tau_mean']:>+9.4f}"
        )
    logger.info("=" * 70)

    # ── CSV ───────────────────────────────────────────────────
    csv_path = OUT_DIR / f"sweep_results_{ts}.csv"
    write_csv(csv_path, all_rows)

    # ── Plots ─────────────────────────────────────────────────
    if not args.no_plots:
        n_configs = len(all_rows)
        plot_dir  = csv_path.parent / "plots"
        try:
            run_plots(csv_path, top_k=n_configs, out_dir=plot_dir)
        except Exception as e:
            logger.warning(f"  Plots failed: {e}. Run manually with:")
            logger.warning(
                f"  python experiments/plot_sweep_results.py "
                f"--csv {csv_path} --top-k {n_configs} --out-dir {plot_dir}"
            )

    logger.info(f"\nDone.  Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
