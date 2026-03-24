"""
sweep_all_layers_kadid.py
=========================
Sweep ALL layers of sd_vae, dc_ae, dinov2_vitb14 on KADID-10k.

Each backbone is loaded ONCE; a single forward pass captures activations
from all layers simultaneously (one hook per layer).

For each backbone + layer:
  - Streaming MMD² N=1 (biased) on all 10 125 KADID degraded images
  - Spearman(score, level), Kendall tau
  - Monotonicity strict / partial rate

Also runs CMMD (CLIP ViT-L/14@336) as baseline.

Outputs (results/sweep_all_layers/) :
  sweep_spearman_per_layer_<ts>.png        ← Spearman per layer per backbone
  sweep_best_model_monotonicity_<ts>.png   ← Best layer per backbone: score vs level
  sweep_results_<ts>.csv                  ← Full results table

Usage:
    python experiments/sweep_all_layers_kadid.py
    python experiments/sweep_all_layers_kadid.py --skip-cmmd
    python experiments/sweep_all_layers_kadid.py --backbones sd_vae dc_ae
    python experiments/sweep_all_layers_kadid.py --device cpu
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
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gram_mmd.config import get_backbone_config
from gram_mmd.features import FeatureHook, ImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR   = Path(__file__).resolve().parent.parent
KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
OUT_DIR    = BASE_DIR / "results" / "sweep_all_layers"

# (backbone_name, display_name, batch_size_for_extraction)
BACKBONES_TO_SWEEP = [
    ("sd_vae",        "SD-VAE",    16),
    ("dc_ae",         "DC-AE",      4),
    ("dinov2_vitb14", "DINOv2-B",   8),
]

# Cap on number of channels used for Gram computation.
# Limits memory for very wide layers (e.g. DC-AE stage 4-5: 1024ch).
# 512 → upper-tri dim = 131 328 floats/image, anchor storage ~43 MB/layer.
MAX_CHANNELS = 512

# RBF kernel gamma — same fixed value as evaluate_kadid_monotonicity.py.
# None = median heuristic (adaptatif par couche mais sous-optimal empiriquement).
SWEEP_GAMMA = 2.8e-5

# kNN-MMD: values of k to sweep. 81 = all anchors = global MMD² (baseline).
K_VALUES = [1, 5, 10, 25, 81]

# CMMD
ENABLE_CMMD   = True
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005
BATCH_SIZE_CLIP = 32

# Visual style per backbone
STYLE = {
    "SD-VAE":   {"color": "#1f77b4", "marker": "o"},
    "DC-AE":    {"color": "#d62728", "marker": "s"},
    "DINOv2-B": {"color": "#2ca02c", "marker": "^"},
}


# ============================================================
# Multi-Layer Gram Extractor
# ============================================================

class MultiLayerGramExtractor:
    """
    Loads a backbone ONCE and extracts Gram features from ALL configured
    layers in a single forward pass per batch.

    Returns a dict {layer_idx: gram_array (B, D_l)} per batch.
    """

    def __init__(self, backbone_name: str, device: str,
                 max_channels: int = MAX_CHANNELS):
        self.backbone_name = backbone_name
        self.device = device
        self.max_channels = max_channels
        self.backbone_cfg = get_backbone_config(backbone_name)

        self.model = None
        self.hooks: dict = {}        # layer_idx -> FeatureHook
        self.hook_handles: list = []

        self._load_model()
        self._register_all_hooks()
        self.image_transform = self._make_transform()

    def _load_model(self):
        cfg = self.backbone_cfg
        name = self.backbone_name
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
            raise ValueError(f"Backbone not supported in sweep: {name}")

        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _get_module(self, dotted_name: str) -> nn.Module:
        """Resolve a dotted layer name (e.g. 'encoder.down_blocks.0.2')."""
        module = self.model
        for part in dotted_name.split("."):
            try:
                module = getattr(module, part)
            except AttributeError:
                module = module[int(part)]
        return module

    def _register_all_hooks(self):
        for layer_idx, layer_name in self.backbone_cfg["layer_names"].items():
            try:
                mod = self._get_module(layer_name)
                hook = FeatureHook()
                handle = mod.register_forward_hook(hook)
                self.hooks[layer_idx] = hook
                self.hook_handles.append(handle)
            except Exception as e:
                logger.warning(
                    f"    Could not hook layer {layer_idx} ({layer_name}): {e}"
                )
        logger.info(f"  Hooks registered: {len(self.hooks)} / "
                    f"{len(self.backbone_cfg['layer_names'])} layers")

    def _make_transform(self):
        from torchvision import transforms
        cfg = self.backbone_cfg
        return transforms.Compose([
            transforms.Resize((cfg["input_size"], cfg["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["normalize_mean"],
                                 std=cfg["normalize_std"]),
        ])

    def _forward(self, images: torch.Tensor):
        images = images.to(self.device)
        for hook in self.hooks.values():
            hook.clear()
        with torch.no_grad():
            if self.backbone_name in ("sd_vae", "flux_vae", "dc_ae"):
                _ = self.model.encoder(images)
            else:
                _ = self.model(images)

    def _gram_from_hook(self, hook: FeatureHook) -> np.ndarray | None:
        """
        Compute spatial Gram matrix (Gatys et al.) from hook activation.
        Optionally subsamples channels if C > max_channels.
        Returns float32 array (B, C*(C+1)/2).
        """
        feat = hook.features
        if feat is None:
            return None

        feat = feat.float()

        if feat.dim() == 4:
            B, C, H, W = feat.shape
            if C > self.max_channels:
                gen = torch.Generator(device="cpu").manual_seed(42)
                idx = torch.randperm(C, generator=gen)[:self.max_channels]
                feat = feat[:, idx.to(feat.device), :, :]
                C = self.max_channels
            F = feat.reshape(B, C, H * W)

        elif feat.dim() == 3:
            B, N, D = feat.shape
            if D > self.max_channels:
                gen = torch.Generator(device="cpu").manual_seed(42)
                idx = torch.randperm(D, generator=gen)[:self.max_channels]
                feat = feat[:, :, idx.to(feat.device)]
                D = self.max_channels
            F = feat.permute(0, 2, 1)   # (B, D, N) → treat D as "channels"
            C = D

        elif feat.dim() == 2:
            # Already compact (e.g. attention output): return as-is
            return feat.cpu().numpy().astype(np.float32)

        else:
            return None

        S = F.shape[2]                                           # spatial size
        G = torch.bmm(F, F.transpose(1, 2)) / S                 # (B, C, C)
        triu_mask = torch.triu(
            torch.ones(C, C, device=feat.device, dtype=torch.bool)
        )
        gram = G[:, triu_mask]                                   # (B, D_gram)
        return gram.cpu().numpy().astype(np.float32)

    def extract_gram_batch(self, images: torch.Tensor) -> dict:
        """
        Single forward pass → per-layer Gram features.
        Returns {layer_idx: float32 array (B, D_l)}.
        """
        self._forward(images)
        result = {}
        for layer_idx, hook in self.hooks.items():
            g = self._gram_from_hook(hook)
            if g is not None:
                result[layer_idx] = g
        return result

    def extract_anchor_grams(self, image_paths: list, batch_size: int) -> dict:
        """
        Extract Gram features for all layers from anchor images.
        Returns {layer_idx: float32 array (N_anchor, D_l)}.
        """
        dataset = ImageDataset(image_paths, transform=self.image_transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=(self.device == "cuda"),
            persistent_workers=True,
        )
        layer_grams: dict = defaultdict(list)
        for batch_imgs, _ in loader:
            for lid, g in self.extract_gram_batch(batch_imgs).items():
                layer_grams[lid].append(g)

        return {lid: np.concatenate(gs, axis=0) for lid, gs in layer_grams.items()}

    def cleanup(self):
        for h in self.hook_handles:
            h.remove()
        self.hooks.clear()
        self.hook_handles.clear()
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# Anchor statistics (gamma via median heuristic + K_rr)
# ============================================================

def build_anchor_stats(anchor_grams: dict) -> dict:
    """
    For each layer: fit StandardScaler, compute gamma (median heuristic),
    mean_Krr, ref_sq — exactly like FeatureExtractor.transform_features().
    Returns {layer_idx: {"ref", "ref_sq", "gamma", "mean_Krr", "scaler"}}.
    """
    from sklearn.preprocessing import StandardScaler

    stats = {}
    for lid, ref in anchor_grams.items():
        # 1. Fit StandardScaler — same as FeatureExtractor.transform_features()
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(ref).astype(np.float32)

        # 2. Gamma — fixed value matching evaluate_kadid_monotonicity.py,
        #    or median heuristic if SWEEP_GAMMA is None
        ref_sq = np.einsum("ij,ij->i", ref_scaled, ref_scaled)
        D2 = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref_scaled @ ref_scaled.T)
        if SWEEP_GAMMA is None:
            triu_vals = D2[np.triu_indices(len(ref_scaled), k=1)]
            gamma = 1.0 / (2.0 * float(np.median(triu_vals)) + 1e-8)
        else:
            gamma = SWEEP_GAMMA
        K_rr = np.exp(-gamma * D2)
        stats[lid] = {
            "ref":       ref_scaled,
            "ref_sq":    ref_sq,
            "gamma":     gamma,
            "mean_Krr":  float(np.mean(K_rr)),
            "K_rr_full": K_rr,   # (n_anchor, n_anchor) — used by kNN-MMD
            "scaler":    scaler,
        }
    logger.info(f"  Anchor stats ready for {len(stats)} layers")
    return stats


# ============================================================
# kNN-MMD scoring helpers
# ============================================================

def _compute_D2_gpu(query_feats: np.ndarray, ref_feats: np.ndarray,
                    ref_sq: np.ndarray, device: str) -> np.ndarray:
    """
    Squared Euclidean distance matrix (n_query, n_anchor) computed on GPU.
    Falls back to CPU if device is 'cpu'.
    """
    q_t = torch.from_numpy(query_feats).to(device)
    r_t = torch.from_numpy(ref_feats).to(device)
    r_sq_t = torch.from_numpy(ref_sq).to(device)
    with torch.no_grad():
        q_sq_t = (q_t * q_t).sum(dim=1)
        D2_t = torch.clamp(
            q_sq_t[:, None] + r_sq_t[None, :] - 2.0 * (q_t @ r_t.T),
            min=0.0,
        )
    return D2_t.cpu().numpy()


def score_all_layers_knn_mmd(
    extractor: MultiLayerGramExtractor,
    anchor_stats: dict,
    dist_paths: list,
    batch_size: int,
    k_values: list,
    device: str,
) -> dict:
    """
    Score all eval images for all layers × all k values in one streaming pass.

    kNN-MMD² (biased):
        MMD²_k(y, X) = mean(K_rr^k)  +  1  -  (2/k) * Σ_{i ∈ kNN(y,X)} k(x_i, y)

    Returns {(layer_idx, k): float64 scores array (N,)}.
    GPU is used for the costly D2_qr distance-matrix computation.
    """
    n = len(dist_paths)
    result: dict = {}
    for lid in anchor_stats:
        for k in k_values:
            result[(lid, k)] = np.empty(n, dtype=np.float64)

    dataset = ImageDataset(dist_paths, transform=extractor.image_transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(extractor.device == "cuda"),
        persistent_workers=True,
    )

    idx = 0
    t0 = time.time()
    for batch_imgs, _ in loader:
        gram_dict = extractor.extract_gram_batch(batch_imgs)
        b = batch_imgs.shape[0]

        for lid, g in gram_dict.items():
            if lid not in anchor_stats:
                continue
            st = anchor_stats[lid]

            # Scale query features with the anchor-fitted scaler
            g32 = st["scaler"].transform(g).astype(np.float32)

            # GPU: (b, n_anchor) squared distance matrix
            D2_qr = _compute_D2_gpu(g32, st["ref"], st["ref_sq"], device)
            K_qr  = np.exp(-st["gamma"] * D2_qr)   # (b, n_anchor)

            K_rr_full = st["K_rr_full"]             # (n_anchor, n_anchor)
            n_anchor  = st["ref"].shape[0]

            for k in k_values:
                k_eff = min(k, n_anchor)
                scores_k = np.empty(b, dtype=np.float64)
                for i in range(b):
                    # k nearest anchors to query i (by squared L2 in feature space)
                    nn_idx = np.argpartition(D2_qr[i], k_eff - 1)[:k_eff]
                    K_rr_k = K_rr_full[np.ix_(nn_idx, nn_idx)]
                    term1  = float(K_rr_k.mean())
                    term3  = 2.0 / k_eff * float(K_qr[i, nn_idx].sum())
                    scores_k[i] = term1 + 1.0 - term3
                result[(lid, k)][idx: idx + b] = scores_k

        idx += b
        if idx % 1000 < batch_size or idx >= n:
            logger.info(
                f"    Scored {idx}/{n} ({idx/n:.0%}) — {time.time()-t0:.1f}s"
            )
        del gram_dict
        gc.collect()

    return result


# ============================================================
# CMMD (CLIP ViT-L/14@336)
# ============================================================

def build_clip_anchor(anchor_paths: list, device: str):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    logger.info(f"  Loading {CLIP_MODEL_ID} ...")
    model = (
        CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
        .to(device)
        .eval()
    )
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    embeds = []
    for i in range(0, len(anchor_paths), BATCH_SIZE_CLIP):
        batch = [Image.open(p).convert("RGB")
                 for p in anchor_paths[i: i + BATCH_SIZE_CLIP]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            embeds.append(
                model(pixel_values=pv).image_embeds.float().cpu().numpy()
            )

    ref = np.concatenate(embeds, axis=0).astype(np.float32)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    K_rr = np.exp(
        -CMMD_GAMMA * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T))
    )
    mean_Krr = float(np.mean(K_rr))
    logger.info(f"  CLIP anchor: {ref.shape}, mean_Krr={mean_Krr:.4f}")
    return model, proc, ref, ref_sq, mean_Krr


def score_cmmd_streaming(model, proc, ref, ref_sq, mean_Krr,
                         dist_paths: list, device: str) -> np.ndarray:
    n = len(dist_paths)
    scores = np.empty(n, dtype=np.float64)
    chunk = BATCH_SIZE_CLIP * 4
    t0 = time.time()

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        batch = [Image.open(p).convert("RGB") for p in dist_paths[s:e]]
        pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            emb = (
                model(pixel_values=pv).image_embeds.float().cpu().numpy()
                .astype(np.float32)
            )
        t_sq = np.einsum("ij,ij->i", emb, emb)
        K_rt = np.exp(
            -CMMD_GAMMA * (
                ref_sq[:, None] + t_sq[None, :] - 2.0 * (ref @ emb.T)
            )
        )
        scores[s:e] = mean_Krr + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        if s % 1000 < chunk or e == n:
            logger.info(
                f"    CMMD {e}/{n} ({e/n:.0%}) — {time.time()-t0:.1f}s"
            )
        del emb, K_rt

    return scores


# ============================================================
# Statistics
# ============================================================

def load_kadid_csv() -> list:
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
    logger.info(f"KADID CSV: {len(rows)} images")
    return rows


def analyze_scores(rows: list, scores: np.ndarray) -> dict:
    """Compute Spearman, Kendall, and monotonicity metrics."""
    degradation = np.array([5.0 - r["dmos"] for r in rows])
    levels = np.array([r["level"] for r in rows])

    sp,   sp_p  = spearmanr(scores, degradation)
    kt,   kt_p  = kendalltau(scores, degradation)
    sp_lv, _    = spearmanr(scores, levels)

    # Monotonicity: per (ref_id, dist_type) group, levels 1..5
    groups: dict = defaultdict(dict)
    for i, row in enumerate(rows):
        groups[(row["ref_id"], row["dist_type"])][row["level"]] = scores[i]

    n_total = n_strict = n_partial = 0
    for lvl_scores in groups.values():
        if len(lvl_scores) < 5:
            continue
        seq = [lvl_scores[lv] for lv in sorted(lvl_scores)]
        n_total += 1
        if all(seq[i] < seq[i + 1] for i in range(4)):
            n_strict += 1
        if sum(1 for i in range(4) if seq[i] < seq[i + 1]) >= 3:
            n_partial += 1

    by_level: dict = defaultdict(list)
    for i, row in enumerate(rows):
        by_level[row["level"]].append(scores[i])

    return {
        "spearman":       float(sp)   if not np.isnan(sp)   else 0.0,
        "spearman_p":     float(sp_p),
        "kendall":        float(kt)   if not np.isnan(kt)   else 0.0,
        "kendall_p":      float(kt_p),
        "spearman_level": float(sp_lv) if not np.isnan(sp_lv) else 0.0,
        "mono_strict":    n_strict,
        "mono_partial":   n_partial,
        "n_total":        n_total,
        "rate_strict":    n_strict  / n_total if n_total else 0.0,
        "rate_partial":   n_partial / n_total if n_total else 0.0,
        "by_level": {
            lv: float(np.mean(v)) for lv, v in sorted(by_level.items())
        },
    }


# ============================================================
# Plots
# ============================================================

def plot_spearman_per_layer(
    backbone_results: dict,
    cmmd_spearman: float | None,
    out_path: Path,
    k_values: list,
):
    """
    One subplot per backbone: Spearman per layer, one line per k value.
    CMMD shown as horizontal dashed line.
    """
    n_bb = sum(1 for v in backbone_results.values() if v)
    if n_bb == 0:
        return
    fig, axes = plt.subplots(1, n_bb, figsize=(7 * n_bb, 6), sharey=True)
    if n_bb == 1:
        axes = [axes]

    cmap = plt.cm.plasma
    k_colors = {k: cmap(i / max(len(k_values) - 1, 1))
                for i, k in enumerate(k_values)}

    ax_iter = iter(axes)
    for display_name, layer_k_stats in backbone_results.items():
        if not layer_k_stats:
            continue
        ax = next(ax_iter)
        style = STYLE.get(display_name, {})

        for k in k_values:
            layers = sorted(lid for (lid, kk) in layer_k_stats if kk == k)
            if not layers:
                continue
            sp_vals = [layer_k_stats[(l, k)]["spearman"] for l in layers]
            ax.plot(
                layers, sp_vals,
                color=k_colors[k],
                marker=style.get("marker", "o"),
                label=f"k={k}",
                linewidth=1.5, markersize=4,
            )

        if cmmd_spearman is not None:
            ax.axhline(
                cmmd_spearman, color="purple", linestyle="--", linewidth=1.5,
                label=f"CMMD ({cmmd_spearman:.3f})",
            )

        ax.set_title(display_name, fontsize=12)
        ax.set_xlabel("Index de couche", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Corrélation de Spearman", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.15, 1.05)

    fig.suptitle(
        "kNN-MMD² Spearman par couche — KADID-10k (Gram spatial)",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def plot_best_model_monotonicity(
    backbone_results: dict,
    cmmd_by_level: dict | None,
    out_path: Path,
):
    """
    For each backbone's best (layer, k) combo, show normalised mean kNN-MMD
    score vs degradation level (1=light → 5=severe). Ideal: strictly increasing.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = [1, 2, 3, 4, 5]

    for display_name, layer_k_stats in backbone_results.items():
        if not layer_k_stats:
            continue
        style = STYLE.get(display_name, {})
        best_key = max(layer_k_stats.keys(),
                       key=lambda key: layer_k_stats[key]["spearman"])
        best_lid, best_k = best_key
        by_level = layer_k_stats[best_key]["by_level"]
        sp = layer_k_stats[best_key]["spearman"]
        v = np.array([by_level.get(lv, np.nan) for lv in levels], dtype=float)
        span = v.max() - v.min()
        v_norm = (v - v.min()) / (span + 1e-10)
        ax.plot(
            levels, v_norm,
            marker=style.get("marker", "o"),
            color=style.get("color"),
            label=f"{display_name} L{best_lid} k={best_k} (ρ={sp:.3f})",
            linewidth=2, markersize=7,
        )

    if cmmd_by_level:
        v = np.array([cmmd_by_level.get(lv, np.nan) for lv in levels], dtype=float)
        span = v.max() - v.min()
        v_norm = (v - v.min()) / (span + 1e-10)
        ax.plot(
            levels, v_norm,
            marker="D", linestyle="--", color="purple",
            label="CMMD", linewidth=2, markersize=7,
        )

    ax.set_xlabel("Niveau de dégradation (1 = léger → 5 = sévère)", fontsize=12)
    ax.set_ylabel("Score normalisé [0, 1]", fontsize=12)
    ax.set_title(
        "Monotonie du meilleur modèle par backbone — KADID-10k (kNN-MMD²)",
        fontsize=13,
    )
    ax.set_xticks(levels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ============================================================
# Results saving + console summary
# ============================================================

def save_csv(backbone_results: dict, cmmd_stats: dict | None, out_path: Path):
    rows = []
    for display_name, layer_k_stats in backbone_results.items():
        for (layer_idx, k), st in sorted(layer_k_stats.items()):
            rows.append({
                "backbone":          display_name,
                "layer":             layer_idx,
                "k":                 k,
                "spearman":          st["spearman"],
                "kendall":           st["kendall"],
                "spearman_level":    st["spearman_level"],
                "mono_strict_rate":  st["rate_strict"],
                "mono_partial_rate": st["rate_partial"],
                "mono_strict_n":     st["mono_strict"],
                "n_total":           st["n_total"],
                "score_lv1": st["by_level"].get(1, ""),
                "score_lv2": st["by_level"].get(2, ""),
                "score_lv3": st["by_level"].get(3, ""),
                "score_lv4": st["by_level"].get(4, ""),
                "score_lv5": st["by_level"].get(5, ""),
            })
    if cmmd_stats:
        st = cmmd_stats
        rows.append({
            "backbone": "CMMD", "layer": -1,
            "spearman":          st["spearman"],
            "kendall":           st["kendall"],
            "spearman_level":    st["spearman_level"],
            "mono_strict_rate":  st["rate_strict"],
            "mono_partial_rate": st["rate_partial"],
            "mono_strict_n":     st["mono_strict"],
            "n_total":           st["n_total"],
            "score_lv1": st["by_level"].get(1, ""),
            "score_lv2": st["by_level"].get(2, ""),
            "score_lv3": st["by_level"].get(3, ""),
            "score_lv4": st["by_level"].get(4, ""),
            "score_lv5": st["by_level"].get(5, ""),
        })
    if not rows:
        logger.warning("No results to save.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"  CSV saved: {out_path}")


def print_summary(backbone_results: dict, cmmd_stats: dict | None):
    SEP = "=" * 115
    logger.info("\n" + SEP)
    logger.info("RÉSUMÉ — KADID-10k Sweep All Layers (kNN-MMD²)")
    logger.info(SEP)
    logger.info(
        f"  {'Backbone':<20} {'L':>3}  {'k':>4}  {'Spearman':>9}  {'Kendall':>8}  "
        f"{'Mono.strict':>13}  {'Mono.>=3':>11}"
    )
    logger.info("-" * 115)

    for display_name, layer_k_stats in backbone_results.items():
        if not layer_k_stats:
            continue
        best_key = max(layer_k_stats, key=lambda key: layer_k_stats[key]["spearman"])
        for (layer_idx, k), st in sorted(layer_k_stats.items()):
            star = " ★" if (layer_idx, k) == best_key else ""
            logger.info(
                f"  {display_name:<20} {layer_idx:>3}  {k:>4}  "
                f"{st['spearman']:>+9.4f}  {st['kendall']:>+8.4f}  "
                f"{st['mono_strict']:>5}/{st['n_total']:<4} ({st['rate_strict']:>5.1%})  "
                f"{st['mono_partial']:>5}/{st['n_total']:<4} ({st['rate_partial']:>5.1%})"
                f"{star}"
            )

    if cmmd_stats:
        st = cmmd_stats
        logger.info(
            f"  {'CMMD':<20} {'—':>3}  {'—':>4}  "
            f"{st['spearman']:>+9.4f}  {st['kendall']:>+8.4f}  "
            f"{st['mono_strict']:>5}/{st['n_total']:<4} ({st['rate_strict']:>5.1%})  "
            f"{st['mono_partial']:>5}/{st['n_total']:<4} ({st['rate_partial']:>5.1%})"
        )
    logger.info(SEP)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep all layers on KADID-10k — sd_vae / dc_ae / dinov2"
    )
    p.add_argument("--device",    default=None,
                   help="cuda / cpu (auto-detect if omitted)")
    p.add_argument("--skip-cmmd", action="store_true",
                   help="Skip CMMD baseline")
    p.add_argument("--backbones", nargs="+", default=None,
                   help="Restrict to a subset e.g. sd_vae dc_ae")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    rows = load_kadid_csv()
    ref_names   = sorted({r["ref_img"]   for r in rows})
    anchor_paths = [str(IMAGES_DIR / n)            for n in ref_names]
    dist_paths   = [str(IMAGES_DIR / r["dist_img"]) for r in rows]
    logger.info(f"{len(anchor_paths)} références  |  {len(dist_paths)} images d'éval")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    backbones_to_run = BACKBONES_TO_SWEEP
    if args.backbones:
        selected = set(args.backbones)
        backbones_to_run = [
            (n, d, b) for n, d, b in BACKBONES_TO_SWEEP if n in selected
        ]

    # ── Per-backbone layer sweep ─────────────────────────────────────────
    backbone_results: dict = {}

    for backbone_name, display_name, batch_size in backbones_to_run:
        logger.info("\n" + "=" * 80)
        logger.info(f"BACKBONE: {backbone_name}  ({display_name})  batch={batch_size}")
        logger.info("=" * 80)

        layer_stats: dict = {}
        ext = None
        try:
            ext = MultiLayerGramExtractor(
                backbone_name, device, max_channels=MAX_CHANNELS
            )

            # -- Anchor: extract gram for all layers in one pass
            t0 = time.time()
            anchor_grams = ext.extract_anchor_grams(anchor_paths, batch_size)
            logger.info(f"  Anchor extraction: {time.time()-t0:.1f}s")

            # -- Build gamma + K_rr per layer
            anchor_stats = build_anchor_stats(anchor_grams)
            del anchor_grams
            gc.collect()

            # -- Eval: stream all 10 125 images, score all layers × all k values
            t1 = time.time()
            all_scores = score_all_layers_knn_mmd(
                ext, anchor_stats, dist_paths, batch_size, K_VALUES, device
            )
            logger.info(f"  Streaming kNN-MMD scoring: {time.time()-t1:.1f}s")

            # -- Analyse each (layer, k) combination
            for (lid, k), scores in sorted(all_scores.items()):
                st = analyze_scores(rows, scores)
                layer_stats[(lid, k)] = st
                logger.info(
                    f"  Layer {lid:2d} k={k:>3}: Spearman={st['spearman']:+.4f}  "
                    f"Mono.strict={st['rate_strict']:.1%}"
                )

        except Exception as exc:
            logger.error(f"  ERROR {backbone_name}: {exc}", exc_info=True)
        finally:
            if ext is not None:
                ext.cleanup()
            torch.cuda.empty_cache()
            gc.collect()

        backbone_results[display_name] = layer_stats

    # ── CMMD baseline ────────────────────────────────────────────────────
    cmmd_stats = None
    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info("\n" + "=" * 80)
        logger.info(f"CMMD — {CLIP_MODEL_ID}")
        logger.info("=" * 80)
        try:
            clip_model, clip_proc, clip_ref, clip_ref_sq, clip_mean_Krr = \
                build_clip_anchor(anchor_paths, device)
            scores_clip = score_cmmd_streaming(
                clip_model, clip_proc, clip_ref, clip_ref_sq,
                clip_mean_Krr, dist_paths, device,
            )
            del clip_model, clip_proc, clip_ref, clip_ref_sq
            torch.cuda.empty_cache()
            gc.collect()

            cmmd_stats = analyze_scores(rows, scores_clip)
            logger.info(
                f"  CMMD Spearman={cmmd_stats['spearman']:+.4f}  "
                f"Mono.strict={cmmd_stats['rate_strict']:.1%}"
            )
        except Exception as exc:
            logger.error(f"  CMMD ERROR: {exc}", exc_info=True)
    else:
        logger.info("CMMD skipped")

    # ── Summary + output ─────────────────────────────────────────────────
    print_summary(backbone_results, cmmd_stats)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(backbone_results, cmmd_stats, OUT_DIR / f"sweep_results_{ts}.csv")

    cmmd_sp = cmmd_stats["spearman"]   if cmmd_stats else None
    cmmd_lv = cmmd_stats["by_level"]   if cmmd_stats else None

    plot_spearman_per_layer(
        backbone_results, cmmd_sp,
        OUT_DIR / f"sweep_spearman_per_layer_{ts}.png",
        K_VALUES,
    )
    plot_best_model_monotonicity(
        backbone_results, cmmd_lv,
        OUT_DIR / f"sweep_best_model_monotonicity_{ts}.png",
    )

    logger.info(f"\nTous les résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
