"""
compare_gram_logeucl_kadid.py
==============================
Comparaison sur KADID-10k de deux distances point-vs-distribution :

  1. Gram vectorisé  + RBF γ=2.8e-5 (approche courante)
     feature = vec(triu(G))  avec G = F Fᵀ / S (C×C)

  2. MMDlogE          + RBF γ=médiane  (Log-Euclidean sur SPD)
     feature = vec(triu(log(G + εI)))   via décomposition propre

Ancrage = 81 images de référence KADID (identiques au sweep).
Éval    = 10 125 images dégradées, scoring N=1 biaisé en streaming.

Sorties → results/compare_gram_logeucl/
  violin_par_niveau_<ts>.png   — violin + strip par niveau 1→5
  scatter_dmos_<ts>.png        — nuage (dmos, score) coloré par niveau
  summary_<ts>.csv             — Spearman, Kendall, mono par métrique

Usage :
    python experiments/compare_gram_logeucl_kadid.py
    python experiments/compare_gram_logeucl_kadid.py --no-cache
    python experiments/compare_gram_logeucl_kadid.py --device cpu
    python experiments/compare_gram_logeucl_kadid.py --skip-cmmd
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.config import get_backbone_config
from gram_mmd.features import FeatureHook, ImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
OUT_DIR    = BASE_DIR / "results" / "compare_gram_logeucl"
CACHE_DIR  = OUT_DIR / "cache"

# (backbone, layer_idx, display_name, batch_size)
BACKBONE_CONFIGS = [
    ("sd_vae",        7,  "SD-VAE l7",   16),
    ("dc_ae",        10,  "DC-AE l10",    4),
    ("dinov2_vitb14", 5,  "DINOv2-B l5",  8),
]

MAX_CHANNELS = 512      # cap canaux pour Gram + LogGram
GAMMA_RBF    = 2.8e-5   # gamma fixé pour la branche Gram vectorisé
LOG_EPS      = 1e-6     # regularisation G_reg = G + eps·I

# CMMD baseline
ENABLE_CMMD     = True
CLIP_MODEL_ID   = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA      = 0.005
BATCH_SIZE_CLIP = 32

LEVELS = [1, 2, 3, 4, 5]
LEVEL_COLORS = {1: "#2166ac", 2: "#74add1", 3: "#fdae61",
                4: "#f46d43", 5: "#d73027"}


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
    logger.info(f"KADID : {len(rows)} images")
    return rows


# ============================================================
# Feature extraction (gram vectorisé + log-gram, en parallèle)
# ============================================================

def _get_module(model, dotted_name):
    m = model
    for part in dotted_name.split("."):
        try:
            m = getattr(m, part)
        except AttributeError:
            m = m[int(part)]
    return m


def _load_model(backbone_name, device):
    cfg = get_backbone_config(backbone_name)
    if backbone_name in ("sd_vae", "flux_vae"):
        from diffusers import AutoencoderKL
        model = AutoencoderKL.from_pretrained(cfg["weights"])
    elif backbone_name == "dc_ae":
        from diffusers import AutoencoderDC
        model = AutoencoderDC.from_pretrained(cfg["weights"])
    elif backbone_name == "dinov2_vitb14":
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
    else:
        raise ValueError(f"Backbone non supporté: {backbone_name}")
    return model.to(device).eval()


def _make_transform(backbone_name):
    from torchvision import transforms
    cfg = get_backbone_config(backbone_name)
    return transforms.Compose([
        transforms.Resize((cfg["input_size"], cfg["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["normalize_mean"], std=cfg["normalize_std"]),
    ])


def _gram_vec(feat: torch.Tensor, max_channels: int) -> np.ndarray | None:
    """Gram vectorisé — upper triangle de G = F Fᵀ / S."""
    feat = feat.float()
    if feat.dim() == 4:
        B, C, H, W = feat.shape
        if C > max_channels:
            g = torch.Generator(device="cpu").manual_seed(42)
            idx = torch.randperm(C, generator=g)[:max_channels]
            feat = feat[:, idx.to(feat.device), :, :]
            C = max_channels
        F = feat.reshape(B, C, H * W)
    elif feat.dim() == 3:
        B, N, D = feat.shape
        if D > max_channels:
            g = torch.Generator(device="cpu").manual_seed(42)
            idx = torch.randperm(D, generator=g)[:max_channels]
            feat = feat[:, :, idx.to(feat.device)]
            D = max_channels
        F = feat.permute(0, 2, 1)  # (B, D, N)
        C = D
    elif feat.dim() == 2:
        return feat.cpu().numpy().astype(np.float32)
    else:
        return None

    S = F.shape[2]
    G = torch.bmm(F, F.transpose(1, 2)) / S  # (B, C, C)
    triu = torch.triu(torch.ones(C, C, device=feat.device, dtype=torch.bool))
    return G[:, triu].cpu().numpy().astype(np.float32)


def _log_gram_vec(feat: torch.Tensor, max_channels: int,
                  eps: float = LOG_EPS) -> np.ndarray | None:
    """
    Log-Euclidean Gram — upper triangle de log(G + εI).

    G_reg = G + εI   (rend G strictement SPD)
    log(G_reg) = V diag(log λ) Vᵀ  via np.linalg.eigh
    """
    feat = feat.float()
    if feat.dim() == 4:
        B, C, H, W = feat.shape
        if C > max_channels:
            g = torch.Generator(device="cpu").manual_seed(42)
            idx = torch.randperm(C, generator=g)[:max_channels]
            feat = feat[:, idx.to(feat.device), :, :]
            C = max_channels
        F = feat.reshape(B, C, H * W).cpu().numpy()
        S = H * W
    elif feat.dim() == 3:
        B, N, D = feat.shape
        if D > max_channels:
            g = torch.Generator(device="cpu").manual_seed(42)
            idx = torch.randperm(D, generator=g)[:max_channels]
            feat = feat[:, :, idx.to(feat.device)]
            D = max_channels
        F = feat.permute(0, 2, 1).cpu().numpy()  # (B, D, N)
        C = D
        S = N
    elif feat.dim() == 2:
        return None   # pas de structure spatiale — log-gram non applicable
    else:
        return None

    I_C     = np.eye(C, dtype=np.float64)
    triu_i, triu_j = np.triu_indices(C)
    log_vecs = []

    for i in range(B):
        G     = (F[i].astype(np.float64) @ F[i].astype(np.float64).T) / S
        G_reg = G + eps * I_C
        eigvals, eigvecs = np.linalg.eigh(G_reg)         # λ triés croissant
        eigvals = np.maximum(eigvals, 1e-12)              # sécurité numérique
        log_G   = (eigvecs * np.log(eigvals)) @ eigvecs.T # log(G_reg)
        log_vecs.append(log_G[triu_i, triu_j].astype(np.float32))

    return np.array(log_vecs, dtype=np.float32)


# ============================================================
# Construction cache features pour ancrage + éval
# ============================================================

def extract_all_features(backbone_name, layer_idx, batch_size, device,
                         image_paths):
    """
    Extrait gram vectorisé et log-gram pour toutes les images données.
    Retourne (gram_arr, log_gram_arr) de shape (N, D_gram) et (N, D_gram).
    """
    cfg = get_backbone_config(backbone_name)
    model = _load_model(backbone_name, device)
    transform = _make_transform(backbone_name)

    layer_name = cfg["layer_names"][layer_idx]
    mod  = _get_module(model, layer_name)
    hook = FeatureHook()
    handle = mod.register_forward_hook(hook)

    dataset = ImageDataset(image_paths, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=(device == "cuda"),
                         persistent_workers=True)

    gram_list     = []
    log_gram_list = []
    n = len(image_paths)
    t0 = time.time()

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            hook.clear()
            if backbone_name in ("sd_vae", "flux_vae", "dc_ae"):
                _ = model.encoder(imgs)
            else:
                _ = model(imgs)

            gv  = _gram_vec(hook.features, MAX_CHANNELS)
            lgv = _log_gram_vec(hook.features, MAX_CHANNELS)
            if gv is not None:
                gram_list.append(gv)
            if lgv is not None:
                log_gram_list.append(lgv)

        idx_done = sum(len(x) for x in gram_list)
        if idx_done % 2000 < batch_size or idx_done >= n:
            logger.info(f"    {idx_done}/{n} — {time.time()-t0:.1f}s")

    handle.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    gram     = np.concatenate(gram_list, axis=0)     if gram_list     else None
    log_gram = np.concatenate(log_gram_list, axis=0) if log_gram_list else None
    return gram, log_gram


# ============================================================
# Scoring MMD N=1 biaisé
# ============================================================

def _build_anchor_stats_gram(ref_gram):
    """Gram vec + StandardScaler + RBF γ=2.8e-5."""
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    ref_sc  = scaler.fit_transform(ref_gram).astype(np.float32)
    ref_sq  = np.einsum("ij,ij->i", ref_sc, ref_sc)
    K_rr    = np.exp(-GAMMA_RBF * (ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref_sc @ ref_sc.T)))
    return {"ref": ref_sc, "ref_sq": ref_sq, "mean_Krr": float(K_rr.mean()),
            "gamma": GAMMA_RBF, "scaler": scaler}


def _build_anchor_stats_loggram(ref_log):
    """Log-gram + pas de scaler + RBF γ=heuristique médiane."""
    ref = ref_log.astype(np.float32)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    D2     = ref_sq[:, None] + ref_sq[None, :] - 2.0 * (ref @ ref.T)
    triu   = D2[np.triu_indices(len(ref), k=1)]
    gamma  = 1.0 / (2.0 * float(np.median(triu)) + 1e-8)
    K_rr   = np.exp(-gamma * np.maximum(D2, 0))
    logger.info(f"  LogGram γ médiane = {gamma:.4e}")
    return {"ref": ref, "ref_sq": ref_sq, "mean_Krr": float(K_rr.mean()), "gamma": gamma}


def _score_mmd_n1(query, st):
    """Score MMD² N=1 biaisé pour un batch de requêtes."""
    q = query.astype(np.float32)
    if "scaler" in st:
        q = st["scaler"].transform(q).astype(np.float32)
    q_sq = np.einsum("ij,ij->i", q, q)
    K_rt = np.exp(-st["gamma"] * (st["ref_sq"][:, None] + q_sq[None, :] - 2.0 * (st["ref"] @ q.T)))
    return st["mean_Krr"] + 1.0 - 2.0 * np.mean(K_rt, axis=0)


# ============================================================
# Évaluation complète d'un backbone
# ============================================================

def evaluate_backbone(backbone_name, layer_idx, display_name, batch_size,
                      anchor_paths, dist_paths, device, use_cache):
    """
    Retourne {"gram_rbf": scores_array, "loggram_med": scores_array}.
    """
    tag = display_name.replace(" ", "_").replace("/", "-")
    cache_gram    = CACHE_DIR / f"{tag}_gram.npy"
    cache_loggram = CACHE_DIR / f"{tag}_loggram.npy"

    # ── Cache anchor gram + log-gram ──────────────────────────────────
    cache_ref_gram    = CACHE_DIR / f"{tag}_ref_gram.npy"
    cache_ref_loggram = CACHE_DIR / f"{tag}_ref_loggram.npy"

    if use_cache and cache_gram.exists() and cache_loggram.exists():
        logger.info(f"  Cache scores trouvé pour {display_name}")
        return {
            "gram_rbf":    np.load(cache_gram),
            "loggram_med": np.load(cache_loggram),
        }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Ancrage ───────────────────────────────────────────────────────
    if use_cache and cache_ref_gram.exists() and cache_ref_loggram.exists():
        logger.info(f"  Cache ancrage trouvé pour {display_name}")
        ref_gram    = np.load(cache_ref_gram)
        ref_loggram = np.load(cache_ref_loggram)
    else:
        logger.info(f"  Extraction ancrage ({len(anchor_paths)} images)…")
        ref_gram, ref_loggram = extract_all_features(
            backbone_name, layer_idx, batch_size, device, anchor_paths
        )
        np.save(cache_ref_gram,    ref_gram)
        np.save(cache_ref_loggram, ref_loggram)
        logger.info(f"  Ancrage: gram={ref_gram.shape}  loggram={ref_loggram.shape if ref_loggram is not None else 'None'}")

    st_gram    = _build_anchor_stats_gram(ref_gram)
    st_loggram = _build_anchor_stats_loggram(ref_loggram) if ref_loggram is not None else None

    # ── Éval streaming ────────────────────────────────────────────────
    logger.info(f"  Extraction eval ({len(dist_paths)} images)…")
    cfg = get_backbone_config(backbone_name)
    model = _load_model(backbone_name, device)
    transform = _make_transform(backbone_name)
    layer_name = cfg["layer_names"][layer_idx]
    mod    = _get_module(model, layer_name)
    hook   = FeatureHook()
    handle = mod.register_forward_hook(hook)

    dataset = ImageDataset(dist_paths, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=(device == "cuda"),
                         persistent_workers=True)

    n = len(dist_paths)
    scores_gram    = np.empty(n, dtype=np.float64)
    scores_loggram = np.empty(n, dtype=np.float64) if st_loggram else None
    idx = 0
    t0  = time.time()

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            hook.clear()
            if backbone_name in ("sd_vae", "flux_vae", "dc_ae"):
                _ = model.encoder(imgs)
            else:
                _ = model(imgs)
            b = imgs.shape[0]

            gv  = _gram_vec(hook.features, MAX_CHANNELS)
            lgv = _log_gram_vec(hook.features, MAX_CHANNELS)

            if gv is not None:
                scores_gram[idx:idx+b] = _score_mmd_n1(gv, st_gram)
            if lgv is not None and st_loggram is not None:
                scores_loggram[idx:idx+b] = _score_mmd_n1(lgv, st_loggram)

            idx += b
            if idx % 2000 < batch_size or idx >= n:
                logger.info(f"    {idx}/{n} ({idx/n:.0%})  {time.time()-t0:.1f}s")

            gc.collect()

    handle.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    np.save(cache_gram, scores_gram)
    if scores_loggram is not None:
        np.save(cache_loggram, scores_loggram)

    return {
        "gram_rbf":    scores_gram,
        "loggram_med": scores_loggram,
    }


# ============================================================
# Baseline CMMD
# ============================================================

def evaluate_cmmd(anchor_paths, dist_paths, device, use_cache):
    cache_path = CACHE_DIR / "CMMD.npy"
    if use_cache and cache_path.exists():
        logger.info("  Cache CMMD trouvé")
        return np.load(cache_path)

    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
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
    mean_K = float(K_rr.mean())

    scores = np.empty(len(dist_paths), dtype=np.float64)
    chunk  = BATCH_SIZE_CLIP * 4
    t0     = time.time()
    for s in range(0, len(dist_paths), chunk):
        e   = min(s + chunk, len(dist_paths))
        emb = encode(dist_paths[s:e])
        t_sq = np.einsum("ij,ij->i", emb, emb)
        K_rt = np.exp(-CMMD_GAMMA * (ref_sq[:, None] + t_sq[None, :] - 2.0*(ref @ emb.T)))
        scores[s:e] = mean_K + 1.0 - 2.0 * np.mean(K_rt, axis=0)
        if s % 2000 < chunk or e == len(dist_paths):
            logger.info(f"    CLIP {e}/{len(dist_paths)}  {time.time()-t0:.1f}s")

    del model, proc
    torch.cuda.empty_cache()
    gc.collect()
    np.save(cache_path, scores)
    return scores


# ============================================================
# Statistiques
# ============================================================

def compute_stats(rows, scores):
    levels     = np.array([r["level"]     for r in rows])
    dmos       = np.array([r["dmos"]      for r in rows])
    degr       = 5.0 - dmos

    sp,  _  = spearmanr(scores, degr)
    kt,  _  = kendalltau(scores, degr)
    sp_lv, _ = spearmanr(scores, levels)

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

    by_level = {lv: scores[levels == lv] for lv in LEVELS}

    return {
        "spearman":  float(sp),
        "kendall":   float(kt),
        "sp_level":  float(sp_lv),
        "mono_rate": n_strict / n_total if n_total else 0.0,
        "n_total":   n_total,
        "by_level":  by_level,
        "scores":    scores,
        "dmos":      dmos,
        "levels":    levels,
    }


# ============================================================
# Figures
# ============================================================

STYLE = {
    "gram_rbf":    {"color": "#1f77b4", "ls": "-",  "label": "Gram + RBF (γ=2.8×10⁻⁵)"},
    "loggram_med": {"color": "#d62728", "ls": "--", "label": "LogGram + RBF (γ médiane)"},
    "cmmd":        {"color": "#9467bd", "ls": ":",  "label": "CMMD (CLIP)"},
}


def plot_violin(all_results: dict, out_path: Path):
    """
    Un subplot par backbone, deux violins superposés (gram vs loggram) par niveau.
    """
    backbones = [cfg[2] for cfg in BACKBONE_CONFIGS]
    n_bb = len(backbones)
    fig, axes = plt.subplots(1, n_bb, figsize=(6 * n_bb, 6), sharey=False)
    if n_bb == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, bb_name in zip(axes, backbones):
        if bb_name not in all_results:
            continue
        bb = all_results[bb_name]

        # Pour chaque distance disponible
        offsets = {"gram_rbf": -0.15, "loggram_med": 0.15}
        for dist_key, offset in offsets.items():
            if dist_key not in bb or bb[dist_key] is None:
                continue
            st     = bb[dist_key]
            data   = [st["by_level"][lv] for lv in LEVELS]
            style  = STYLE[dist_key]
            pos    = [lv + offset for lv in LEVELS]
            vp = ax.violinplot(data, positions=pos, widths=0.25,
                               showmedians=True, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(style["color"])
                body.set_alpha(0.45)
            vp["cmedians"].set_color(style["color"])
            vp["cmedians"].set_linewidth(1.5)

            # Ligne des moyennes
            means = [float(np.mean(data[i])) for i in range(5)]
            ax.plot([lv + offset for lv in LEVELS], means,
                    color=style["color"], linestyle=style["ls"],
                    marker="o", linewidth=1.5, markersize=4, zorder=5,
                    label=f"{style['label']}  ρ={st['spearman']:+.3f} mono={st['mono_rate']:.1%}")

        ax.set_title(bb_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Niveau de dégradation (1 → 5)", fontsize=10)
        ax.set_ylabel("Score MMD²", fontsize=10)
        ax.set_xticks(LEVELS)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)

    # CMMD baseline en ligne horizontale sur chaque subplot
    if "cmmd" in all_results:
        for ax in axes:
            # On annote juste le Spearman CMMD
            pass  # déjà dans le titre global

    fig.suptitle("Gram vectorisé vs Log-Euclidean Gram — KADID-10k (N=1 biaisé)\n"
                 "Score MMD² par niveau de dégradation (1=léger → 5=sévère)",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Violin : {out_path}")


def plot_scatter(all_results: dict, out_path: Path):
    """
    Scatter (dmos, score) — un subplot par (backbone × distance).
    """
    entries = []
    for bb_name in [cfg[2] for cfg in BACKBONE_CONFIGS]:
        if bb_name not in all_results:
            continue
        for dist_key in ["gram_rbf", "loggram_med"]:
            if dist_key in all_results[bb_name] and all_results[bb_name][dist_key] is not None:
                entries.append((bb_name, dist_key, all_results[bb_name][dist_key]))
    if "cmmd" in all_results and all_results["cmmd"] is not None:
        entries.append(("CMMD", "cmmd", all_results["cmmd"]))

    n = len(entries)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, (bb_name, dist_key, st) in zip(axes, entries):
        dmos   = st["dmos"]
        scores = st["scores"]
        lvs    = st["levels"]
        style  = STYLE.get(dist_key, {"color": "gray"})

        n_show = min(len(dmos), 2000)
        idx    = rng.choice(len(dmos), n_show, replace=False)
        for lv in LEVELS:
            m = lvs[idx] == lv
            ax.scatter(dmos[idx][m], scores[idx][m],
                       s=5, alpha=0.4, color=LEVEL_COLORS[lv],
                       linewidths=0, label=f"niv.{lv}")

        m2, b2 = np.polyfit(dmos, scores, 1)
        x_fit  = np.linspace(dmos.min(), dmos.max(), 200)
        ax.plot(x_fit, m2 * x_fit + b2, "k--", linewidth=1.2, zorder=10)

        rho, _ = spearmanr(dmos, scores)
        ax.text(0.04, 0.97, f"ρ = {rho:+.3f}",
                transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="gray"))

        label = f"{bb_name}\n{style.get('label', dist_key)}"
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("dmos (↑ = meilleure qualité)", fontsize=9)
        ax.set_ylabel("Score MMD²", fontsize=9)
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Score MMD² vs dmos — KADID-10k (2000 pts/metric, colorés par niveau)",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Scatter : {out_path}")


# ============================================================
# CSV + console
# ============================================================

def save_csv(all_results, out_path):
    rows_csv = []
    for bb_name in [cfg[2] for cfg in BACKBONE_CONFIGS]:
        if bb_name not in all_results:
            continue
        for dist_key in ["gram_rbf", "loggram_med"]:
            st = all_results[bb_name].get(dist_key)
            if st is None:
                continue
            rows_csv.append({
                "backbone":   bb_name,
                "distance":   dist_key,
                "spearman":   st["spearman"],
                "kendall":    st["kendall"],
                "sp_level":   st["sp_level"],
                "mono_rate":  st["mono_rate"],
            })
    if "cmmd" in all_results and all_results["cmmd"]:
        st = all_results["cmmd"]
        rows_csv.append({
            "backbone":  "CMMD", "distance": "cmmd",
            "spearman":  st["spearman"],  "kendall": st["kendall"],
            "sp_level":  st["sp_level"],  "mono_rate": st["mono_rate"],
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    logger.info(f"  CSV : {out_path}")


def print_summary(all_results):
    SEP = "=" * 100
    logger.info("\n" + SEP)
    logger.info("RÉSUMÉ — Gram-RBF vs MMDlogE (KADID-10k)")
    logger.info(SEP)
    logger.info(f"  {'Backbone':<18} {'Distance':<20}  {'Spearman':>9}  "
                f"{'Kendall':>8}  {'Mono.strict':>11}")
    logger.info("-" * 100)
    for bb_name in [cfg[2] for cfg in BACKBONE_CONFIGS]:
        if bb_name not in all_results:
            continue
        for dist_key in ["gram_rbf", "loggram_med"]:
            st = all_results[bb_name].get(dist_key)
            if st is None:
                continue
            logger.info(
                f"  {bb_name:<18} {dist_key:<20}  "
                f"{st['spearman']:>+9.4f}  {st['kendall']:>+8.4f}  "
                f"{st['mono_rate']:>10.1%}"
            )
    if "cmmd" in all_results and all_results["cmmd"]:
        st = all_results["cmmd"]
        logger.info(
            f"  {'CMMD':<18} {'cmmd':<20}  "
            f"{st['spearman']:>+9.4f}  {st['kendall']:>+8.4f}  "
            f"{st['mono_rate']:>10.1%}"
        )
    logger.info(SEP)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Gram RBF vs MMDlogE — KADID-10k")
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
    logger.info("Comparaison Gram-RBF vs MMDlogE — KADID-10k")
    logger.info(f"  γ_RBF   = {GAMMA_RBF:.2e}  (Gram vectorisé)")
    logger.info(f"  γ_LogE  = médiane heuristique (Log-Euclidean)")
    logger.info(f"  Device  = {device}  |  Cache = {use_cache}")
    logger.info("=" * 70)

    rows       = load_kadid_csv()
    ref_names  = sorted({r["ref_img"] for r in rows})
    anchor_paths = [str(IMAGES_DIR / n)             for n in ref_names]
    dist_paths   = [str(IMAGES_DIR / r["dist_img"]) for r in rows]

    all_results = {}   # {bb_name: {"gram_rbf": stats, "loggram_med": stats}}

    for backbone_name, layer_idx, display_name, batch_size in BACKBONE_CONFIGS:
        logger.info(f"\n── {display_name} ──")
        try:
            scores_dict = evaluate_backbone(
                backbone_name, layer_idx, display_name, batch_size,
                anchor_paths, dist_paths, device, use_cache,
            )
            bb_stats = {}
            for dist_key, sc in scores_dict.items():
                if sc is not None:
                    bb_stats[dist_key] = compute_stats(rows, sc)
                    st = bb_stats[dist_key]
                    logger.info(
                        f"  [{dist_key:>12}]  ρ={st['spearman']:+.4f}  "
                        f"mono={st['mono_rate']:.1%}"
                    )
            all_results[display_name] = bb_stats
        except Exception as exc:
            logger.error(f"  ERREUR {display_name} : {exc}", exc_info=True)

    if ENABLE_CMMD and not args.skip_cmmd:
        logger.info("\n── CMMD (CLIP ViT-L/14@336) ──")
        try:
            sc = evaluate_cmmd(anchor_paths, dist_paths, device, use_cache)
            st = compute_stats(rows, sc)
            all_results["cmmd"] = st
            logger.info(f"  ρ={st['spearman']:+.4f}  mono={st['mono_rate']:.1%}")
        except Exception as exc:
            logger.error(f"  ERREUR CMMD : {exc}", exc_info=True)

    print_summary(all_results)
    save_csv(all_results, OUT_DIR / f"summary_{ts}.csv")
    plot_violin(all_results, OUT_DIR / f"violin_par_niveau_{ts}.png")
    plot_scatter(all_results, OUT_DIR / f"scatter_dmos_{ts}.png")

    logger.info(f"\nTous les résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
