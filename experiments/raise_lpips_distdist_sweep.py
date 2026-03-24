"""
raise_lpips_distdist_sweep.py
==============================
Sweep γ → ρ(γ) pour LPIPS-VGG (13 layers) sur RAISE.

Mode : distribution-vs-distribution (MMD² non-biaisé)
  96 groupes × 5 images triés par MOS moyen
  vs 1000 images COCO anchor

  MMD²_u(P, Q) = (1/n_g(n_g-1)) Σ_{i≠j} k(qi,qj)
               + (1/n_r(n_r-1)) Σ_{i≠j} k(ri,rj)
               - (2/(n_g·n_r)) Σ_{i,j} k(ri,qj)

Réutilise le cache D_rr / D_rq de raise_lpips_gamma_sweep.py.
Calcule D_qq intra-groupe (96 × 10 paires) et le met en cache.

Référence CMMD : CLIP ViT-L/14@336, γ=0.005, 96 groupes × 5, dist-vs-dist.

Sorties dans results/raise_lpips_distdist/ :
  cache/   — D_qq_triu par layer, cmmd_distdist_rho.json
  plots/
    01_rho_vs_gamma_distdist.png
    02_heatmap_distdist.png
  lpips_raise_distdist_sweep.csv

Usage :
    .venv/bin/python experiments/raise_lpips_distdist_sweep.py
"""

import csv
import glob as glob_module
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import rankdata, spearmanr, t as t_dist
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Chemins ──────────────────────────────────────────────────────────────────
RAISE_ROOT          = BASE_DIR / "dataset" / "RAISE" / "dataset"
RAISE_TRAIN_IMAGES  = RAISE_ROOT / "images" / "train_images"
RAISE_TEST_IMAGES   = RAISE_ROOT / "images" / "test_images"
RAISE_TRAIN_RATINGS = RAISE_ROOT / "ratings" / "train.csv"
RAISE_TEST_RATINGS  = RAISE_ROOT / "ratings" / "test.csv"
COCO_DIR            = BASE_DIR / "dataset" / "coco_5000"

# Réutilise D_rr / D_rq du sweep point-vs-dist
SWEEP_CACHE = BASE_DIR / "results" / "raise_lpips_sweep" / "cache"

OUT_BASE   = BASE_DIR / "results" / "raise_lpips_distdist"
CACHE_DIR  = OUT_BASE / "cache"
PLOTS_DIR  = OUT_BASE / "plots"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
N_ANCHOR    = 1000
ANCHOR_SEED = 42
N_GROUPS    = 96
GROUP_SIZE  = 5
BATCH_SIZE  = 32
SWEEP_BATCH = 25   # gammas par batch GPU (mémoire réduite pour K_rq 4D)

CMMD_GAMMA    = 0.005          # σ=10 → γ = 1/(2σ²)
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
BATCH_CLIP    = 32

LPIPS_LAYER_NAMES = {
    0:  "net.slice1.0",  1: "net.slice1.2",
    2:  "net.slice2.5",  3: "net.slice2.7",
    4:  "net.slice3.10", 5: "net.slice3.12", 6: "net.slice3.14",
    7:  "net.slice4.17", 8: "net.slice4.19", 9: "net.slice4.21",
    10: "net.slice5.24", 11: "net.slice5.26", 12: "net.slice5.28",
}
SHORT_NAMES = {
    0: "relu1_1", 1: "relu1_2",
    2: "relu2_1", 3: "relu2_2",
    4: "relu3_1", 5: "relu3_2", 6: "relu3_3",
    7: "relu4_1", 8: "relu4_2", 9: "relu4_3",
    10: "relu5_1", 11: "relu5_2", 12: "relu5_3",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MultiLayer LPIPS Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class _Hook:
    def __init__(self):
        self.feat = None
    def __call__(self, m, i, o):
        self.feat = (o[0] if isinstance(o, tuple) else o).detach()
    def clear(self):
        self.feat = None


class LPIPSMultiExtractor:
    def __init__(self, device: str):
        import lpips
        self.device = device
        logger.info("  Chargement LPIPS-VGG…")
        self.model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.hooks = {}
        self._handles = []
        for l_idx, name in LPIPS_LAYER_NAMES.items():
            m = self._get_module(name)
            h = _Hook()
            self._handles.append(m.register_forward_hook(h))
            self.hooks[l_idx] = h

    def _get_module(self, name: str) -> nn.Module:
        parts = name.split(".")
        m = self.model
        for p in parts:
            if p.isdigit():
                try:
                    m = getattr(m, p)
                except AttributeError:
                    m = m[int(p)]
            else:
                m = getattr(m, p)
        return m

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        feat = feat.float()
        if feat.dim() == 4:
            B, C, H, W = feat.shape
            F = feat.reshape(B, C, H * W); S = H * W
        elif feat.dim() == 3:
            B, N, D = feat.shape
            F = feat.permute(0, 2, 1); C = D; S = N
        elif feat.dim() == 2:
            B, C = feat.shape
            F = feat.unsqueeze(2); S = 1
        else:
            raise ValueError(f"dim {feat.dim()}")
        G = torch.bmm(F, F.transpose(1, 2)) / S
        mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=feat.device))
        return G[:, mask]

    def extract_all(self, paths: list, batch_size: int = BATCH_SIZE) -> dict:
        class _DS(Dataset):
            def __init__(self, paths, tf):
                self.paths = paths; self.tf = tf
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                return self.tf(Image.open(self.paths[i]).convert("RGB"))

        loader = DataLoader(_DS(paths, self.transform),
                            batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        per_layer = {l: [] for l in LPIPS_LAYER_NAMES}
        n_done = 0
        for batch in loader:
            batch = batch.to(self.device)
            for h in self.hooks.values():
                h.clear()
            with torch.no_grad():
                scaled = self.model.scaling_layer(batch)
                self.model.net(scaled)
            for l, hook in self.hooks.items():
                if hook.feat is not None:
                    per_layer[l].append(self._gram(hook.feat).cpu().numpy().astype(np.float32))
            n_done += batch.shape[0]
            print(f"    {n_done}/{len(paths)}\r", end="", flush=True)
        print()
        return {l: np.concatenate(v, axis=0) for l, v in per_layer.items() if v}

    def __del__(self):
        for h in self._handles:
            h.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# Données
# ═══════════════════════════════════════════════════════════════════════════════

def load_coco_anchor() -> list:
    all_imgs = sorted(
        glob_module.glob(str(COCO_DIR / "*.jpg")) +
        glob_module.glob(str(COCO_DIR / "*.png"))
    )
    rng = np.random.default_rng(ANCHOR_SEED)
    idx = sorted(rng.choice(len(all_imgs), size=N_ANCHOR, replace=False).tolist())
    return [all_imgs[i] for i in idx]


def load_raise_items() -> tuple:
    mos_dict = {}
    for cp in [RAISE_TRAIN_RATINGS, RAISE_TEST_RATINGS]:
        with open(cp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mos_dict[row["filename"]] = float(row["MOS_Rating"])
    all_imgs = {}
    for folder in [RAISE_TRAIN_IMAGES, RAISE_TEST_IMAGES]:
        for ext in ["png", "PNG", "jpg", "jpeg"]:
            for p in glob_module.glob(str(folder / f"*.{ext}")):
                all_imgs[os.path.basename(p)] = p
    items = [(path, mos_dict[fname])
             for fname, path in all_imgs.items()
             if fname.startswith("f") and fname in mos_dict]
    items.sort(key=lambda x: x[1])
    logger.info(f"RAISE IA : {len(items)} images  MOS [{items[0][1]:.2f}…{items[-1][1]:.2f}]")
    return [x[0] for x in items], np.array([x[1] for x in items])


def make_groups(all_mos: np.ndarray) -> np.ndarray:
    """Returns mean MOS per group (N_GROUPS,), groups formed consecutively."""
    assert len(all_mos) >= N_GROUPS * GROUP_SIZE
    return np.array([all_mos[g * GROUP_SIZE:(g + 1) * GROUP_SIZE].mean()
                     for g in range(N_GROUPS)])


# ═══════════════════════════════════════════════════════════════════════════════
# Distances
# ═══════════════════════════════════════════════════════════════════════════════

def sq_dist_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32); B = B.astype(np.float32)
    return np.maximum(
        (A * A).sum(1, keepdims=True) + (B * B).sum(1, keepdims=True).T
        - 2.0 * (A @ B.T), 0.0
    )


def load_drr_drq() -> dict:
    """Load D_rr and D_rq from the point-vs-dist cache."""
    missing = [l for l in LPIPS_LAYER_NAMES
               if not (SWEEP_CACHE / f"lpips_l{l:02d}_Drr.npy").exists()
               or not (SWEEP_CACHE / f"lpips_l{l:02d}_Drq.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Cache point-vs-dist manquant pour layers {missing}. "
            f"Lancer d'abord raise_lpips_gamma_sweep.py."
        )
    distances = {}
    for l in LPIPS_LAYER_NAMES:
        D_rr = np.load(SWEEP_CACHE / f"lpips_l{l:02d}_Drr.npy").astype(np.float64)
        D_rq = np.load(SWEEP_CACHE / f"lpips_l{l:02d}_Drq.npy").astype(np.float64)
        distances[l] = (D_rr, D_rq)
    logger.info("  Cache D_rr/D_rq chargé (13 layers)")
    return distances


def build_dqq(extractor: LPIPSMultiExtractor,
              anchor_paths: list, query_paths: list,
              device: str) -> dict:
    """Compute D_qq_triu (96, 10) per layer — upper-triangle pairwise distances within each group."""
    N_PAIRS = GROUP_SIZE * (GROUP_SIZE - 1) // 2  # = 10

    missing = [l for l in LPIPS_LAYER_NAMES
               if not (CACHE_DIR / f"lpips_l{l:02d}_Dqq.npy").exists()]
    dqq = {}

    if not missing:
        logger.info("  Cache D_qq complet — chargement…")
        for l in LPIPS_LAYER_NAMES:
            dqq[l] = np.load(CACHE_DIR / f"lpips_l{l:02d}_Dqq.npy").astype(np.float64)
        return dqq

    logger.info(f"  Extraction anchor ({N_ANCHOR} imgs) pour scaler…")
    anc_raw = extractor.extract_all(anchor_paths)
    logger.info(f"  Extraction query ({len(query_paths)} imgs)…")
    qry_raw = extractor.extract_all(query_paths)

    triu_i, triu_j = np.triu_indices(GROUP_SIZE, k=1)

    for l in LPIPS_LAYER_NAMES:
        p = CACHE_DIR / f"lpips_l{l:02d}_Dqq.npy"
        if p.exists():
            dqq[l] = np.load(p).astype(np.float64)
            logger.info(f"    L{l:02d} — cache")
            continue

        scaler = StandardScaler()
        scaler.fit_transform(anc_raw[l].astype(np.float64))
        q_std = scaler.transform(qry_raw[l].astype(np.float64)).astype(np.float32)

        D_qq_triu = np.zeros((N_GROUPS, N_PAIRS), dtype=np.float32)
        for g in range(N_GROUPS):
            feat_g = q_std[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
            sq_A = (feat_g * feat_g).sum(axis=1, keepdims=True)
            D = np.maximum(sq_A + sq_A.T - 2.0 * (feat_g @ feat_g.T), 0.0)
            D_qq_triu[g] = D[triu_i, triu_j]

        np.save(p, D_qq_triu)
        dqq[l] = D_qq_triu.astype(np.float64)
        logger.info(f"    L{l:02d}  D_qq{D_qq_triu.shape}")

    return dqq


# ═══════════════════════════════════════════════════════════════════════════════
# Spearman fast
# ═══════════════════════════════════════════════════════════════════════════════

def spearman_pval(rho: float, n: int) -> float:
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 else np.nan
    t = rho * np.sqrt(n - 2) / np.sqrt(1 - rho ** 2)
    return float(2 * t_dist.sf(abs(t), df=n - 2))


def fast_spearman_batch(scores_batch: np.ndarray,
                        mos_rank_c: np.ndarray,
                        mos_rank_std: float) -> tuple:
    B, N = scores_batch.shape
    rhos = np.empty(B); pvals = np.empty(B)
    for i in range(B):
        r = rankdata(scores_batch[i]).astype(np.float64)
        r -= r.mean()
        rs = np.sqrt((r ** 2).sum())
        if rs < 1e-12:
            rhos[i] = pvals[i] = np.nan
        else:
            rho = float((r * mos_rank_c).sum() / (rs * mos_rank_std))
            rhos[i] = rho
            pvals[i] = spearman_pval(rho, N)
    return rhos, pvals


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep GPU — distribution-vs-distribution
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_distdist(distances: dict, dqq: dict,
                   gammas: np.ndarray, mean_mos_groups: np.ndarray,
                   device: str, batch_size: int = SWEEP_BATCH) -> dict:
    G = len(gammas)
    mr_c = rankdata(mean_mos_groups).astype(np.float64)
    mr_c -= mr_c.mean()
    mr_std = np.sqrt((mr_c ** 2).sum())

    gammas_t = torch.from_numpy(gammas.astype(np.float32)).to(device)
    results = {}

    for l in sorted(distances.keys()):
        D_rr_np, D_rq_np = distances[l]

        # Upper triangle of D_rr for unbiased K_rr mean
        triu_r = np.triu_indices(N_ANCHOR, k=1)
        D_rr_triu_t = torch.from_numpy(
            D_rr_np[triu_r].astype(np.float32)
        ).to(device)  # (499500,)

        # D_rq grouped: (1000, 96, 5)
        D_rq_g_t = torch.from_numpy(
            D_rq_np.astype(np.float32).reshape(N_ANCHOR, N_GROUPS, GROUP_SIZE)
        ).to(device)

        # D_qq upper triangle per group: (96, 10)
        D_qq_t = torch.from_numpy(dqq[l].astype(np.float32)).to(device)

        all_rhos  = np.empty(G)
        all_pvals = np.empty(G)

        for start in range(0, G, batch_size):
            end = min(start + batch_size, G)
            g = gammas_t[start:end]  # (B,)

            # Unbiased K_rr: mean of upper-triangle pairs
            K_rr_triu = torch.exp(-g[:, None] * D_rr_triu_t[None])  # (B, 499500)
            mean_K_rr = K_rr_triu.mean(dim=1)  # (B,)
            del K_rr_triu

            # Unbiased K_qq: mean of 10 upper-triangle pairs per group
            K_qq = torch.exp(-g[:, None, None] * D_qq_t[None])  # (B, 96, 10)
            mean_K_qq = K_qq.mean(dim=2)  # (B, 96)
            del K_qq

            # K_rq cross-term: mean over anchors × group members
            K_rq = torch.exp(
                -g[:, None, None, None] * D_rq_g_t[None]
            )  # (B, 1000, 96, 5)
            mean_K_rq = K_rq.mean(dim=(1, 3))  # (B, 96)
            del K_rq

            mmd2 = mean_K_qq + mean_K_rr[:, None] - 2.0 * mean_K_rq  # (B, 96)
            scores_np = mmd2.cpu().numpy().astype(np.float64)
            del mmd2, mean_K_rr, mean_K_qq, mean_K_rq
            torch.cuda.empty_cache()

            rhos_b, pvals_b = fast_spearman_batch(scores_np, mr_c, mr_std)
            all_rhos[start:end]  = rhos_b
            all_pvals[start:end] = pvals_b

        del D_rr_triu_t, D_rq_g_t, D_qq_t
        torch.cuda.empty_cache()

        results[l] = {"rho": all_rhos, "pval": all_pvals}
        best = float(np.nanmax(all_rhos))
        logger.info(f"  L{l:02d} {SHORT_NAMES[l]:<10}  ρ max = {best:+.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Référence CMMD (CLIP ViT-L/14@336, γ=0.005, dist-vs-dist)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cmmd_distdist(anchor_paths: list, query_paths: list,
                          mean_mos_groups: np.ndarray, device: str) -> float:
    cache_file = CACHE_DIR / "cmmd_distdist_rho.json"
    if cache_file.exists():
        with open(cache_file) as f:
            val = json.load(f)["rho"]
        logger.info(f"  CMMD dist-vs-dist ρ (cache) = {val:+.4f}")
        return float(val)

    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    logger.info(f"\n  CMMD — {CLIP_MODEL_ID}")
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    proc  = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def embed(paths: list) -> np.ndarray:
        embs = []
        for i in range(0, len(paths), BATCH_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                embs.append(model(pixel_values=pv).image_embeds.float().cpu().numpy())
            print(f"    {min(i + BATCH_CLIP, len(paths))}/{len(paths)}\r", end="", flush=True)
        print()
        return np.concatenate(embs, axis=0)

    logger.info(f"  Embeddings anchor ({N_ANCHOR} imgs)…")
    anc_emb = embed(anchor_paths)
    logger.info(f"  Embeddings query ({len(query_paths)} imgs)…")
    qry_emb = embed(query_paths)

    # term1_rr (unbiased)
    D_rr = sq_dist_np(anc_emb, anc_emb)
    n_r = anc_emb.shape[0]
    K_rr = np.exp(-CMMD_GAMMA * D_rr)
    term1 = float((np.sum(K_rr) - np.trace(K_rr)) / (n_r * (n_r - 1)))
    logger.info(f"  term1_rr = {term1:.6f}")

    scores = []
    for g in range(N_GROUPS):
        feat_g = qry_emb[g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
        D_rg = sq_dist_np(anc_emb, feat_g)
        D_gg = sq_dist_np(feat_g, feat_g)
        K_rg = np.exp(-CMMD_GAMMA * D_rg)
        K_gg = np.exp(-CMMD_GAMMA * D_gg)
        n_g = feat_g.shape[0]
        term2 = float((np.sum(K_gg) - np.trace(K_gg)) / (n_g * (n_g - 1)))
        term3 = 2.0 * float(np.sum(K_rg)) / (n_r * n_g)
        scores.append(term1 + term2 - term3)

    rho, p = spearmanr(scores, mean_mos_groups)
    logger.info(f"  CMMD dist-vs-dist  ρ = {rho:+.4f}  (p={p:.3e})  γ={CMMD_GAMMA:.4e}")

    with open(cache_file, "w") as f:
        json.dump({"rho": float(rho), "p": float(p), "gamma": CMMD_GAMMA}, f)

    del model, proc, anc_emb, qry_emb
    torch.cuda.empty_cache()
    return float(rho)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

def _median_gammas(layers):
    mg = {}
    for l in layers:
        # Reuse D_rr from point-vs-dist cache
        p = SWEEP_CACHE / f"lpips_l{l:02d}_Drr.npy"
        if p.exists():
            D = np.load(p).astype(np.float64)
            nz = D[D > 0]
            if len(nz):
                g_med = float(1.0 / (2.0 * np.median(nz)))
                mg[l] = {"med": g_med, "50x": 50.0 * g_med}
    return mg


def plot_rho_vs_gamma(results: dict, gammas: np.ndarray,
                      cmmd_rho: float, out_dir: Path):
    layers = sorted(results.keys())
    n_layers = len(layers)
    cmap = plt.get_cmap("plasma", n_layers)
    med_g = _median_gammas(layers)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f9f9f9")
    best_per_layer = {}

    for idx, l in enumerate(layers):
        rhos = results[l]["rho"]
        valid = np.isfinite(rhos)
        if not valid.any():
            continue
        ax.plot(gammas[valid], rhos[valid], color=cmap(idx), linewidth=1.4,
                alpha=0.85, label=f"L{l:02d} {SHORT_NAMES.get(l, '')}")
        bi = int(np.nanargmax(rhos))
        best_per_layer[l] = (gammas[bi], float(rhos[bi]))
        ax.scatter([gammas[bi]], [rhos[bi]], color=cmap(idx), s=40, zorder=5,
                   edgecolors="black", linewidths=0.4)
        if l in med_g:
            gi = int(np.argmin(np.abs(gammas - med_g[l]["med"])))
            rv = float(rhos[gi]) if np.isfinite(rhos[gi]) else np.nan
            if np.isfinite(rv):
                ax.scatter([gammas[gi]], [rv], color=cmap(idx), s=70, zorder=6,
                           marker="^", edgecolors="black", linewidths=0.6)
            gi50 = int(np.argmin(np.abs(gammas - med_g[l]["50x"])))
            rv50 = float(rhos[gi50]) if np.isfinite(rhos[gi50]) else np.nan
            if np.isfinite(rv50):
                ax.scatter([gammas[gi50]], [rv50], color=cmap(idx), s=70, zorder=6,
                           marker="s", edgecolors="black", linewidths=0.6)

    # Référence CMMD
    ax.axhline(cmmd_rho, color="#E91E63", linewidth=1.8, linestyle="-.",
               zorder=4, label=f"CMMD γ={CMMD_GAMMA:.3f}  (ρ={cmmd_rho:+.3f})")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("γ (log scale)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title(
        "LPIPS-VGG — Spearman ρ as a function of γ  (1 000 values, log grid)\n"
        f"RAISE IA  ·  96 groups × 5  ·  dist-vs-dist  ·  anchor = {N_ANCHOR} COCO  ·  std ON",
        fontsize=11, fontweight="bold",
    )
    extra = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6,
               markeredgecolor="black", markeredgewidth=0.4, label="Optimal γ (best ρ)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=7,
               markeredgecolor="black", markeredgewidth=0.6,
               label="γ = 1/(2·median(D_rr))"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None", markersize=7,
               markeredgecolor="black", markeredgewidth=0.6,
               label="γ = 50 × 1/(2·median(D_rr))"),
        Line2D([0], [0], color="#E91E63", linestyle="-.", linewidth=1.8,
               label=f"CMMD ViT-L/14@336  γ={CMMD_GAMMA}  ρ={cmmd_rho:+.3f}"),
    ]
    ll = ax.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.85)
    ax.add_artist(ll)
    ax.legend(handles=extra, fontsize=8, loc="upper left",
              framealpha=0.9, bbox_to_anchor=(0.0, 1.0))
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    out = out_dir / "01_rho_vs_gamma_distdist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")
    return best_per_layer


def plot_heatmap(results: dict, gammas: np.ndarray, out_dir: Path):
    layers = sorted(results.keys())
    G = len(gammas)
    step = max(1, G // 100)
    g_idx = np.arange(0, G, step)
    matrix = np.full((len(layers), len(g_idx)), np.nan)
    for i, l in enumerate(layers):
        matrix[i] = results[l]["rho"][g_idx]
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.4, vmax=0.4, origin="upper")
    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.02, pad=0.02)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l:02d} {SHORT_NAMES.get(l, '')}" for l in layers], fontsize=9)
    xs = max(1, len(g_idx) // 10)
    ax.set_xticks(range(0, len(g_idx), xs))
    ax.set_xticklabels([f"{gammas[g_idx[j]]:.1e}" for j in range(0, len(g_idx), xs)],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("γ", fontsize=11)
    ax.set_title(
        "LPIPS-VGG — Heatmap ρ(layer × γ)  ·  1 000 gammas\n"
        f"RAISE IA  ·  96 groups × 5  ·  dist-vs-dist",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = out_dir / "02_heatmap_distdist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


def export_csv(results: dict, gammas: np.ndarray, out_dir: Path):
    rows = []
    for l in sorted(results.keys()):
        for gi, g in enumerate(gammas):
            rho = results[l]["rho"][gi]
            pv  = results[l]["pval"][gi]
            rows.append({
                "layer": l, "layer_name": SHORT_NAMES.get(l, ""),
                "gamma": float(g),
                "spearman_rho": float(rho) if np.isfinite(rho) else "",
                "pval":         float(pv)  if np.isfinite(pv)  else "",
            })
    out = out_dir / "lpips_raise_distdist_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device : {device}")
    logger.info(f"Mode   : dist-vs-dist  ·  {N_GROUPS} groups × {GROUP_SIZE}")

    gammas = np.logspace(-9, -1, 1000)

    anchor_paths = load_coco_anchor()
    query_paths, all_mos = load_raise_items()

    mean_mos_groups = make_groups(all_mos)
    logger.info(f"Groupes MOS : {mean_mos_groups.min():.2f} … {mean_mos_groups.max():.2f}")

    # ── D_rr / D_rq depuis cache point-vs-dist ─────────────────────────
    distances = load_drr_drq()

    # ── D_qq intra-groupes ─────────────────────────────────────────────
    logger.info("\nD_qq intra-groupes…")
    t0 = time.time()
    missing_dqq = [l for l in LPIPS_LAYER_NAMES
                   if not (CACHE_DIR / f"lpips_l{l:02d}_Dqq.npy").exists()]
    if missing_dqq:
        extractor = LPIPSMultiExtractor(device)
        dqq = build_dqq(extractor, anchor_paths, query_paths, device)
        del extractor
        torch.cuda.empty_cache()
    else:
        dqq = build_dqq(None, anchor_paths, query_paths, device)
    logger.info(f"  D_qq : {time.time()-t0:.1f}s")

    # ── Référence CMMD ─────────────────────────────────────────────────
    logger.info("\nCMmd référence (dist-vs-dist)…")
    cmmd_rho = compute_cmmd_distdist(anchor_paths, query_paths, mean_mos_groups, device)

    # ── Sweep GPU ───────────────────────────────────────────────────────
    logger.info(f"\nSweep 1 000 gammas × {len(distances)} layers (dist-vs-dist)…")
    t1 = time.time()
    results = sweep_distdist(distances, dqq, gammas, mean_mos_groups, device)
    logger.info(f"  Sweep : {time.time()-t1:.1f}s")

    # ── Figures + CSV ───────────────────────────────────────────────────
    logger.info("\nFigures…")
    best = plot_rho_vs_gamma(results, gammas, cmmd_rho, PLOTS_DIR)
    plot_heatmap(results, gammas, PLOTS_DIR)
    export_csv(results, gammas, OUT_BASE)

    # ── Résumé ──────────────────────────────────────────────────────────
    med_g = _median_gammas(list(results.keys()))
    print("\n" + "=" * 72)
    print(f"  {'Layer':<6} {'Nom':<12} {'γ optimal':>10}  {'ρ max':>8}  "
          f"{'ρ@median':>9}  {'ρ@50×med':>9}")
    print("  " + "-" * 60)
    for l, (g_opt, rho_opt) in sorted(best.items(), key=lambda x: -x[1][1]):
        rhos = results[l]["rho"]

        def rho_at(key):
            if l not in med_g: return float("nan")
            gi = int(np.argmin(np.abs(gammas - med_g[l][key])))
            return float(rhos[gi]) if np.isfinite(rhos[gi]) else float("nan")

        r_med = rho_at("med")
        r_50x = rho_at("50x")
        print(f"  L{l:02d}   {SHORT_NAMES.get(l,''):<12} {g_opt:>10.2e}  "
              f"{rho_opt:>+8.4f}  {r_med:>+9.4f}  {r_50x:>+9.4f}")
    print(f"\n  CMMD ViT-L/14@336  γ={CMMD_GAMMA}  ρ = {cmmd_rho:+.4f}")
    print("=" * 72)
    logger.info(f"\nRésultats : {OUT_BASE}")


if __name__ == "__main__":
    main()
