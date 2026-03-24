"""
gamma_sweep_sdvae_indep_anchor.py
==================================
Même sweep γ → ρ(γ) que gamma_sweep_sdvae.py, mais avec un ancrage
**indépendant** : 200 images propres de coco_5000 qui ne font PAS partie
du dataset COCO-dégradé.

Optimisations GPU :
  • MultiLayerExtractor : modèle chargé UNE seule fois, 17 hooks simultanés,
    un seul passage DataLoader  →  ~17× plus rapide que l'approche layer-by-layer
  • Sweep kernels sur GPU : exp(-γ·D) calculé en fp32 sur CUDA, Spearman CPU
  • Distances (D_rr, D_rq) calculées sur GPU via torch.cdist

Usage :
    .venv/bin/python experiments/gamma_sweep_sdvae_indep_anchor.py
"""

import csv
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
from scipy.stats import rankdata, t as t_dist
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
COCO5K_DIR   = BASE_DIR / "dataset" / "coco_5000"
COCO_DEG_DIR = BASE_DIR / "dataset" / "coco_degraded"
OLD_CACHE    = BASE_DIR / "results" / "sweep_coco_degraded" / "cache"
OUT_BASE     = BASE_DIR / "results" / "sweep_indep_anchor"
NEW_CACHE    = OUT_BASE / "cache"
PLOTS_DIR    = OUT_BASE / "plots"

NEW_CACHE.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_ANCHORS  = 200
DIST_TYPES = [1, 10, 11, 21, 25]
BATCH_SIZE = 16        # GPU batch for feature extraction
SWEEP_BATCH = 50       # gamma batch size for kernel computation

SDVAE_LAYER_NAMES = {
    0: "encoder.conv_in",
    1: "encoder.down_blocks.0.resnets.0",
    2: "encoder.down_blocks.0.resnets.1",
    3: "encoder.down_blocks.0.downsamplers.0",
    4: "encoder.down_blocks.1.resnets.0",
    5: "encoder.down_blocks.1.resnets.1",
    6: "encoder.down_blocks.1.downsamplers.0",
    7: "encoder.down_blocks.2.resnets.0",
    8: "encoder.down_blocks.2.resnets.1",
    9: "encoder.down_blocks.2.downsamplers.0",
    10: "encoder.down_blocks.3.resnets.0",
    11: "encoder.down_blocks.3.resnets.1",
    12: "encoder.mid_block.resnets.0",
    13: "encoder.mid_block.attentions.0",
    14: "encoder.mid_block.resnets.1",
    15: "encoder.conv_norm_out",
    16: "encoder.conv_out",
}

SHORT_NAMES = {
    0:  "conv_in",       1:  "down.0.res.0",   2:  "down.0.res.1",
    3:  "down.0.ds",     4:  "down.1.res.0",   5:  "down.1.res.1",
    6:  "down.1.ds",     7:  "down.2.res.0",   8:  "down.2.res.1",
    9:  "down.2.ds",     10: "down.3.res.0",   11: "down.3.res.1",
    12: "mid.res.0",     13: "mid.attn",        14: "mid.res.1",
    15: "conv_norm_out", 16: "conv_out",
}


# ===========================================================================
# MultiLayerExtractor  (un seul chargement modèle, tous les layers en 1 passe)
# ===========================================================================

class _FeatureHook:
    def __init__(self):
        self.features = None
    def __call__(self, module, inp, out):
        self.features = (out[0] if isinstance(out, tuple) else out).detach()
    def clear(self):
        self.features = None


class MultiLayerExtractor:
    """
    Extrait les Gram features de tous les layers SD-VAE en un seul passage.
    Appeler extract_all() une fois pour les anchors, une fois pour les queries.
    """

    def __init__(self, device: str):
        from diffusers import AutoencoderKL
        self.device = device

        print("  Chargement SD-VAE…", flush=True)
        self.model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Enregistrement des hooks
        self.hooks = {}
        self._handles = []
        for l_idx, name in SDVAE_LAYER_NAMES.items():
            module = self._get_module(name)
            hook = _FeatureHook()
            self._handles.append(module.register_forward_hook(hook))
            self.hooks[l_idx] = hook

    def _get_module(self, name: str) -> nn.Module:
        parts = name.split(".")
        m = self.model
        for p in parts:
            m = m[int(p)] if p.isdigit() else getattr(m, p)
        return m

    @staticmethod
    def _gram_spatial(feat: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C*(C+1)//2) Gram upper-tri / S."""
        feat = feat.float()
        if feat.dim() == 4:
            B, C, H, W = feat.shape
            F = feat.reshape(B, C, H * W)
        elif feat.dim() == 3:
            B, N, D = feat.shape
            F = feat.permute(0, 2, 1)
            C = D
        else:
            raise ValueError(f"Unexpected dim {feat.dim()}")
        S = F.shape[2]
        G = torch.bmm(F, F.transpose(1, 2)) / S            # (B, C, C)
        mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=feat.device))
        return G[:, mask]                                   # (B, C*(C+1)//2)

    def extract_all(self, paths: list, batch_size: int = BATCH_SIZE) -> dict:
        """
        Retourne {layer_idx: np.ndarray(N, D_l)} pour toutes les 17 layers.
        """
        class _DS(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                return self.transform(img)

        loader = DataLoader(_DS(paths, self.transform),
                            batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        per_layer = {l: [] for l in SDVAE_LAYER_NAMES}
        n_total = 0

        for batch in loader:
            batch = batch.to(self.device)
            for h in self.hooks.values():
                h.clear()
            with torch.no_grad():
                self.model.encoder(batch)
            for l_idx, hook in self.hooks.items():
                if hook.features is not None:
                    g = self._gram_spatial(hook.features)   # (B, D_l)
                    per_layer[l_idx].append(g.cpu().numpy().astype(np.float32))
            n_total += batch.shape[0]
            print(f"    {n_total}/{len(paths)} images…", end="\r", flush=True)

        print()
        return {l: np.concatenate(v, axis=0) for l, v in per_layer.items() if v}

    def __del__(self):
        for h in self._handles:
            h.remove()


# ===========================================================================
# Distances sur GPU
# ===========================================================================

def squared_l2_gpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """(n,d)×(m,d) → (n,m)  squared L2, GPU."""
    a2 = (A * A).sum(dim=1, keepdim=True)
    b2 = (B * B).sum(dim=1, keepdim=True)
    return torch.clamp(a2 + b2.T - 2.0 * (A @ B.T), min=0.0)


# ===========================================================================
# Construction et mise en cache des distances par layer
# ===========================================================================

def build_all_distances(extractor: MultiLayerExtractor,
                         anchor_paths: list,
                         device: str) -> dict:
    """
    Extrait anchor features + charge query features, standardise,
    calcule D_rr et D_rq pour tous les layers.

    Retourne {l: (D_rr np.float64, D_rq np.float64)}.
    Charge depuis le cache si disponible.
    """
    # Vérifier si tout est déjà caché
    missing = [l for l in SDVAE_LAYER_NAMES
               if not (NEW_CACHE / f"sd_vae_indep_l{l:02d}_Drr.npy").exists()
               or not (NEW_CACHE / f"sd_vae_indep_l{l:02d}_Drq.npy").exists()]

    distances = {}

    if not missing:
        print("  Tout le cache est disponible — chargement…", flush=True)
        for l in SDVAE_LAYER_NAMES:
            D_rr = np.load(NEW_CACHE / f"sd_vae_indep_l{l:02d}_Drr.npy").astype(np.float64)
            D_rq = np.load(NEW_CACHE / f"sd_vae_indep_l{l:02d}_Drq.npy").astype(np.float64)
            distances[l] = (D_rr, D_rq)
        return distances

    # Extraction anchor (une passe modèle)
    print(f"  Extraction features anchor ({N_ANCHORS} images, 17 layers)…", flush=True)
    anchor_raw = extractor.extract_all(anchor_paths)

    # Chargement query raw depuis l'ancien cache
    print("  Chargement query features…", flush=True)

    for l in SDVAE_LAYER_NAMES:
        drr_path = NEW_CACHE / f"sd_vae_indep_l{l:02d}_Drr.npy"
        drq_path = NEW_CACHE / f"sd_vae_indep_l{l:02d}_Drq.npy"

        if drr_path.exists() and drq_path.exists():
            D_rr = np.load(drr_path).astype(np.float64)
            D_rq = np.load(drq_path).astype(np.float64)
            distances[l] = (D_rr, D_rq)
            print(f"    L{l:02d} — cache", flush=True)
            continue

        q_path = OLD_CACHE / f"sd_vae_l{l:02d}_query_r200_t1-10-11-21-25.npy"
        if not q_path.exists():
            print(f"    [skip] L{l:02d} — query cache absent")
            continue
        query_raw = np.load(q_path).astype(np.float32)

        # Standardisation (scaler fitté sur anchor)
        scaler = StandardScaler()
        a_std = scaler.fit_transform(anchor_raw[l].astype(np.float64))
        q_std = scaler.transform(query_raw.astype(np.float64))

        # Distances GPU
        A_t = torch.from_numpy(a_std.astype(np.float32)).to(device)
        Q_t = torch.from_numpy(q_std.astype(np.float32)).to(device)
        D_rr_t = squared_l2_gpu(A_t, A_t)
        D_rq_t = squared_l2_gpu(A_t, Q_t)

        D_rr = D_rr_t.cpu().numpy().astype(np.float32)
        D_rq = D_rq_t.cpu().numpy().astype(np.float32)
        np.save(drr_path, D_rr)
        np.save(drq_path, D_rq)
        print(f"    L{l:02d}  D_rr{D_rr.shape}  D_rq{D_rq.shape}", flush=True)

        distances[l] = (D_rr.astype(np.float64), D_rq.astype(np.float64))

    return distances


# ===========================================================================
# Métadonnées
# ===========================================================================

def load_levels() -> tuple:
    meta_path = COCO_DEG_DIR / "metadata.csv"
    dist_filter = set(DIST_TYPES)
    sorted_types = sorted(dist_filter)
    type_to_rank = {t: i for i, t in enumerate(sorted_types)}
    n_types = len(sorted_types)

    rows = []
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    all_refs, seen = [], set()
    for r in rows:
        if r["ref_img"] not in seen:
            all_refs.append(r["ref_img"])
            seen.add(r["ref_img"])

    selected_refs = set(all_refs[:200])
    ref_to_idx = {r: i for i, r in enumerate(all_refs[:200])}

    levels, seq_ids = [], []
    for r in rows:
        if r["ref_img"] not in selected_refs:
            continue
        dist_type = int(r["dist_type"])
        if dist_type not in dist_filter:
            continue
        ref_idx = ref_to_idx[r["ref_img"]]
        level = int(r["level"])
        seq_id = ref_idx * n_types + type_to_rank[dist_type]
        levels.append(level)
        seq_ids.append(seq_id)

    return np.array(levels, dtype=np.int32), np.array(seq_ids, dtype=np.int32)


# ===========================================================================
# Spearman vectorisé (CPU, identique à gamma_sweep_sdvae.py)
# ===========================================================================

def spearman_pval(rho: float, n: int) -> float:
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 else np.nan
    t_stat = rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2)
    return float(2 * t_dist.sf(abs(t_stat), df=n - 2))


def fast_spearman_batch(scores_batch, lr_c, lr_std):
    B, N = scores_batch.shape
    rhos = np.empty(B)
    pvals = np.empty(B)
    for i in range(B):
        r = rankdata(scores_batch[i]).astype(np.float64)
        r -= r.mean()
        rs = np.sqrt((r ** 2).sum())
        if rs < 1e-12:
            rhos[i] = pvals[i] = np.nan
        else:
            rho = float((r * lr_c).sum() / (rs * lr_std))
            rhos[i] = rho
            pvals[i] = spearman_pval(rho, N)
    return rhos, pvals


def _precompute_seq_structure(levels, seq_ids):
    unique_seqs = np.unique(seq_ids)
    seq_len = None
    rows = []
    for sid in unique_seqs:
        mask = seq_ids == sid
        lv = levels[mask]
        idx = np.where(mask)[0]
        order = np.argsort(lv)
        rows.append(idx[order])
        if seq_len is None:
            seq_len = len(order)
    rows = [r for r in rows if len(r) == seq_len]
    return np.array(rows, dtype=np.intp), len(rows), seq_len


def _strict_mono_tau_batch(scores_batch, sorted_idx):
    B, _ = scores_batch.shape
    n_seqs, seq_len = sorted_idx.shape
    C = seq_len * (seq_len - 1) // 2
    ss = scores_batch[:, sorted_idx]
    strict = np.all(np.diff(ss, axis=2) > 0, axis=2).mean(axis=1).astype(np.float64)
    tau = np.zeros(B, dtype=np.float64)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            tau += (ss[:, :, i] < ss[:, :, j]).sum(axis=1) - \
                   (ss[:, :, i] > ss[:, :, j]).sum(axis=1)
    tau /= n_seqs * C
    return strict, tau


# ===========================================================================
# Sweep GPU + Spearman CPU
# ===========================================================================

def sweep_gammas(distances: dict, gammas: np.ndarray,
                 levels: np.ndarray, seq_ids: np.ndarray,
                 device: str, batch_size: int = SWEEP_BATCH) -> dict:
    G = len(gammas)
    lr_c = rankdata(levels).astype(np.float64)
    lr_c -= lr_c.mean()
    lr_std = np.sqrt((lr_c ** 2).sum())

    print("  Pré-calcul structure séquences…", flush=True)
    sorted_idx, n_seqs, seq_len = _precompute_seq_structure(levels, seq_ids)
    print(f"  {n_seqs} séquences × {seq_len} niveaux", flush=True)

    gammas_t = torch.from_numpy(gammas.astype(np.float32)).to(device)
    results = {}

    for l in sorted(distances.keys()):
        D_rr_np, D_rq_np = distances[l]
        D_rr_t = torch.from_numpy(D_rr_np.astype(np.float32)).to(device)
        D_rq_t = torch.from_numpy(D_rq_np.astype(np.float32)).to(device)

        all_rhos   = np.empty(G)
        all_pvals  = np.empty(G)
        all_strict = np.empty(G)
        all_tau    = np.empty(G)

        for start in range(0, G, batch_size):
            end = min(start + batch_size, G)
            g = gammas_t[start:end]             # (B,)
            B = g.shape[0]

            # K_rr : mean_ij exp(-g * D_rr[i,j])
            # Shape: (B, n_r, n_r) → mean(dim=1,2)
            # D_rr is (n_r, n_r), D_rq is (n_r, n_q)
            K_rr_mean = torch.exp(-g[:, None, None] * D_rr_t[None]).mean(dim=(1, 2))  # (B,)

            # K_rq_col : mean over anchors for each query
            K_rq = torch.exp(-g[:, None, None] * D_rq_t[None]).mean(dim=1)            # (B, n_q)

            scores_batch_t = 1.0 + K_rr_mean[:, None] - 2.0 * K_rq                   # (B, n_q)
            scores_batch = scores_batch_t.cpu().numpy().astype(np.float64)

            rhos_b, pvals_b = fast_spearman_batch(scores_batch, lr_c, lr_std)
            all_rhos[start:end]  = rhos_b
            all_pvals[start:end] = pvals_b
            strict_b, tau_b = _strict_mono_tau_batch(scores_batch, sorted_idx)
            all_strict[start:end] = strict_b
            all_tau[start:end]    = tau_b

        results[l] = {
            "rho":    all_rhos,
            "pval":   all_pvals,
            "strict": all_strict,
            "tau":    all_tau,
        }
        rho_best = float(np.nanmax(all_rhos))
        print(f"  L{l:02d} {SHORT_NAMES[l]:<18}  ρ max = {rho_best:+.4f}", flush=True)

    return results


# ===========================================================================
# Figures
# ===========================================================================

TITLE_SUFFIX = (
    "independent anchor : 200 × coco_5000  ·  "
    "queries : COCO-degraded  ·  standardisation ON"
)


def _median_gammas(layers):
    mg = {}
    for l in layers:
        p = NEW_CACHE / f"sd_vae_indep_l{l:02d}_Drr.npy"
        if p.exists():
            D = np.load(p).astype(np.float64)
            nz = D[D > 0]
            if len(nz):
                g_med = float(1.0 / (2.0 * np.median(nz)))
                mg[l] = {"med": g_med, "10x": 10.0 * g_med, "50x": 50.0 * g_med}
    return mg


def plot_rho_vs_gamma(results, gammas, out_dir):
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
            # ▲ — médiane heuristique
            gi = int(np.argmin(np.abs(gammas - med_g[l]["med"])))
            rho_m = float(rhos[gi]) if np.isfinite(rhos[gi]) else np.nan
            if np.isfinite(rho_m):
                ax.scatter([gammas[gi]], [rho_m], color=cmap(idx), s=70,
                           zorder=6, marker="^", edgecolors="black", linewidths=0.6)
            # ■ — 50× médiane heuristique
            gi50 = int(np.argmin(np.abs(gammas - med_g[l]["50x"])))
            rho_50x = float(rhos[gi50]) if np.isfinite(rhos[gi50]) else np.nan
            if np.isfinite(rho_50x):
                ax.scatter([gammas[gi50]], [rho_50x], color=cmap(idx), s=70,
                           zorder=6, marker="s", edgecolors="black", linewidths=0.6)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("γ (log scale)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SD-VAE — Spearman ρ as a function of γ  (1 000 values, log grid)\n"
                 + TITLE_SUFFIX, fontsize=11, fontweight="bold")
    extra = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=6, markeredgecolor="black", markeredgewidth=0.4,
               label="Optimal γ (best ρ)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=7, markeredgecolor="black", markeredgewidth=0.6,
               label="γ = 1/(2·median(D_rr))"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None",
               markersize=7, markeredgecolor="black", markeredgewidth=0.6,
               label="γ = 50 × 1/(2·median(D_rr))"),
    ]
    layer_leg = ax.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.85)
    ax.add_artist(layer_leg)
    ax.legend(handles=extra, fontsize=8, loc="upper left",
              framealpha=0.9, bbox_to_anchor=(0.0, 1.0))
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    out = out_dir / "11_sdvae_indep_anchor_rho_vs_gamma.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return best_per_layer


def plot_heatmap(results, gammas, out_dir):
    layers = sorted(results.keys())
    G = len(gammas)
    n_display, step = 100, max(1, G // 100)
    g_idx = np.arange(0, G, step)
    g_d = gammas[g_idx]
    matrix = np.full((len(layers), len(g_idx)), np.nan)
    for i, l in enumerate(layers):
        matrix[i] = results[l]["rho"][g_idx]
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=1.0, origin="upper")
    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.02, pad=0.02)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l:02d} {SHORT_NAMES.get(l, '')}" for l in layers], fontsize=8)
    xs = max(1, len(g_idx) // 10)
    ax.set_xticks(range(0, len(g_idx), xs))
    ax.set_xticklabels([f"{g_d[j]:.1e}" for j in range(0, len(g_idx), xs)],
                        rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("γ", fontsize=11)
    ax.set_title("SD-VAE — Heatmap ρ(layer × γ)  ·  1 000 gammas\n" + TITLE_SUFFIX,
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "12_sdvae_indep_anchor_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_pvalues(results, gammas, out_dir, n_tests):
    layers = sorted(results.keys())
    n_layers = len(layers)
    cmap = plt.get_cmap("plasma", n_layers)
    N = 5000
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.38)
    ax_c = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    ax_c.set_facecolor("#f9f9f9")
    for idx, l in enumerate(layers):
        pvals = results[l]["pval"]
        rhos  = results[l]["rho"]
        valid = np.isfinite(pvals) & (pvals > 0)
        if not valid.any():
            continue
        logp = -np.log10(np.clip(pvals[valid], 1e-300, 1.0))
        ax_c.plot(gammas[valid], logp, color=cmap(idx), linewidth=1.8, alpha=0.85,
                  label=f"L{l:02d} {SHORT_NAMES.get(l, '')}")
        bi = int(np.nanargmax(rhos))
        if valid[bi]:
            lp = float(-np.log10(max(float(pvals[bi]), 1e-300)))
            ax_c.scatter([gammas[bi]], [lp], color=cmap(idx), s=55, zorder=6,
                         edgecolors="black", linewidths=0.5)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("γ  (log scale)", fontsize=13)
    ax_c.set_ylabel("−log₁₀(p-value)", fontsize=13)
    ax_c.set_title("−log₁₀(p-value) of Spearman ρ  vs  γ  —  all encoder layers\n"
                   "Dots = optimal γ (best ρ) per layer", fontsize=12, fontweight="bold")
    ax_c.legend(fontsize=9, ncol=2, loc="lower left", framealpha=0.9)
    ax_c.grid(True, which="both", alpha=0.25)
    ax_t.axis("off")
    table_data = []
    for l in layers:
        rl = results[l]["rho"]
        pl = results[l]["pval"]
        bi = int(np.nanargmax(rl))
        rb = float(rl[bi])
        pb = float(pl[bi])
        gb = float(gammas[bi])
        ts = rb * np.sqrt(N - 2) / np.sqrt(1 - rb**2) if (np.isfinite(rb) and abs(rb) < 1) else np.inf
        ps = f"{pb:.2e}" if (np.isfinite(pb) and pb > 0) else "< 1e-300"
        table_data.append([f"L{l:02d}", SHORT_NAMES.get(l, ""),
                           f"{gb:.3e}", f"{rb:+.4f}",
                           f"{ts:.1f}" if np.isfinite(ts) else "∞", ps])
    cols = ["Layer", "Layer name", "Optimal γ", "ρ max", "t-stat", "p-value"]
    tbl = ax_t.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(range(len(cols)))
    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(0.052)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)
    for i in range(n_layers):
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor("#ffffff")
    ax_t.set_title(f"Spearman significance tests — best γ per layer\n"
                   f"N = {N:,} images  (200 refs × 5 dist. types × 5 levels)",
                   fontsize=12, fontweight="bold", pad=16)
    fig.suptitle("SD-VAE — Statistical Significance  (1 000-gamma sweep)\n" + TITLE_SUFFIX,
                 fontsize=12, fontweight="bold", y=1.02)
    out = out_dir / "13_sdvae_indep_anchor_pvalue.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def export_csv(results, gammas, out_dir):
    rows = []
    for l in sorted(results.keys()):
        r = results[l]
        for gi in range(len(gammas)):
            rows.append({
                "layer":        l,
                "layer_name":   SHORT_NAMES.get(l, ""),
                "gamma":        float(gammas[gi]),
                "spearman_rho": float(r["rho"][gi])   if np.isfinite(r["rho"][gi])   else "",
                "pval":         float(r["pval"][gi])  if np.isfinite(r["pval"][gi])  else "",
                "strict_mono":  float(r["strict"][gi]) if np.isfinite(r["strict"][gi]) else "",
                "kendall_tau":  float(r["tau"][gi])   if np.isfinite(r["tau"][gi])   else "",
            })
    out = out_dir / "sdvae_indep_anchor_gamma_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    gammas = np.logspace(-9, -1, 1000)

    # Anchor paths
    a_paths = sorted(COCO5K_DIR.glob("*.jpg")) + sorted(COCO5K_DIR.glob("*.png"))
    if len(a_paths) < N_ANCHORS:
        raise FileNotFoundError(f"Pas assez d'images dans {COCO5K_DIR}")
    a_paths = [str(p) for p in a_paths[:N_ANCHORS]]
    print(f"Anchor : {len(a_paths)} images ({COCO5K_DIR})")

    print("\nChargement métadonnées…")
    levels, seq_ids = load_levels()
    print(f"  {len(levels)} images  ·  {len(np.unique(seq_ids))} séquences")

    # ── Extraction features (une seule passe modèle) ─────────────────────
    print("\nExtraction features SD-VAE (toutes layers en 1 passe)…")
    t0 = time.time()
    extractor = MultiLayerExtractor(device=device)
    distances = build_all_distances(extractor, a_paths, device)
    del extractor
    torch.cuda.empty_cache()
    t_extr = time.time() - t0
    print(f"  Features + distances : {t_extr:.1f}s ({t_extr/60:.1f} min)")

    # ── Sweep GPU ────────────────────────────────────────────────────────
    print(f"\nSweep 1 000 gammas × {len(distances)} layers (GPU)…")
    t1 = time.time()
    results = sweep_gammas(distances, gammas, levels, seq_ids, device)
    t_sw = time.time() - t1
    print(f"  Sweep : {t_sw:.1f}s ({t_sw/60:.1f} min)")

    n_tests = len(results) * len(gammas)

    print("\n--- Figure 1 : ρ(γ) ---")
    best = plot_rho_vs_gamma(results, gammas, PLOTS_DIR)

    print("\n--- Figure 2 : Heatmap ---")
    plot_heatmap(results, gammas, PLOTS_DIR)

    print("\n--- Figure 3 : p-values ---")
    plot_pvalues(results, gammas, PLOTS_DIR, n_tests)

    print("\n--- Export CSV ---")
    export_csv(results, gammas, PLOTS_DIR)

    # Résumé
    print("\n" + "=" * 68)
    print(f"  {'Layer':<6} {'Nom':<18} {'γ optimal':>10}  {'ρ max':>8}  {'p-value':>12}")
    print("  " + "-" * 62)
    for l, (g_opt, rho_opt) in sorted(best.items(), key=lambda x: -x[1][1]):
        pv = float(results[l]["pval"][int(np.nanargmax(results[l]["rho"]))])
        pv_s = f"{pv:.2e}" if (np.isfinite(pv) and pv > 0) else "~0"
        print(f"  L{l:02d}   {SHORT_NAMES.get(l, ''):<18} {g_opt:>10.2e}  {rho_opt:>+8.4f}  {pv_s:>12}")
    print("=" * 68)
    total = t_extr + t_sw
    print(f"\nTemps total : {total:.1f}s ({total/60:.1f} min)")
    print(f"Figures dans : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
