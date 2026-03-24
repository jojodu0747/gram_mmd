#!/usr/bin/env python3
"""
ρ vs γ (log scale) — même style que 08_sdvae_rho_vs_gamma.png
mais avec les données de synthetic_kadid_sweep.

Génère :
  rho_vs_gamma_{backbone}.png   pour chaque backbone dans BACKBONES
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from gram_mmd.config import BACKBONE_CONFIGS
from sweep_synthetic_kadid import build_groups, CACHE_DIR, N_GROUPS

# ── Configuration ─────────────────────────────────────────────────────────────
# Convention rate : kernel = exp(-γ · d²)   (même que gamma_sweep_sdvae.py)

GAMMAS      = np.logspace(-3, 2, 1000)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR     = BASE_DIR / "results" / "synthetic_kadid_sweep" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = ["sd_vae", "dinov2_vitb14"]
DIST_CACHE_SUFFIX = "_dist_indep"   # "_dist" pour cumulatif, "_dist_indep" pour indépendant


# ── Helpers ────────────────────────────────────────────────────────────────────

def layer_label(backbone: str, layer: int) -> str:
    names = BACKBONE_CONFIGS.get(backbone, {}).get("layer_names", {})
    return f"L{layer:02d} {names.get(layer, '?')}"


def pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """
    X : (N, D) float32/64
    Returns D_rr (N, N) pairwise squared Euclidean distances.
    """
    X = X.astype(np.float64)
    sq = (X ** 2).sum(axis=1, keepdims=True)
    D = sq + sq.T - 2.0 * (X @ X.T)
    np.clip(D, 0.0, None, out=D)
    return D


def cross_sq_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    X : (Nx, D),  Y : (Ny, D)
    Returns D_xy (Nx, Ny) pairwise squared distances.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    sq_x = (X ** 2).sum(axis=1, keepdims=True)
    sq_y = (Y ** 2).sum(axis=1, keepdims=True)
    D = sq_x + sq_y.T - 2.0 * (X @ Y.T)
    np.clip(D, 0.0, None, out=D)
    return D


# ── Core sweep ─────────────────────────────────────────────────────────────────

def compute_rho_vs_gamma(backbone: str) -> dict:
    """
    Returns {layer: {"rhos": np.array(G,), "gamma_med": float}}
    """
    anc_cache  = CACHE_DIR / f"{backbone}_anc.npz"
    dist_cache = CACHE_DIR / f"{backbone}{DIST_CACHE_SUFFIX}.npz"

    if not anc_cache.exists() or not dist_cache.exists():
        raise FileNotFoundError(
            f"Cache manquant pour {backbone}.\n"
            f"  attendu: {anc_cache}\n  attendu: {dist_cache}"
        )

    print(f"  Chargement caches {backbone}…", flush=True)
    anc_data  = dict(np.load(anc_cache))
    dist_data = dict(np.load(dist_cache))
    anc_feats  = {int(k[1:]): v for k, v in anc_data.items()}
    dist_feats = {int(k[1:]): v for k, v in dist_data.items()}

    quality_ranks = np.arange(1, N_GROUPS + 1, dtype=np.float64)

    layers = sorted(set(anc_feats.keys()) & set(dist_feats.keys()))
    print(f"  {len(layers)} layers disponibles: {layers}", flush=True)

    G = len(GAMMAS)
    results = {}

    for l in layers:
        print(f"    Layer {l}…", flush=True)

        # ── Standardisation par dimension (convention sweep_coco_degraded) ──
        anc_raw = anc_feats[l].astype(np.float32)
        mu  = anc_raw.mean(axis=0, keepdims=True)
        sig = anc_raw.std(axis=0, keepdims=True) + 1e-8
        anc_std = ((anc_raw - mu) / sig).astype(np.float64)      # (N_anc, D)

        # dist_feats[l] : (N_GROUPS, N_REFS, D)
        dist_raw = dist_feats[l].astype(np.float32)               # (G, R, D)
        dist_std = ((dist_raw - mu) / sig).astype(np.float64)     # (G, R, D)

        # ── Distance matrices ────────────────────────────────────────────────
        print(f"      D_rr…", flush=True)
        D_rr = pairwise_sq_dist(anc_std)                          # (N_anc, N_anc)

        print(f"      D_rq ({N_GROUPS} groupes)…", flush=True)
        # D_rq[k] : (N_refs, N_anc)  cross distances group k ↔ anchor
        D_rq_all = np.zeros((N_GROUPS, dist_std.shape[1],
                             anc_std.shape[0]), dtype=np.float64)
        for k in range(N_GROUPS):
            D_rq_all[k] = cross_sq_dist(dist_std[k], anc_std)

        # ── Gamma médian (convention rate) ───────────────────────────────────
        nonzero = D_rr[D_rr > 0]
        gamma_med = float(1.0 / (2.0 * np.median(nonzero))) if len(nonzero) > 0 else np.nan

        # ── Sweep gammas ─────────────────────────────────────────────────────
        print(f"      Sweep {G} gammas…", flush=True)
        rhos = np.empty(G)

        for gi, g in enumerate(GAMMAS):
            # K_rr_mean : scalar
            K_rr_mean = np.exp(-g * D_rr).mean()
            # K_rq_mean per group : (N_GROUPS,)
            K_rq_means = np.exp(-g * D_rq_all).mean(axis=(1, 2))
            scores = 1.0 + K_rr_mean - 2.0 * K_rq_means
            res = spearmanr(quality_ranks, scores)
            rhos[gi] = res.statistic if hasattr(res, "statistic") else res[0]

        results[l] = {"rhos": rhos, "gamma_med": gamma_med}
        print(f"      ρ_max={np.nanmax(rhos):+.4f}  γ_med={gamma_med:.3e}", flush=True)

    return results


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_backbone(backbone: str, results: dict):
    layers = sorted(results.keys())
    n_layers = len(layers)
    cmap = plt.get_cmap("plasma", n_layers)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f9f9f9")

    for idx, l in enumerate(layers):
        rhos = results[l]["rhos"]
        valid = np.isfinite(rhos)
        if not valid.any():
            continue
        lbl = layer_label(backbone, l)
        ax.plot(GAMMAS[valid], rhos[valid],
                color=cmap(idx), linewidth=1.4, alpha=0.85, label=lbl)

        # Dot at optimal γ (best ρ)
        best_idx = int(np.nanargmax(rhos))
        ax.scatter([GAMMAS[best_idx]], [float(rhos[best_idx])],
                   color=cmap(idx), s=40, zorder=5,
                   edgecolors="black", linewidths=0.4)

        # Triangle at γ_med
        g_med = results[l]["gamma_med"]
        if np.isfinite(g_med) and GAMMAS[0] <= g_med <= GAMMAS[-1]:
            gi_med = int(np.argmin(np.abs(GAMMAS - g_med)))
            rho_med = float(rhos[gi_med])
            if np.isfinite(rho_med):
                ax.scatter([GAMMAS[gi_med]], [rho_med],
                           color=cmap(idx), s=70, zorder=6,
                           marker="^", edgecolors="black", linewidths=0.6)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("γ (log scale)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"{backbone} — Spearman ρ as a function of γ  (1 000 values, log grid)\n"
        f"synthetic_kadid  ·  anchor = 1 000 COCO  ·  200 groups × 50 refs  ·  standardisation ON",
        fontsize=11, fontweight="bold",
    )

    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=6, markeredgecolor="black", markeredgewidth=0.4,
               label="Optimal γ (best ρ)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=7, markeredgecolor="black", markeredgewidth=0.6,
               label="Median heuristic γ = 1/(2·median(D_rr))"),
    ]
    ncol = 2 if n_layers > 10 else 1
    layer_legend = ax.legend(fontsize=7.5, ncol=ncol,
                             loc="lower left", framealpha=0.85)
    ax.add_artist(layer_legend)
    ax.legend(handles=extra_handles, fontsize=8, loc="upper left",
              framealpha=0.9, bbox_to_anchor=(0.0, 1.0))

    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    out = OUT_DIR / f"rho_vs_gamma_{backbone}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Gamma sweep: {len(GAMMAS)} points in [{GAMMAS[0]:.0e}, {GAMMAS[-1]:.0e}]", flush=True)

    for bb in BACKBONES:
        print(f"\n{'='*60}", flush=True)
        print(f"Backbone: {bb}", flush=True)
        results = compute_rho_vs_gamma(bb)
        plot_backbone(bb, results)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
