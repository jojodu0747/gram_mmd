#!/usr/bin/env python3
"""
Trace mean ρ (Spearman, Option B) en fonction de gamma sur 500 points log-scale.

Principe :
  • Distances² calculées UNE SEULE FOIS par (backbone, layer)
  • Pour chaque gamma g : k(x,y) = exp(−d² / 2g)   → vectorisé sur les 500 gammas

Modes :
  --mode best   (défaut) : une courbe par backbone, meilleure layer selon --csv
  --mode all              : toutes les layers, un sous-plot par backbone

Usage :
  python experiments/plot_gamma_curve.py
  python experiments/plot_gamma_curve.py --mode all --backbones sd_vae dc_ae
  python experiments/plot_gamma_curve.py --n-gammas 200 --gamma-range 1e-4 1e3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

BASE_DIR  = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from gram_mmd.config import BACKBONE_CONFIGS

def layer_label(backbone: str, layer: int) -> str:
    names = BACKBONE_CONFIGS.get(backbone, {}).get("layer_names", {})
    return names.get(layer, "?")

CACHE_DIR = BASE_DIR / "results" / "synthetic_kadid_sweep" / "cache"
OUT_DIR   = BASE_DIR / "results" / "synthetic_kadid_sweep" / "plots_new"

N_DIST_TYPES = 20
N_LEVELS     = 10
N_GROUPS     = N_DIST_TYPES * N_LEVELS   # 200
GAMMA_SUBSAMPLE = 500

BACKBONES_ALL = ["sd_vae", "dc_ae", "lpips_vgg", "vgg19", "resnet50", "dinov2_vitb14"]
BB_COLORS = {
    "sd_vae":        "#e41a1c",
    "dc_ae":         "#377eb8",
    "lpips_vgg":     "#4daf4a",
    "vgg19":         "#984ea3",
    "resnet50":      "#a65628",
    "dinov2_vitb14": "#ff7f00",
}

# ── groups (20 types × 10 levels) ────────────────────────────────────────────

def build_groups():
    groups = []
    for t in range(1, N_DIST_TYPES + 1):
        for lv in range(1, N_LEVELS + 1):
            groups.append({"dist_type": t, "level": lv})
    return groups

GROUPS = build_groups()

# ── helpers ───────────────────────────────────────────────────────────────────

def gamma_med(feats: np.ndarray, subsample: int = GAMMA_SUBSAMPLE) -> float:
    n = min(subsample, feats.shape[0])
    idx = np.random.default_rng(0).choice(feats.shape[0], n, replace=False)
    X = torch.tensor(feats[idx], dtype=torch.float32)
    D2 = torch.cdist(X, X, p=2).pow(2)
    triu = D2[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
    return float(triu.median())


def load_layer(backbone: str, layer: int):
    """
    Retourne (anc_scaled, dist_raw, scaler).
    dist_raw reste en mmap — ne jamais pré-allouer le tableau entier (peut faire 21 GB).
    """
    bb_dir  = CACHE_DIR / backbone
    a_path  = bb_dir / f"anc_L{layer}.npz"
    d_npy   = bb_dir / f"dist_L{layer}.npy"
    d_npz   = bb_dir / f"dist_L{layer}.npz"

    if not a_path.exists():
        return None, None, None

    anc_raw = np.load(str(a_path))["data"]

    if d_npy.exists():
        dist_raw = np.load(str(d_npy), mmap_mode="r")
    elif d_npz.exists():
        dist_raw = np.load(str(d_npz))["data"]
    else:
        return None, None, None

    scaler = StandardScaler()
    anc = scaler.fit_transform(anc_raw).astype(np.float32)
    return anc, dist_raw, scaler


def mean_rho_curve(anc: np.ndarray, dist_raw, scaler,
                   gammas: np.ndarray, gamma_chunk: int = 50) -> np.ndarray:
    """
    Pour chaque gamma, calcule mean ρ (Option B).
    dist_raw : mmap (N_GROUPS, N_REFS, D) — chargé groupe par groupe pour éviter OOM.
    """
    G = len(gammas)
    anc_t  = torch.tensor(anc,  dtype=torch.float32)
    gam_t  = torch.tensor(gammas, dtype=torch.float64)

    # Precompute anchor pairwise distances² (une fois)
    n_anc = anc_t.shape[0]
    D_rr  = torch.cdist(anc_t, anc_t, p=2).pow(2)
    triu_mask = torch.triu(torch.ones(n_anc, n_anc, dtype=torch.bool), diagonal=1)
    triu_rr   = D_rr[triu_mask].double()
    del D_rr

    K_rr = torch.zeros(G, dtype=torch.float64)
    for i in range(0, G, gamma_chunk):
        g_c = gam_t[i:i+gamma_chunk]
        K   = torch.exp(-triu_rr[:, None] / (2.0 * g_c[None, :]))
        K_rr[i:i+gamma_chunk] = K.mean(dim=0)
    del triu_rr

    # Accumulate MMD² per group — dist chargé un groupe à la fois (pas de pré-alloc)
    mmd2_mat = np.zeros((N_GROUPS, G), dtype=np.float32)

    for k in range(N_GROUPS):
        # Charge et scale un seul groupe (50 × D) depuis le mmap
        group_raw = np.array(dist_raw[k], dtype=np.float32)   # force copy from mmap
        group_sc  = scaler.transform(group_raw).astype(np.float32)
        X = torch.tensor(group_sc, dtype=torch.float32)
        del group_raw, group_sc
        m = X.shape[0]

        D_qq  = torch.cdist(X, X, p=2).pow(2)
        triu_qq = D_qq[torch.triu(torch.ones(m, m, dtype=torch.bool), diagonal=1)].double()
        del D_qq

        D_rq  = torch.cdist(anc_t, X, p=2).pow(2).double()

        K_qq = torch.zeros(G, dtype=torch.float64)
        K_rq = torch.zeros(G, dtype=torch.float64)
        for i in range(0, G, gamma_chunk):
            g_c = gam_t[i:i+gamma_chunk]
            K_qq[i:i+gamma_chunk] = torch.exp(
                -triu_qq[:, None] / (2.0 * g_c[None, :])).mean(0)
            K_rq[i:i+gamma_chunk] = torch.exp(
                -D_rq[:, :, None] / (2.0 * g_c[None, None, :])).mean((0, 1))

        mmd2_mat[k] = (K_qq + K_rr - 2.0 * K_rq).float().numpy()
        del X, triu_qq, D_rq, K_qq, K_rq

    # Compute mean ρ (Option B)
    rhos = np.zeros((N_DIST_TYPES, G))
    for ti, t in enumerate(range(1, N_DIST_TYPES + 1)):
        idx    = [k for k, g in enumerate(GROUPS) if g["dist_type"] == t]
        levels = np.array([GROUPS[k]["level"] for k in idx], dtype=float)
        for gi in range(G):
            rho, _ = spearmanr(levels, mmd2_mat[idx, gi])
            rhos[ti, gi] = rho if not np.isnan(rho) else 0.0

    return rhos.mean(axis=0)   # (G,)


# ── Best layer per backbone from CSV ─────────────────────────────────────────

def best_layers_from_csv(csv_path: Path, backbones: list) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for bb in backbones:
        sub = df[df["backbone"] == bb]
        if sub.empty:
            continue
        best = sub.loc[sub["spearman"].idxmax()]
        out[bb] = int(best["layer"])
        print(f"  {bb}: best layer = L{int(best['layer'])} "
              f"(ρ={best['spearman']:.3f} @ γ×{best['gamma_mult']})")
    return out


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_best_per_backbone(results: dict, out_path: Path):
    """results: {backbone: (gammas, mean_rho, gamma_med_val, best_layer)}"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for bb, (gammas, rho, gmed, layer) in results.items():
        mult = gammas / gmed
        color = BB_COLORS.get(bb, "gray")
        ax.plot(mult, rho, color=color, linewidth=1.8,
                label=f"{bb} L{layer}")
        # marker at max
        idx_max = np.argmax(rho)
        ax.plot(mult[idx_max], rho[idx_max], "o", color=color, markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("γ multiplier (γ = mult × γ_med)", fontsize=11)
    ax.set_ylabel("mean Spearman ρ  (20 types)", fontsize=11)
    ax.set_title("ρ vs γ — meilleure layer par backbone\n(500 gammas log-scale)", fontsize=12)
    ax.axhline(0.8, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0.9, color="gray", linewidth=0.8, linestyle=":",  alpha=0.6)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(-0.1, 1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_all_layers(bb_results: dict, backbone: str, out_path: Path):
    """
    bb_results: {layer: (gammas, mean_rho, gmed)}
    X-axis: γ = 1/(2g)  — convention exp(−γ·d²)
    Markers: dot = optimal γ, triangle = γ_med = 1/(2·gmed)
    """
    n = len(bb_results)
    cmap = plt.colormaps.get_cmap("plasma")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    layers_sorted = sorted(bb_results.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#f9f9f9")

    for i, l in enumerate(layers_sorted):
        gammas, rho, gmed = bb_results[l]
        # Convert to γ_rbf = 1/(2g) — note: gammas sorted ascending → γ_rbf descending
        gamma_rbf = 1.0 / (2.0 * gammas)   # (G,) decreasing
        valid = np.isfinite(rho)
        ax.plot(gamma_rbf[valid], rho[valid],
                color=colors[i], linewidth=1.3, alpha=0.85,
                label=f"L{l:02d} {layer_label(backbone, l)}")

        # Dot at optimal γ_rbf
        best_i = int(np.nanargmax(rho))
        ax.scatter([gamma_rbf[best_i]], [float(rho[best_i])],
                   color=colors[i], s=40, zorder=5,
                   edgecolors="black", linewidths=0.4)

        # Triangle at γ_med = 1/(2·gmed)
        g_med_rbf = 1.0 / (2.0 * gmed)
        med_i = int(np.argmin(np.abs(gamma_rbf - g_med_rbf)))
        ax.scatter([gamma_rbf[med_i]], [float(rho[med_i])],
                   color=colors[i], s=70, zorder=6,
                   marker="^", edgecolors="black", linewidths=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("γ (log scale)", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title(f"{backbone} — Spearman ρ as a function of γ",
                 fontsize=17, fontweight="bold", pad=15)
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax.axhline(0.8, color="steelblue", linewidth=0.8, linestyle="--", alpha=0.5)

    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=6, markeredgecolor="black", markeredgewidth=0.4,
               label="Optimal γ (best ρ)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=7, markeredgecolor="black", markeredgewidth=0.6,
               label="γ_med = 1 / (2·median(D²))"),
    ]
    ncol = 2 if n > 10 else 1
    layer_leg = ax.legend(fontsize=7.5, ncol=ncol, loc="lower left", framealpha=0.85)
    ax.add_artist(layer_leg)
    ax.legend(handles=extra, fontsize=8, loc="upper right", framealpha=0.9)

    from matplotlib.ticker import LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_minor_locator(LogLocator(base=2.0, subs=()))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.2e}" if x < 0.01 or x > 999 else f"{x:.4g}")
    ax.tick_params(axis="x", which="major", labelsize=7, rotation=45)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.set_ylim(-0.1, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
        default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "summary_full_sweep.csv")
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--mode", choices=["best", "all"], default="best",
        help="best=meilleure layer par backbone; all=toutes layers")
    parser.add_argument("--backbones", nargs="+", default=BACKBONES_ALL)
    parser.add_argument("--n-gammas", type=int, default=500)
    parser.add_argument("--gamma-range", nargs=2, type=float,
        default=[1e-4, 1e3],
        metavar=("MULT_MIN", "MULT_MAX"),
        help="Plage des multiplicateurs gamma (log-space)")
    parser.add_argument("--plot-only", action="store_true",
        help="Skip computation, load saved rho curves from cache and replot only")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "best":
        print("Mode: best layer per backbone")
        print(f"Loading {args.csv}")
        best = best_layers_from_csv(args.csv, args.backbones)

        results = {}
        for bb, layer in best.items():
            print(f"\n{bb} L{layer} — loading cache…")
            anc, dist_raw, scaler = load_layer(bb, layer)
            if anc is None:
                print(f"  Cache manquant pour {bb} L{layer}, skip.")
                continue

            gmed = gamma_med(anc)
            gammas = np.logspace(
                np.log10(gmed * args.gamma_range[0]),
                np.log10(gmed * args.gamma_range[1]),
                args.n_gammas,
            )
            print(f"  γ_med={gmed:.3e} → plage [{gammas[0]:.3e}, {gammas[-1]:.3e}]")
            print(f"  Computing {args.n_gammas} gammas…", flush=True)
            rho = mean_rho_curve(anc, dist_raw, scaler, gammas)
            print(f"  max ρ={rho.max():.4f} @ γ×{gammas[np.argmax(rho)]/gmed:.4f}")
            results[bb] = (gammas, rho, gmed, layer)
            del anc, dist_raw, scaler

        out = args.outdir / "08_gamma_curve_best.png"
        plot_best_per_backbone(results, out)

    else:  # all layers
        print("Mode: all layers per backbone")
        for bb in args.backbones:
            rho_cache = args.outdir / f"rho_curves_{bb}.npz"

            if args.plot_only:
                if not rho_cache.exists():
                    print(f"  No rho cache found for {bb} at {rho_cache}, skipping.")
                    continue
                print(f"\n{bb}: loading rho cache from {rho_cache}…")
                data = np.load(str(rho_cache), allow_pickle=True)
                bb_results = {int(k): (data[f"{k}_gammas"], data[f"{k}_rho"], float(data[f"{k}_gmed"]))
                              for k in data["layers"]}
            else:
                bb_dir = CACHE_DIR / bb
                layers = sorted(int(p.stem.split("_L")[1])
                                for p in bb_dir.glob("anc_L*.npz"))
                print(f"\n{bb}: {len(layers)} layers")
                bb_results = {}
                for layer in layers:
                    anc, dist_raw, scaler = load_layer(bb, layer)
                    if anc is None:
                        continue
                    gmed = gamma_med(anc)
                    gammas = np.logspace(
                        np.log10(gmed * args.gamma_range[0]),
                        np.log10(gmed * args.gamma_range[1]),
                        args.n_gammas,
                    )
                    print(f"  L{layer} (D={anc.shape[1]})…", end=" ", flush=True)
                    rho = mean_rho_curve(anc, dist_raw, scaler, gammas)
                    print(f"max ρ={rho.max():.4f}")
                    bb_results[layer] = (gammas, rho, gmed)
                    del anc, dist_raw, scaler

                # Save rho curves so future style changes don't need recomputation
                if bb_results:
                    save_dict = {"layers": np.array(list(bb_results.keys()))}
                    for layer, (gammas, rho, gmed) in bb_results.items():
                        save_dict[f"{layer}_gammas"] = gammas
                        save_dict[f"{layer}_rho"]    = rho
                        save_dict[f"{layer}_gmed"]   = np.float64(gmed)
                    np.savez(str(rho_cache), **save_dict)
                    print(f"  Rho curves saved to {rho_cache}")

            if bb_results:
                out = args.outdir / f"08_gamma_curve_{bb}.png"
                plot_all_layers(bb_results, bb, out)


if __name__ == "__main__":
    main()
