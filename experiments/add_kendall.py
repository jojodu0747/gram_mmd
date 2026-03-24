#!/usr/bin/env python3
"""
Adds kendall_tau column to summary_full_sweep.csv by recomputing MMD²
from cached features (no model re-inference needed).

Usage:
    python experiments/add_kendall.py
    python experiments/add_kendall.py --csv path/to/summary.csv --device cpu
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler

BASE_DIR  = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "results" / "synthetic_kadid_sweep" / "cache"

N_DIST_TYPES = 20
N_LEVELS     = 10
N_GROUPS     = N_DIST_TYPES * N_LEVELS
GAMMA_MULTS  = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
GAMMA_SUBSAMPLE = 500


def make_groups():
    groups = []
    for t in range(1, N_DIST_TYPES + 1):
        for lv in range(1, N_LEVELS + 1):
            groups.append({"dist_type": t, "level": lv})
    return groups


def compute_gamma_med(feats: np.ndarray, device: str = "cpu") -> float:
    n = min(GAMMA_SUBSAMPLE, feats.shape[0])
    idx = np.random.default_rng(0).choice(feats.shape[0], n, replace=False)
    X = torch.tensor(feats[idx], dtype=torch.float32, device=device)
    D2 = torch.cdist(X, X, p=2).pow(2)
    triu = D2[torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)]
    return float(triu.median())


def precompute_anchor_dists(Y: torch.Tensor):
    n = Y.shape[0]
    D_rr = torch.cdist(Y, Y, p=2).pow(2)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=Y.device), diagonal=1)
    triu_rr = D_rr[mask]
    del D_rr
    return triu_rr


def mmd2_all_gammas(X: torch.Tensor, Y: torch.Tensor, gammas: list,
                    triu_rr: torch.Tensor) -> np.ndarray:
    m = X.shape[0]
    D_qq = torch.cdist(X, X, p=2).pow(2)
    D_rq = torch.cdist(Y, X, p=2).pow(2)
    triu_qq = D_qq[torch.triu(torch.ones(m, m, dtype=torch.bool, device=X.device), diagonal=1)]
    del D_qq
    results = []
    for g in gammas:
        two_g = 2.0 * g
        k_qq = torch.exp(-triu_qq / two_g).mean()
        k_rr = torch.exp(-triu_rr / two_g).mean()
        k_rq = torch.exp(-D_rq  / two_g).mean()
        results.append((k_qq + k_rr - 2 * k_rq).item())
    del D_rq, triu_qq
    return np.array(results)


def compute_kendall_for_layer(anc: np.ndarray, dist_raw: np.ndarray,
                               groups: list, gammas: list,
                               device: str) -> np.ndarray:
    """Returns mean Kendall τ over N_DIST_TYPES, shape (n_gammas,)."""
    # StandardScaler (same as sweep)
    scaler = StandardScaler()
    anc_s  = scaler.fit_transform(anc).astype(np.float32)

    gamma_med = compute_gamma_med(anc_s, device=device)
    gamma_vals = [gamma_med * m for m in gammas]

    anc_t = torch.tensor(anc_s, dtype=torch.float32, device=device)
    del anc_s; gc.collect()
    triu_rr = precompute_anchor_dists(anc_t)

    N_G, N_R, D = dist_raw.shape
    mmd2_matrix = np.zeros((N_G, len(gammas)), dtype=np.float32)

    with torch.no_grad():
        for k in range(N_G):
            row_s = scaler.transform(dist_raw[k]).astype(np.float32)
            X = torch.tensor(row_s, dtype=torch.float32, device=device)
            mmd2_matrix[k] = mmd2_all_gammas(X, anc_t, gamma_vals, triu_rr=triu_rr)
            del X

    del triu_rr, anc_t; gc.collect()

    # Compute mean Kendall τ per gamma
    n_gammas = len(gammas)
    per_type_tau = np.zeros((N_DIST_TYPES, n_gammas))
    for ti, t in enumerate(range(1, N_DIST_TYPES + 1)):
        idx    = [k for k, g in enumerate(groups) if g["dist_type"] == t]
        levels = np.array([groups[k]["level"] for k in idx], dtype=float)
        for gi in range(n_gammas):
            tau, _ = kendalltau(levels, mmd2_matrix[idx, gi])
            per_type_tau[ti, gi] = tau if not np.isnan(tau) else 0.0

    return per_type_tau.mean(axis=0)   # (n_gammas,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
        default=BASE_DIR / "results" / "synthetic_kadid_sweep" / "summary_full_sweep.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    if "kendall" in df.columns:
        print("  'kendall' column already exists — overwriting.")

    groups = make_groups()
    df["kendall"] = np.nan

    backbones = df["backbone"].unique().tolist()
    for bb in backbones:
        bb_dir  = CACHE_DIR / bb
        layers  = sorted(df[df["backbone"] == bb]["layer"].unique().tolist())
        print(f"\n{bb} — {len(layers)} layers")

        for l in layers:
            anc_path = bb_dir / f"anc_L{l}.npz"
            d_npy    = bb_dir / f"dist_L{l}.npy"
            d_npz    = bb_dir / f"dist_L{l}.npz"

            if not anc_path.exists():
                print(f"  L{l}: anchor cache missing, skip")
                continue

            if d_npy.exists():
                dist_raw = np.load(str(d_npy), mmap_mode="r")
            elif d_npz.exists():
                dist_raw = np.load(str(d_npz))["data"]
            else:
                print(f"  L{l}: dist cache missing, skip")
                continue

            anc_raw = np.load(str(anc_path))["data"]
            print(f"  L{l}: anc{anc_raw.shape} dist{dist_raw.shape} ...", end="", flush=True)

            mean_tau = compute_kendall_for_layer(
                anc_raw, dist_raw, groups, GAMMA_MULTS, args.device)

            # Write into df
            for gi, mult in enumerate(GAMMA_MULTS):
                mask = (df["backbone"] == bb) & (df["layer"] == l) & (df["gamma_mult"] == mult)
                df.loc[mask, "kendall"] = mean_tau[gi]

            print(f"  best τ={mean_tau.max():+.4f} @ γ×{GAMMA_MULTS[int(mean_tau.argmax())]}")
            del anc_raw, dist_raw; gc.collect()

    out_path = args.csv
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} with 'kendall' column.")

    # Quick summary
    best = df.loc[df.groupby(["backbone", "layer"])["kendall"].idxmax()]
    best = best.loc[best.groupby("backbone")["kendall"].idxmax()]
    print("\nBest Kendall τ per backbone:")
    for _, row in best.iterrows():
        print(f"  {row['backbone']:20s} L{int(row['layer']):<3d} "
              f"γ×{row['gamma_mult']:<5}  τ={row['kendall']:+.4f}")


if __name__ == "__main__":
    main()
