#!/usr/bin/env python3
"""
Test rapide : sd_vae L11, N_REFS=50, 200 groupes cumulatifs.
Affiche Spearman ρ pour γ_med et γ_med×10.
Réutilise le cache anchor sd_vae existant.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

BASE_DIR  = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Import distortion helpers from the sweep script
sys.path.insert(0, str(Path(__file__).parent))
from sweep_synthetic_kadid import (
    build_groups, apply_distortion_sequence,
    get_coco_split, MultiLayerExtractor,
    compute_gamma_med, mmd2_all_gammas, CACHE_DIR,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER  = 11
BS     = 8

def main():
    t0 = time.time()
    print(f"Device: {DEVICE}  |  Layer: L{LAYER}  |  batch_size: {BS}")

    # ── Anchor features (depuis cache) ──────────────────────────────────
    anc_cache = CACHE_DIR / "sd_vae_anc.npz"
    feat_anc = np.load(anc_cache)[f"L{LAYER:02d}"]   # (1000, D)
    print(f"Anchor  : {feat_anc.shape}  ({time.time()-t0:.1f}s)")

    # ── γ_med ────────────────────────────────────────────────────────────
    g_med = compute_gamma_med(feat_anc, device=DEVICE)
    gammas = [g_med, g_med * 10.0]
    print(f"γ_med   : {g_med:.4e}   γ_med×10 : {gammas[1]:.4e}")

    # ── Images ref ──────────────────────────────────────────────────────
    ref_paths, _ = get_coco_split()   # 50 refs
    groups = build_groups()           # 200 groupes

    # ── Chargement modèle ───────────────────────────────────────────────
    ext = MultiLayerExtractor("sd_vae", [LAYER], DEVICE)
    anc_t  = torch.tensor(feat_anc, dtype=torch.float32, device=DEVICE)

    # ── Extraction dist + scoring ────────────────────────────────────────
    mmd2_med  = np.zeros(len(groups))
    mmd2_10x  = np.zeros(len(groups))

    for k, grp in enumerate(groups):
        dist_imgs = [apply_distortion_sequence(Image.open(p).convert("RGB"),
                                               grp["operations"])
                     for p in ref_paths]
        feats = ext.extract(dist_imgs, batch_size=BS)
        fk = torch.tensor(feats[LAYER], dtype=torch.float32, device=DEVICE)
        scores = mmd2_all_gammas(fk, anc_t, gammas)
        mmd2_med[k] = scores[0]
        mmd2_10x[k] = scores[1]

        if (k + 1) % 25 == 0 or k == 0:
            print(f"  group {k+1:3d}/200  MMD²_med={scores[0]:.4e}  "
                  f"MMD²_10x={scores[1]:.4e}  ({time.time()-t0:.0f}s)")

    ranks = np.arange(1, len(groups) + 1, dtype=float)
    rho_med, _ = spearmanr(ranks, mmd2_med)
    rho_10x, _ = spearmanr(ranks, mmd2_10x)

    print("\n" + "=" * 50)
    print(f"  sd_vae L{LAYER}  γ_med    : Spearman ρ = {rho_med:+.4f}")
    print(f"  sd_vae L{LAYER}  γ_med×10 : Spearman ρ = {rho_10x:+.4f}")
    print(f"  Temps total : {time.time()-t0:.1f}s")
    print("=" * 50)

if __name__ == "__main__":
    main()
