#!/usr/bin/env python3
"""
Mini-test : sd_vae L11, 20 groupes, 10 refs → Spearman ρ pour γ_med et γ_med×10.
Force offline mode pour éviter les requêtes HuggingFace.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sweep_synthetic_kadid import (
    build_groups, apply_distortion_sequence,
    get_coco_split, MultiLayerExtractor,
    compute_gamma_med, mmd2_all_gammas, CACHE_DIR,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER  = 11
BS     = 16
N_GROUPS_TEST = 20   # premiers 20 groupes (rapide)
N_REFS_TEST   = 50

def main():
    t0 = time.time()
    print(f"Device: {DEVICE}  Layer: L{LAYER}  groups: {N_GROUPS_TEST}  refs: {N_REFS_TEST}", flush=True)

    # Anchor depuis cache
    anc_cache = CACHE_DIR / "sd_vae_anc.npz"
    feat_anc = np.load(anc_cache)[f"L{LAYER:02d}"]   # (1000, D)
    print(f"Anchor: {feat_anc.shape}  ({time.time()-t0:.1f}s)", flush=True)

    g_med = compute_gamma_med(feat_anc, device=DEVICE)
    gammas = [g_med, g_med * 10.0]
    print(f"γ_med={g_med:.3e}  γ_med×10={gammas[1]:.3e}", flush=True)

    ref_paths, _ = get_coco_split()
    ref_paths = ref_paths[:N_REFS_TEST]
    groups = build_groups()[:N_GROUPS_TEST]

    print(f"Chargement modèle...", flush=True)
    ext = MultiLayerExtractor("sd_vae", [LAYER], DEVICE)
    anc_t = torch.tensor(feat_anc, dtype=torch.float32, device=DEVICE)
    print(f"Modèle chargé  ({time.time()-t0:.1f}s)", flush=True)

    mmd2_med = np.zeros(len(groups))
    mmd2_10x = np.zeros(len(groups))

    for k, grp in enumerate(groups):
        dist_imgs = [apply_distortion_sequence(Image.open(p).convert("RGB"), grp["operations"])
                     for p in ref_paths]
        feats = ext.extract(dist_imgs, batch_size=BS)
        fk = torch.tensor(feats[LAYER], dtype=torch.float32, device=DEVICE)
        scores = mmd2_all_gammas(fk, anc_t, gammas)
        mmd2_med[k] = scores[0]
        mmd2_10x[k] = scores[1]
        print(f"  grp {k+1:2d}/{len(groups)}  MMD²_med={scores[0]:.3e}  MMD²_10x={scores[1]:.3e}  ({time.time()-t0:.0f}s)", flush=True)

    ranks = np.arange(1, len(groups) + 1, dtype=float)
    rho_med, _ = spearmanr(ranks, mmd2_med)
    rho_10x, _ = spearmanr(ranks, mmd2_10x)

    print("=" * 50, flush=True)
    print(f"  γ_med    : Spearman ρ = {rho_med:+.4f}", flush=True)
    print(f"  γ_med×10 : Spearman ρ = {rho_10x:+.4f}", flush=True)
    print(f"  Temps total : {time.time()-t0:.1f}s", flush=True)
    print("=" * 50, flush=True)

if __name__ == "__main__":
    main()
