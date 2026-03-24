"""
GMMD_test — config.py
=====================
Configuration for the Gram-MMD distance tool.

Edit the settings below to match your needs.
"""

import torch

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────
# "cuda" for GPU, "cpu" for CPU-only machines.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# STANDARDISATION
# ──────────────────────────────────────────────────────────────────────────────
# Fit a StandardScaler on anchor Gram features (recommended).
USE_STD = True

# ──────────────────────────────────────────────────────────────────────────────
# GAMMA VALUES  (RBF kernel bandwidth)
# ──────────────────────────────────────────────────────────────────────────────
# None = automatic median heuristic (safe default).
# You can also specify fixed values, e.g. [None, 2.8e-5, 1e-4].
GAMMAS = [None]

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_CSV = "results.csv"

# ──────────────────────────────────────────────────────────────────────────────
# BACKBONES
# ──────────────────────────────────────────────────────────────────────────────
# Set "enabled": True for the backbones you want to use.
# "layer" or "layers" selects which encoder layer(s) to extract.
# "batch_size" controls GPU memory usage (reduce if OOM).
#
# Recommended: sd_vae layer 7 with gamma 2.8e-5.
#
BACKBONES = {
    "sd_vae": {
        "enabled":    True,
        "layers":     [7],
        "batch_size": 16,
    },
    "flux_vae": {
        "enabled":    False,
        "layers":     [7],
        "batch_size": 8,
    },
    "dc_ae": {
        "enabled":    False,
        "layers":     [10],
        "batch_size": 8,
    },
    "dinov2_vitb14": {
        "enabled":    False,
        "layers":     [9],
        "batch_size": 16,
    },
    "vgg19": {
        "enabled":    False,
        "layers":     [8],
        "batch_size": 32,
    },
    "lpips_vgg": {
        "enabled":    False,
        "layers":     [5],
        "batch_size": 32,
    },
}
