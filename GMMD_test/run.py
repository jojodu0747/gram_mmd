#!/usr/bin/env python3
"""
GMMD_test — run.py
==================
Compute Gram-MMD distances between images (or sets of images) and an
anchor distribution.

Usage:
    python run.py

Configuration is in config.py.
Results are saved to results.csv (configurable via OUTPUT_CSV in config.py).

Detection of evaluation mode (automatic):
  - evaluation_set/ contains sub-folders → distribution vs distribution MMD
  - evaluation_set/ contains image files  → point vs distribution MMD
"""

import csv
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import config as cfg
from gmmd_core import (
    FeatureExtractor,
    list_images,
    _sq_dists,
    median_gamma,
    IMG_EXTS,
)

ANCHOR_DIR = HERE / "dataset" / "anchor_set"
EVAL_DIR   = HERE / "dataset" / "evaluation_set"
OUTPUT_CSV = HERE / cfg.OUTPUT_CSV


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def detect_eval_mode() -> str:
    items = [p for p in EVAL_DIR.iterdir() if not p.name.startswith(".")]
    if any(p.is_dir() for p in items):
        return "distribution"
    return "point"


def get_eval_units(mode: str):
    """Returns list of (name, [image_paths]) tuples."""
    if mode == "point":
        return [
            (p.name, [str(p)])
            for p in sorted(EVAL_DIR.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
    else:
        units = []
        for p in sorted(EVAL_DIR.iterdir()):
            if p.is_dir():
                imgs = list_images(p)
                if imgs:
                    units.append((p.name, imgs))
                else:
                    logger.warning(f"Folder '{p.name}' has no images — skipped.")
        return units


def resolve_gamma(gamma_cfg, D_rr: np.ndarray) -> float:
    return median_gamma(D_rr) if gamma_cfg is None else float(gamma_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Validate inputs ───────────────────────────────────────────────────────
    anchor_paths = list_images(ANCHOR_DIR)
    if not anchor_paths:
        logger.error(f"Anchor set empty or missing: {ANCHOR_DIR}")
        sys.exit(1)
    logger.info(f"Anchor set: {len(anchor_paths)} images")

    if not EVAL_DIR.exists():
        logger.error(f"Evaluation set folder missing: {EVAL_DIR}")
        sys.exit(1)

    mode = detect_eval_mode()
    logger.info(f"Evaluation mode: {mode}")

    eval_units = get_eval_units(mode)
    if not eval_units:
        logger.error("No images or folders found in evaluation_set/")
        sys.exit(1)
    logger.info(f"Evaluation units: {len(eval_units)}")

    enabled = {n: b for n, b in cfg.BACKBONES.items() if b.get("enabled", False)}
    if not enabled:
        logger.error("No backbone enabled in config.py")
        sys.exit(1)

    gammas_cfg = cfg.GAMMAS
    rows = []

    # ── Loop over backbones — model loaded ONCE per backbone ──────────────────
    for backbone_name, bcfg in enabled.items():
        layers     = bcfg.get("layers") or [bcfg["layer"]]
        batch_size = bcfg.get("batch_size", 16)

        logger.info(f"\n{'='*60}")
        logger.info(f"Backbone: {backbone_name}   layers: {layers}")
        logger.info(f"{'='*60}")

        # One model load — all layers hooked simultaneously
        extractor = FeatureExtractor(
            backbone=backbone_name,
            layers=layers,
            device=cfg.DEVICE,
            batch_size=batch_size,
            use_std=cfg.USE_STD,
        )

        # Extract anchor features for ALL layers in one pass
        # fit_scaler=True: fits a StandardScaler per layer on the anchor set
        logger.info(f"Extracting anchor features… (standardisation: {cfg.USE_STD})")
        anchor_all = extractor.extract_all(anchor_paths, fit_scaler=cfg.USE_STD)   # {layer: (N_r, D)}

        # Precompute D_rr per layer (geometric, gamma-independent)
        logger.info("Computing D_rr per layer…")
        D_rr_all = {
            l: _sq_dists(anchor_all[l], anchor_all[l], cfg.DEVICE)
            for l in layers
        }

        # ── POINT mode ────────────────────────────────────────────────────────
        if mode == "point":
            all_paths = [paths[0] for _, paths in eval_units]
            names     = [name    for name, _   in eval_units]

            logger.info(f"Extracting {len(all_paths)} query images…")
            query_all = extractor.extract_all(all_paths)    # {layer: (N_q, D)}

            for layer in layers:
                D_rr = D_rr_all[layer]
                D_rq = _sq_dists(anchor_all[layer], query_all[layer], cfg.DEVICE)

                for gamma_cfg in gammas_cfg:
                    actual_gamma = resolve_gamma(gamma_cfg, D_rr)
                    label = "median" if gamma_cfg is None else gamma_cfg
                    logger.info(f"  layer {layer}  γ={actual_gamma:.3e} ({label})")

                    K_rr   = np.exp(-actual_gamma * D_rr)
                    K_rq   = np.exp(-actual_gamma * D_rq)    # (N_r, N_q)
                    term_rr = float(K_rr.mean())
                    # MMD²(x_j, R) = 1 + mean(K_rr) - 2·mean_i K(r_i, x_j)
                    scores = 1.0 + term_rr - 2.0 * K_rq.mean(axis=0)

                    for i, name in enumerate(names):
                        rows.append({
                            "name":      name,
                            "mode":      "point",
                            "backbone":  backbone_name,
                            "layer":     layer,
                            "gamma":     actual_gamma,
                            "mmd_score": float(scores[i]),
                        })

        # ── DISTRIBUTION mode ─────────────────────────────────────────────────
        else:
            for name, img_paths in eval_units:
                logger.info(f"\n  Folder: '{name}'  ({len(img_paths)} images)")
                query_all = extractor.extract_all(img_paths)  # {layer: (N_q, D)}

                for layer in layers:
                    D_rr = D_rr_all[layer]
                    D_qq = _sq_dists(query_all[layer], query_all[layer], cfg.DEVICE)
                    D_rq = _sq_dists(anchor_all[layer], query_all[layer], cfg.DEVICE)

                    for gamma_cfg in gammas_cfg:
                        actual_gamma = resolve_gamma(gamma_cfg, D_rr)
                        label = "median" if gamma_cfg is None else gamma_cfg
                        logger.info(f"    layer {layer}  γ={actual_gamma:.3e} ({label})")

                        K_rr = np.exp(-actual_gamma * D_rr)
                        K_qq = np.exp(-actual_gamma * D_qq)
                        K_rq = np.exp(-actual_gamma * D_rq)
                        # MMD²(Q, R) = mean(K_rr) + mean(K_qq) - 2·mean(K_rq)
                        score = float(K_rr.mean()) + float(K_qq.mean()) - 2.0 * float(K_rq.mean())

                        rows.append({
                            "name":      name,
                            "mode":      "distribution",
                            "backbone":  backbone_name,
                            "layer":     layer,
                            "gamma":     actual_gamma,
                            "mmd_score": float(score),
                        })

        extractor.cleanup()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    fieldnames = ["name", "mode", "backbone", "layer", "gamma", "mmd_score"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"\nDone! Saved {len(rows)} rows → {OUTPUT_CSV}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  RESULTS")
    print("=" * 74)
    print(f"  {'Name':<28}  {'Backbone':<14}  {'L':>3}  {'γ':>10}  {'MMD²':>10}")
    print("  " + "-" * 70)
    for r in sorted(rows, key=lambda x: x["mmd_score"]):
        print(
            f"  {r['name']:<28}  {r['backbone']:<14}  {r['layer']:>3}  "
            f"{r['gamma']:>10.3e}  {r['mmd_score']:>10.6f}"
        )
    print("=" * 74)


if __name__ == "__main__":
    main()
