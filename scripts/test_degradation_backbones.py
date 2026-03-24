"""
Test de degradation comparatif : sd_vae vs flux_vae vs dc_ae

Pour chaque backbone, on :
1. Charge N_REF images de reference, extrait les features Gram → distribution D_ref
2. Charge N_EVAL images d'evaluation
3. Applique des degradations progressives (blur, noise, aliasing, contrast)
4. Calcule MMD(D_ref, D_eval_degraded) a chaque niveau
5. Mesure la monotonicity (Spearman) → un bon backbone donne rho proche de 1.0

Usage:
    python scripts/test_degradation_backbones.py
    python scripts/test_degradation_backbones.py --n-ref 50 --n-eval 10 --device cuda
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure gram_mmd is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gram_mmd.config import CONFIG, BACKBONE_CONFIGS, get_backbone_config
from gram_mmd.features import FeatureExtractor
from gram_mmd.distances import DistanceMetric
from gram_mmd.degradations import DegradationGenerator, BatchDegradationGenerator
from gram_mmd.evaluation import evaluate_monotonicity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Backbone configs to test: (backbone_name, layer_index)
# On choisit une couche mid-depth representative par backbone
# ============================================================================
BACKBONE_TESTS = [
    # SD VAE : couche 9 = encoder.down_blocks.2.downsamplers.0 (milieu encoder)
    ("sd_vae", 9),
    # FLUX VAE : meme architecture, meme couche pour comparer
    ("flux_vae", 9),
    # DC-AE : couche 10 = encoder.down_blocks.3.0 (stage 3, 512ch)
    ("dc_ae", 10),
]

# Degradations a tester (subset pour aller vite)
DEGRADATION_TYPES = ["blur", "noise", "aliasing", "contrast"]


def load_image_paths(dataset_path: str, n_images: int, seed: int = 42) -> list:
    """Load random image paths from dataset directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    all_paths = sorted([
        str(p) for p in Path(dataset_path).rglob("*")
        if p.suffix.lower() in extensions
    ])

    if not all_paths:
        raise FileNotFoundError(f"No images found in {dataset_path}")

    rng = np.random.RandomState(seed)
    n_images = min(n_images, len(all_paths))
    indices = rng.choice(len(all_paths), n_images, replace=False)
    selected = [all_paths[i] for i in sorted(indices)]

    logger.info(f"Selected {len(selected)} images from {dataset_path} ({len(all_paths)} total)")
    return selected


def run_degradation_test(
    backbone_name: str,
    layer: int,
    ref_paths: list,
    eval_paths: list,
    degradation_types: list,
    device: str = "cpu",
    batch_size: int = 16,
) -> dict:
    """
    Run degradation test for a single backbone+layer config.

    Returns dict with:
        {deg_type: {"levels": [...], "scores": [...], "spearman": float}}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {backbone_name} / layer {layer}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    # 1. Create extractor
    extractor = FeatureExtractor(
        backbone=backbone_name,
        layer=layer,
        device=device,
    )
    t_load = time.time() - t0
    logger.info(f"Model loaded in {t_load:.1f}s")

    # 2. Extract reference features
    t1 = time.time()
    ref_features = extractor.extract(
        ref_paths, fit_transform=True, batch_size=batch_size
    )
    t_ref = time.time() - t1
    logger.info(f"Reference features: shape={ref_features.shape}, time={t_ref:.1f}s")

    # 3. Create MMD metric (pre-computes K_ref_ref)
    metric = DistanceMetric(
        metric_name="mmd",
        features_ref=ref_features,
        kernel="rbf",
    )

    # 4. Run degradation tests
    deg_gen = DegradationGenerator()
    batch_deg_gen = BatchDegradationGenerator(num_workers=4)
    results = {}

    from PIL import Image
    eval_images = [Image.open(p).convert("RGB") for p in eval_paths]

    for deg_type in degradation_types:
        logger.info(f"\n  Degradation: {deg_type}")
        num_levels = deg_gen.get_num_levels(deg_type)
        levels = list(range(num_levels))
        scores = []

        for level in levels:
            # Degrade all eval images at this level
            degraded = batch_deg_gen.process_image_batch(
                eval_images, deg_type, level
            )

            # Extract features from degraded batch
            features = extractor.extract(degraded, fit_transform=False, batch_size=batch_size)

            # Compute MMD
            score = metric.compute(features)
            scores.append(float(score))

            if level % 5 == 0 or level == num_levels - 1:
                logger.info(f"    level {level:2d}/{num_levels-1}: MMD = {score:.6f}")

        scores_arr = np.array(scores)
        spearman = evaluate_monotonicity(
            np.array(levels, dtype=float), scores_arr, method="spearman"
        )

        results[deg_type] = {
            "levels": levels,
            "scores": scores,
            "spearman": spearman,
        }
        logger.info(f"  → Spearman = {spearman:.4f}")

    # Cleanup
    extractor.cleanup()

    total_time = time.time() - t0
    logger.info(f"\nTotal time for {backbone_name}/layer{layer}: {total_time:.1f}s")

    return results


def save_results_csv(all_results: dict, output_path: str):
    """Save results to CSV."""
    rows = []
    for (backbone, layer), deg_results in all_results.items():
        for deg_type, data in deg_results.items():
            for level, score in zip(data["levels"], data["scores"]):
                rows.append({
                    "backbone": backbone,
                    "layer": layer,
                    "degradation": deg_type,
                    "level": level,
                    "mmd_score": score,
                    "spearman": data["spearman"],
                })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Results saved to {output_path}")


def print_summary(all_results: dict):
    """Print comparison summary table."""
    print(f"\n{'='*70}")
    print(f"MONOTONICITY COMPARISON (Spearman correlation)")
    print(f"{'='*70}")

    # Header
    deg_types = DEGRADATION_TYPES
    header = f"{'Backbone':<20} {'Layer':<6}"
    for dt in deg_types:
        header += f" {dt:>10}"
    header += f" {'MEAN':>10}"
    print(header)
    print("-" * 70)

    # Rows
    for (backbone, layer), deg_results in all_results.items():
        row = f"{backbone:<20} {layer:<6}"
        spearman_values = []
        for dt in deg_types:
            if dt in deg_results:
                sp = deg_results[dt]["spearman"]
                spearman_values.append(sp)
                row += f" {sp:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        mean_sp = np.mean(spearman_values) if spearman_values else 0.0
        row += f" {mean_sp:>10.4f}"
        print(row)

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Test degradation monotonicity across backbones")
    parser.add_argument("--n-ref", type=int, default=50, help="Number of reference images")
    parser.add_argument("--n-eval", type=int, default=10, help="Number of evaluation images")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect if omitted)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for extraction")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--output", type=str, default="results/test_degradation_backbones.csv")
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Override backbones to test (e.g. sd_vae:9 flux_vae:9 dc_ae:10)")
    args = parser.parse_args()

    # Device
    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Dataset
    dataset_path = args.dataset or CONFIG["dataset_path"]
    logger.info(f"Dataset: {dataset_path}")

    # Load images (separate ref and eval sets)
    all_paths = load_image_paths(dataset_path, args.n_ref + args.n_eval, seed=42)
    ref_paths = all_paths[:args.n_ref]
    eval_paths = all_paths[args.n_ref:args.n_ref + args.n_eval]
    logger.info(f"Reference: {len(ref_paths)} images, Evaluation: {len(eval_paths)} images")

    # Parse backbone configs
    if args.backbones:
        backbone_tests = []
        for spec in args.backbones:
            parts = spec.split(":")
            backbone_tests.append((parts[0], int(parts[1])))
    else:
        backbone_tests = BACKBONE_TESTS

    # Run tests
    all_results = {}
    for backbone_name, layer in backbone_tests:
        try:
            results = run_degradation_test(
                backbone_name=backbone_name,
                layer=layer,
                ref_paths=ref_paths,
                eval_paths=eval_paths,
                degradation_types=DEGRADATION_TYPES,
                device=device,
                batch_size=args.batch_size,
            )
            all_results[(backbone_name, layer)] = results
        except Exception as e:
            logger.error(f"FAILED: {backbone_name}/layer{layer}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print_summary(all_results)
        save_results_csv(all_results, args.output)


if __name__ == "__main__":
    main()
