"""
Evaluation module for assessing metric monotonicity under degradations.
Measures how well distance metrics detect increasing image degradation.
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from scipy import stats

from gram_mmd.config import CONFIG
from gram_mmd.degradations import DegradationGenerator, BatchDegradationGenerator
from gram_mmd.distances import DistanceMetric, get_distance_metric, get_metric_score
from gram_mmd.features import FeatureExtractor

logger = logging.getLogger(__name__)


def compute_spearman_correlation(
    degradation_levels: np.ndarray,
    metric_scores: np.ndarray,
) -> float:
    """
    Compute Spearman correlation between degradation levels and metric scores.
    A perfect monotonic metric would have correlation = 1.0.
    """
    if len(degradation_levels) < 2:
        return 0.0

    correlation, p_value = stats.spearmanr(degradation_levels, metric_scores)

    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def compute_pairwise_ordering_score(
    degradation_levels: np.ndarray,
    metric_scores: np.ndarray,
) -> float:
    """Compute the percentage of correctly ordered pairs."""
    n = len(degradation_levels)
    if n < 2:
        return 0.0

    correct_pairs = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if degradation_levels[i] < degradation_levels[j]:
                total_pairs += 1
                if metric_scores[i] < metric_scores[j]:
                    correct_pairs += 1
            elif degradation_levels[i] > degradation_levels[j]:
                total_pairs += 1
                if metric_scores[i] > metric_scores[j]:
                    correct_pairs += 1

    if total_pairs == 0:
        return 0.0

    return correct_pairs / total_pairs


def evaluate_monotonicity(
    degradation_levels: np.ndarray,
    metric_scores: np.ndarray,
    method: str = "spearman",
) -> float:
    """Evaluate monotonicity of metric scores with respect to degradation levels."""
    if method == "spearman":
        return compute_spearman_correlation(degradation_levels, metric_scores)
    elif method == "pairwise":
        return compute_pairwise_ordering_score(degradation_levels, metric_scores)
    else:
        raise ValueError(f"Unknown method: {method}")


class MonotonicityEvaluator:
    """Evaluates how well distance metrics detect degradations monotonically."""

    def __init__(
        self,
        extractor: FeatureExtractor,
        reference_features: np.ndarray,
        degradation_config: Dict = None,
    ):
        self.extractor = extractor
        self.reference_features = reference_features
        self.degradation_generator = DegradationGenerator(degradation_config)
        self.degradation_config = degradation_config or CONFIG["degradations"]

    def evaluate_single_image(
        self,
        image: Union[str, Image.Image],
        metric: DistanceMetric,
        degradation_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a single image across all degradation levels."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        degraded_sequence = self.degradation_generator.generate_degradation_sequence(
            image, degradation_type
        )

        num_levels = len(degraded_sequence)
        levels = np.arange(num_levels)
        scores = []

        for degraded_img in degraded_sequence:
            features = self.extractor.extract([degraded_img], fit_transform=False)
            score = metric.compute(features)
            score = get_metric_score(score)
            scores.append(score)

        return levels, np.array(scores)

    def evaluate_image_batch(
        self,
        images: List[Union[str, Image.Image]],
        metric: DistanceMetric,
        degradation_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of images across all degradation levels."""
        batch_generator = BatchDegradationGenerator(self.degradation_config, num_workers=8)
        num_levels = self.degradation_generator.get_num_levels(degradation_type)

        levels = np.arange(num_levels)
        scores = []

        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        for level in range(num_levels):
            degraded_batch = batch_generator.process_image_batch(
                pil_images, degradation_type, level
            )
            features = self.extractor.extract(degraded_batch, fit_transform=False)
            score = metric.compute(features)
            score = get_metric_score(score)
            scores.append(score)

            logger.debug(
                f"Degradation {degradation_type}, level {level}: score = {score:.4f}"
            )

        return levels, np.array(scores)

    def evaluate_all_degradations(
        self,
        images: List[Union[str, Image.Image]],
        metric: DistanceMetric,
        monotonicity_method: str = "spearman",
    ) -> Dict[str, Dict]:
        """Evaluate metric monotonicity across all degradation types."""
        results = {}

        for deg_type in self.degradation_config.keys():
            logger.info(f"Evaluating degradation type: {deg_type}")

            levels, scores = self.evaluate_image_batch(images, metric, deg_type)
            mono_score = evaluate_monotonicity(levels, scores, monotonicity_method)

            results[deg_type] = {
                "levels": levels.tolist(),
                "scores": scores.tolist(),
                "monotonicity": mono_score,
            }

        return results


def evaluate_all_degradations(
    eval_images: List[Union[str, Image.Image]],
    extractor: FeatureExtractor,
    reference_features: np.ndarray,
    metric: DistanceMetric,
    degradation_config: Dict = None,
    monotonicity_method: str = "spearman",
) -> Dict[str, float]:
    """Convenience function to evaluate all degradation types."""
    evaluator = MonotonicityEvaluator(
        extractor, reference_features, degradation_config
    )
    full_results = evaluator.evaluate_all_degradations(
        eval_images, metric, monotonicity_method
    )
    return {deg_type: res["monotonicity"] for deg_type, res in full_results.items()}


def compute_aggregate_monotonicity(
    monotonicity_scores: Dict[str, float],
) -> float:
    """Compute mean monotonicity score across all degradation types."""
    if not monotonicity_scores:
        return 0.0
    return float(np.mean(list(monotonicity_scores.values())))
