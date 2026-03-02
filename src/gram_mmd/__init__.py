"""gram-mmd: Image quality metric via MMD on Gram matrices."""

__version__ = "0.1.0"

from gram_mmd.features import FeatureExtractor
from gram_mmd.distances import DistanceMetric, compute_mmd
from gram_mmd.evaluation import MonotonicityEvaluator
