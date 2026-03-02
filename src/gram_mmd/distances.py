"""
Distance metrics module.
Implements distribution distance metrics:
- MMD (Maximum Mean Discrepancy) with RBF kernel
- FID (Frechet Inception Distance)
- Sinkhorn Distance (Optimal Transport)
- Energy Distance
- Adversarial Distance (GAN-style critic)
"""

import logging
from typing import Union, Optional

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


def compute_fid(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    max_dim: int = 10000,
) -> float:
    """
    Compute Frechet Inception Distance (FID) between reference and test features.

    FID = ||mu_ref - mu_test||^2 + Tr(Sigma_ref + Sigma_test - 2*(Sigma_ref @ Sigma_test)^0.5)
    """
    n_ref, dim_orig = features_ref.shape
    n_test, _ = features_test.shape

    if dim_orig > max_dim:
        logger.warning(
            f"Dimension {dim_orig} too high for FID. Applying PCA to {max_dim}."
        )
        from sklearn.decomposition import PCA
        pca_dim = min(max_dim, n_ref - 1, n_test - 1, dim_orig)
        pca = PCA(n_components=pca_dim)
        features_ref = pca.fit_transform(features_ref)
        features_test = pca.transform(features_test)

    mu_ref = np.mean(features_ref, axis=0, dtype=np.float32)
    mu_test = np.mean(features_test, axis=0, dtype=np.float32)

    features_ref_centered = features_ref - mu_ref
    features_test_centered = features_test - mu_test

    sigma_ref = (features_ref_centered.T @ features_ref_centered) / (n_ref - 1)
    sigma_test = (features_test_centered.T @ features_test_centered) / (n_test - 1)

    if sigma_ref.ndim == 0:
        sigma_ref = np.array([[sigma_ref]], dtype=np.float32)
    if sigma_test.ndim == 0:
        sigma_test = np.array([[sigma_test]], dtype=np.float32)

    eps = 1e-6
    dim = sigma_ref.shape[0]
    sigma_ref = sigma_ref + np.eye(dim, dtype=np.float32) * eps
    sigma_test = sigma_test + np.eye(dim, dtype=np.float32) * eps

    diff = mu_ref - mu_test
    mean_diff_sq = np.sum(diff ** 2)

    try:
        covmean, _ = linalg.sqrtm(sigma_ref @ sigma_test, disp=False)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                logger.warning("Imaginary component in covmean, taking real part")
            covmean = covmean.real
    except Exception as e:
        logger.warning(f"Matrix sqrt failed: {e}, using eigendecomposition")
        eigvals, eigvecs = np.linalg.eigh(sigma_ref @ sigma_test)
        eigvals = np.maximum(eigvals, 0)
        covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    fid = mean_diff_sq + np.trace(sigma_ref + sigma_test - 2 * covmean)
    return float(fid)


def compute_mmd(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    K_ref_ref_cached: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between reference and test features.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

    For N_test == 1: biased estimator with k(y,y) term.
    For N_test > 1: unbiased estimator (diagonal removed).
    """
    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]

    if kernel == "rbf":
        if gamma is None:
            n_samples = min(1000, n_ref + n_test)
            combined = np.vstack([features_ref, features_test])
            indices = np.random.choice(combined.shape[0], n_samples, replace=False)
            sample = combined[indices]

            from scipy.spatial.distance import pdist
            pairwise_sq_dists = pdist(sample, metric="sqeuclidean")
            median_dist_sq = np.median(pairwise_sq_dists)
            gamma = 1.0 / (2 * median_dist_sq + 1e-8)
            logger.debug(f"MMD: using gamma={gamma:.6f} from median heuristic")

        def rbf_kernel(X, Y):
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
            sq_dists = X_sq + Y_sq.T - 2 * X @ Y.T
            return np.exp(-gamma * sq_dists)

        if K_ref_ref_cached is not None:
            K_ref_ref = K_ref_ref_cached
        else:
            K_ref_ref = rbf_kernel(features_ref, features_ref)

        K_test_test = rbf_kernel(features_test, features_test)
        K_ref_test = rbf_kernel(features_ref, features_test)

    elif kernel == "linear":
        if K_ref_ref_cached is not None:
            K_ref_ref = K_ref_ref_cached
        else:
            K_ref_ref = features_ref @ features_ref.T

        K_test_test = features_test @ features_test.T
        K_ref_test = features_ref @ features_test.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if n_test == 1:
        # MMD^2(y, X) = mean(K_XX) + k(y,y) - 2*mean(K_Xy)
        term1 = np.mean(K_ref_ref)
        term2 = float(K_test_test[0, 0])
        term3 = 2.0 * np.mean(K_ref_test)
        mmd_sq = term1 + term2 - term3
    else:
        # Unbiased MMD^2 (diagonal removed)
        term1 = (np.sum(K_ref_ref) - np.trace(K_ref_ref)) / (n_ref * (n_ref - 1))
        term2 = (np.sum(K_test_test) - np.trace(K_test_test)) / (n_test * (n_test - 1))
        term3 = 2 * np.sum(K_ref_test) / (n_ref * n_test)
        mmd_sq = term1 + term2 - term3

    return float(mmd_sq)


def compute_sinkhorn(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Compute Sinkhorn distance (regularized optimal transport)."""
    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]

    a = np.ones(n_ref, dtype=np.float64) / n_ref
    b = np.ones(n_test, dtype=np.float64) / n_test

    X_sq = np.sum(features_ref ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(features_test ** 2, axis=1, keepdims=True)
    C = X_sq + Y_sq.T - 2 * features_ref @ features_test.T
    C = np.maximum(C, 0)

    C_max = np.max(C)
    if C_max > 0:
        C = C / C_max
    else:
        return 0.0

    log_K = -C / reg
    u = np.zeros(n_ref, dtype=np.float64)
    v = np.zeros(n_test, dtype=np.float64)

    for iteration in range(max_iter):
        u_prev = u.copy()
        u = np.log(a + 1e-16) - np.log(np.sum(np.exp(log_K + v[None, :]), axis=1) + 1e-16)
        v = np.log(b + 1e-16) - np.log(np.sum(np.exp(log_K.T + u[None, :]), axis=1) + 1e-16)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    log_P = u[:, None] + log_K + v[None, :]
    P = np.exp(log_P)
    sinkhorn_dist = np.sum(P * C) * C_max

    return float(sinkhorn_dist)


def compute_energy_distance(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    sample_size: int = 1000,
) -> float:
    """Compute Energy Distance: E(X,Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]"""
    from scipy.spatial.distance import cdist

    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]

    X = features_ref
    if n_ref > sample_size:
        X = features_ref[np.random.choice(n_ref, sample_size, replace=False)]

    Y = features_test
    if n_test > sample_size:
        Y = features_test[np.random.choice(n_test, sample_size, replace=False)]

    dists_XY = cdist(X, Y, metric='euclidean')
    term1 = 2 * np.mean(dists_XY)

    dists_XX = cdist(X, X, metric='euclidean')
    np.fill_diagonal(dists_XX, 0)
    term2 = np.sum(dists_XX) / (len(X) * (len(X) - 1))

    dists_YY = cdist(Y, Y, metric='euclidean')
    np.fill_diagonal(dists_YY, 0)
    term3 = np.sum(dists_YY) / (len(Y) * (len(Y) - 1))

    return float(term1 - term2 - term3)


def compute_adversarial_distance(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    n_critics: int = 5,
    max_iter: int = 100,
    learning_rate: float = 0.01,
    hidden_dim: int = 128,
) -> float:
    """Compute Adversarial Distance using a learned WGAN-style critic."""
    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]
    d = features_ref.shape[1]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features_ref)
    Y = scaler.transform(features_test)

    critic_scores = []

    for _ in range(n_critics):
        np.random.seed(None)
        W1 = np.random.randn(d, hidden_dim) * 0.01
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, 1) * 0.01
        b2 = np.zeros(1)

        for iteration in range(max_iter):
            batch_size = min(32, n_ref, n_test)
            idx_ref = np.random.choice(n_ref, batch_size, replace=False)
            idx_test = np.random.choice(n_test, batch_size, replace=False)
            X_batch = X[idx_ref]
            Y_batch = Y[idx_test]

            def critic_forward(x, W1, b1, W2, b2):
                h = np.maximum(0, x @ W1 + b1)
                return h @ W2 + b2

            scores_ref = critic_forward(X_batch, W1, b1, W2, b2)
            scores_test = critic_forward(Y_batch, W1, b1, W2, b2)

            grad_scores_ref = -np.ones_like(scores_ref) / batch_size
            grad_scores_test = np.ones_like(scores_test) / batch_size

            grad_W2 = np.zeros_like(W2)
            grad_b2 = np.zeros_like(b2)

            h_ref = np.maximum(0, X_batch @ W1 + b1)
            grad_W2 += h_ref.T @ grad_scores_ref
            grad_b2 += np.sum(grad_scores_ref, axis=0)
            grad_h_ref = grad_scores_ref @ W2.T
            grad_h_ref[h_ref <= 0] = 0
            grad_W1_ref = X_batch.T @ grad_h_ref
            grad_b1_ref = np.sum(grad_h_ref, axis=0)

            h_test = np.maximum(0, Y_batch @ W1 + b1)
            grad_W2 += h_test.T @ grad_scores_test
            grad_b2 += np.sum(grad_scores_test, axis=0)
            grad_h_test = grad_scores_test @ W2.T
            grad_h_test[h_test <= 0] = 0
            grad_W1_test = Y_batch.T @ grad_h_test
            grad_b1_test = np.sum(grad_h_test, axis=0)

            grad_W1 = grad_W1_ref + grad_W1_test
            grad_b1 = grad_b1_ref + grad_b1_test

            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2

            clip_value = 0.01
            W1 = np.clip(W1, -clip_value, clip_value)
            W2 = np.clip(W2, -clip_value, clip_value)

        scores_ref_full = critic_forward(X, W1, b1, W2, b2)
        scores_test_full = critic_forward(Y, W1, b1, W2, b2)
        wasserstein_approx = np.mean(scores_ref_full) - np.mean(scores_test_full)
        critic_scores.append(wasserstein_approx)

    return float(abs(np.mean(critic_scores)))


class DistanceMetric:
    """
    Wrapper class for distance metrics.
    Pre-computes and caches K_ref_ref for MMD speedup.
    """

    def __init__(
        self,
        metric_name: str,
        features_ref: Optional[np.ndarray] = None,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        sinkhorn_reg: float = 0.1,
        sinkhorn_max_iter: int = 100,
        energy_sample_size: int = 1000,
        adversarial_n_critics: int = 5,
        adversarial_max_iter: int = 100,
        **kwargs,
    ):
        self.metric_name = metric_name.lower()
        self.features_ref = features_ref
        self.kernel = kernel
        self.gamma = gamma
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.energy_sample_size = energy_sample_size
        self.adversarial_n_critics = adversarial_n_critics
        self.adversarial_max_iter = adversarial_max_iter

        valid_metrics = ["fid", "mmd", "sinkhorn", "energy", "adversarial"]
        if self.metric_name not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {', '.join(valid_metrics)}")

        if features_ref is None:
            raise ValueError(f"{metric_name} requires features_ref")

        self.K_ref_ref_cached = None
        if self.metric_name == "mmd":
            self._precompute_mmd_reference()

    def _precompute_mmd_reference(self):
        """Pre-compute reference kernel matrix K_ref_ref for MMD."""
        n_ref = self.features_ref.shape[0]

        if self.kernel == "rbf" and self.gamma is None:
            n_samples = min(1000, n_ref)
            indices = np.random.choice(n_ref, n_samples, replace=False)
            sample = self.features_ref[indices]

            from scipy.spatial.distance import pdist
            pairwise_sq_dists = pdist(sample, metric="sqeuclidean")
            median_dist_sq = np.median(pairwise_sq_dists)
            self.gamma = 1.0 / (2 * median_dist_sq + 1e-8)
            logger.info(f"MMD: median_dist_sq={median_dist_sq:.2e}, gamma={self.gamma:.10e}")

        if self.kernel == "rbf":
            X = self.features_ref
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            sq_dists = X_sq + X_sq.T - 2 * X @ X.T
            self.K_ref_ref_cached = np.exp(-self.gamma * sq_dists)
        elif self.kernel == "linear":
            self.K_ref_ref_cached = self.features_ref @ self.features_ref.T

        logger.info(f"Reference kernel matrix cached: shape {self.K_ref_ref_cached.shape}")

    def compute(self, features_test: np.ndarray) -> float:
        """Compute the distance metric for test features."""
        if self.metric_name == "fid":
            return compute_fid(self.features_ref, features_test)
        elif self.metric_name == "mmd":
            return compute_mmd(
                self.features_ref, features_test,
                kernel=self.kernel, gamma=self.gamma,
                K_ref_ref_cached=self.K_ref_ref_cached
            )
        elif self.metric_name == "sinkhorn":
            return compute_sinkhorn(
                self.features_ref, features_test,
                reg=self.sinkhorn_reg, max_iter=self.sinkhorn_max_iter
            )
        elif self.metric_name == "energy":
            return compute_energy_distance(
                self.features_ref, features_test,
                sample_size=self.energy_sample_size
            )
        elif self.metric_name == "adversarial":
            return compute_adversarial_distance(
                self.features_ref, features_test,
                n_critics=self.adversarial_n_critics,
                max_iter=self.adversarial_max_iter
            )
        else:
            raise ValueError(f"Unknown metric: {self.metric_name}")

    def __call__(self, features_test: np.ndarray) -> float:
        return self.compute(features_test)


def get_distance_metric(name: str, **kwargs) -> DistanceMetric:
    """Factory function to get a distance metric by name."""
    return DistanceMetric(name, **kwargs)


def get_metric_score(metric_result: Union[float, tuple]) -> float:
    """Extract scalar score from metric result."""
    if isinstance(metric_result, tuple):
        return metric_result[0]
    return metric_result
