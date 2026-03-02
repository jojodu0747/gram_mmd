"""
Persistent disk cache for raw activation tensors.
Stores hook outputs (before Gram/GAP/PCA) so experiments can vary
post-processing without re-running the backbone forward pass.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FeatureCache:
    """Disk-based cache for raw activation tensors."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)

    # -- path helpers --

    def image_set_id(self, image_paths: List[str]) -> str:
        """Deterministic hash of sorted image paths (12 hex chars)."""
        content = "\n".join(sorted(image_paths))
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _ref_dir(self, backbone: str, layer_name: str, img_set_id: str) -> Path:
        return self.cache_dir / backbone / layer_name / f"ref_{img_set_id}"

    def _eval_dir(self, backbone: str, layer_name: str, img_set_id: str) -> Path:
        return self.cache_dir / backbone / layer_name / f"eval_{img_set_id}"

    # -- save / load reference activations --

    def save_activations(
        self,
        dir_path: Path,
        activations: np.ndarray,
        image_paths: List[str],
        backbone: str,
        layer_name: str,
        chunk_size: int = 100,
    ):
        """Save raw activation tensors as chunked .npy files + metadata.json."""
        dir_path.mkdir(parents=True, exist_ok=True)
        n = activations.shape[0]

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = start // chunk_size
            chunk_path = dir_path / f"chunk_{chunk_idx:03d}.npy"
            np.save(chunk_path, activations[start:end].astype(np.float32))

        metadata = {
            "backbone": backbone,
            "layer_name": layer_name,
            "n_images": n,
            "shape_per_image": list(activations.shape[1:]),
            "dtype": "float32",
            "chunk_size": chunk_size,
            "n_chunks": (n + chunk_size - 1) // chunk_size,
            "image_paths": sorted(image_paths),
            "timestamp": datetime.now().isoformat(),
        }
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        total_mb = activations.nbytes / 1e6
        logger.info(f"Saved {n} activations to {dir_path} ({total_mb:.1f} MB)")

    def load_activations(self, dir_path: Path) -> np.ndarray:
        """Load and concatenate all chunk .npy files."""
        meta_path = dir_path / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        chunks = []
        for i in range(metadata["n_chunks"]):
            chunk_path = dir_path / f"chunk_{i:03d}.npy"
            chunks.append(np.load(chunk_path))

        activations = np.concatenate(chunks, axis=0).astype(np.float32)
        logger.info(
            f"Loaded {activations.shape[0]} activations from {dir_path}, "
            f"shape={activations.shape}"
        )
        return activations

    # -- save / load degraded activations --

    def save_degraded_activations(
        self, eval_dir: Path, deg_type: str, level: int, activations: np.ndarray,
    ):
        """Save degraded activations: {eval_dir}/{deg_type}/level_{nn}.npy"""
        deg_dir = eval_dir / deg_type
        deg_dir.mkdir(parents=True, exist_ok=True)
        np.save(deg_dir / f"level_{level:02d}.npy", activations.astype(np.float32))

    def load_degraded_activations(
        self, eval_dir: Path, deg_type: str, level: int
    ) -> np.ndarray:
        """Load degraded activations for a specific type/level."""
        path = eval_dir / deg_type / f"level_{level:02d}.npy"
        return np.load(path).astype(np.float32)

    # -- cache existence checks --

    def has_reference(self, backbone: str, layer_name: str, img_set_id: str) -> bool:
        ref_dir = self._ref_dir(backbone, layer_name, img_set_id)
        return (ref_dir / "metadata.json").exists()

    def has_all_degraded(
        self, backbone: str, layer_name: str, img_set_id: str,
        degradation_config: Dict,
    ) -> bool:
        """Check if ALL degraded activations are cached."""
        eval_dir = self._eval_dir(backbone, layer_name, img_set_id)
        if not eval_dir.exists():
            return False

        from gram_mmd.degradations import DegradationGenerator

        deg_gen = DegradationGenerator(degradation_config)

        for deg_type in degradation_config.keys():
            num_levels = deg_gen.get_num_levels(deg_type)
            for level in range(num_levels):
                path = eval_dir / deg_type / f"level_{level:02d}.npy"
                if not path.exists():
                    return False
        return True

    # -- post-processing of cached activations --

    def process_activations(
        self,
        activations: np.ndarray,
        transform_config: Dict,
        pca_model=None,
        pca_dim: int = 10,
        fit_pca: bool = False,
    ):
        """
        Apply post-processing to raw cached activations.

        Returns:
            features: (N, D) processed feature array
            pca_model: PCA model (fitted or passed through)
        """
        from sklearn.decomposition import PCA

        use_gram = transform_config.get("use_gram", False)
        use_pca = transform_config.get("use_pca", False)
        pca_dim = transform_config.get("pca_dim", pca_dim)

        act_tensor = torch.from_numpy(activations)

        if use_gram:
            features = self._compute_gram_spatial_batch(act_tensor)
        else:
            features = self._process_features_batch(act_tensor)

        if use_pca:
            if fit_pca:
                n_components = min(pca_dim, features.shape[0], features.shape[1])
                pca_model = PCA(n_components=n_components)
                pca_model.fit(features)
                logger.info(
                    f"PCA fitted: {features.shape[1]}d -> {n_components}d, "
                    f"explained variance: {pca_model.explained_variance_ratio_.sum():.4f}"
                )
            features = pca_model.transform(features)

        return features, pca_model

    def _process_features_batch(self, activations: torch.Tensor) -> np.ndarray:
        """GAP on raw activations."""
        if activations.dim() == 4:
            return activations.mean(dim=[2, 3]).numpy()
        elif activations.dim() == 3:
            return activations.mean(dim=1).numpy()
        elif activations.dim() == 2:
            return activations.numpy()
        else:
            raise ValueError(f"Unexpected dimension: {activations.dim()}")

    def _compute_gram_spatial_batch(self, activations: torch.Tensor) -> np.ndarray:
        """Spatial Gram matrix on raw activations."""
        activations = activations.float()

        if activations.dim() == 4:
            B, C, H, W = activations.shape
            F = activations.reshape(B, C, H * W)
        elif activations.dim() == 3:
            B, N, D = activations.shape
            F = activations.permute(0, 2, 1)
            C = D
        elif activations.dim() == 2:
            features = activations.numpy()
            batch_size, dim = features.shape
            max_dim = 2048
            if dim > max_dim:
                indices = np.random.choice(dim, max_dim, replace=False)
                features = features[:, indices]
                dim = max_dim
            gram_features = []
            for i in range(batch_size):
                f = features[i:i + 1].T
                gram = f @ f.T
                upper_tri = gram[np.triu_indices(dim)]
                gram_features.append(upper_tri)
            return np.array(gram_features)
        else:
            raise ValueError(f"Unexpected dimension: {activations.dim()}")

        S = F.shape[2]
        triu_idx = np.triu_indices(C)
        gram_features = []
        for i in range(B):
            G = (F[i] @ F[i].T) / S
            gram_features.append(G.numpy()[triu_idx])

        return np.array(gram_features)

    # -- utility --

    def get_cache_summary(self, backbone: str = None) -> str:
        """Return a human-readable summary of cached data."""
        if not self.cache_dir.exists():
            return "No cache directory found."

        lines = [f"Cache directory: {self.cache_dir}"]
        total_size = 0

        for bb_dir in sorted(self.cache_dir.iterdir()):
            if not bb_dir.is_dir():
                continue
            if backbone and bb_dir.name != backbone:
                continue
            for layer_dir in sorted(bb_dir.iterdir()):
                if not layer_dir.is_dir():
                    continue
                for set_dir in sorted(layer_dir.iterdir()):
                    if not set_dir.is_dir():
                        continue
                    size = sum(f.stat().st_size for f in set_dir.rglob("*") if f.is_file())
                    total_size += size
                    meta_path = set_dir / "metadata.json"
                    n_images = "?"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            n_images = json.load(f).get("n_images", "?")
                    lines.append(
                        f"  {bb_dir.name}/{layer_dir.name}/{set_dir.name}: "
                        f"{n_images} images, {size / 1e6:.1f} MB"
                    )

        lines.append(f"Total: {total_size / 1e6:.1f} MB")
        return "\n".join(lines)
