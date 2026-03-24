"""
Feature extraction module using pre-trained CNN backbones.
Supports ResNet50, VGG19, DinoV2, SD-VAE, LPIPS-VGG, and CLIP ViT
with configurable layer extraction and Gram matrix computation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from gram_mmd.config import CONFIG, BACKBONE_CONFIGS, get_backbone_config

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Simple dataset for loading images."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            if self.transform:
                return torch.zeros(3, 224, 224), img_path
            return Image.new("RGB", (224, 224)), img_path


class FeatureHook:
    """Hook to capture intermediate layer activations."""

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.features = output.detach()
        elif isinstance(output, tuple):
            self.features = output[0].detach()
        else:
            self.features = output

    def clear(self):
        self.features = None


class FeatureExtractor:
    """
    Extracts features from images using pre-trained CNN backbones.

    Supports multiple backbone architectures with configurable layer extraction,
    Gram matrix computation, and PCA dimensionality reduction.
    """

    def __init__(
        self,
        backbone: str = None,
        layer: int = None,
        layer_config: Dict = None,
        transform_config: Dict = None,
        device: str = None,
    ):
        self.backbone_name = backbone or CONFIG["backbone"]
        if layer is not None:
            self.layer_config = {"name": f"layer_{layer}", "layers": [layer]}
        elif layer_config is not None:
            self.layer_config = layer_config
        else:
            raise ValueError("Either 'layer' (int) or 'layer_config' (dict) must be provided")
        self.transform_config = transform_config or CONFIG["feature_transforms"][0]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_config = get_backbone_config(self.backbone_name)

        self.model = None
        self.hooks = {}
        self.hook_handles = []
        self.pca = None
        self.pca_fitted = False
        self.scaler = None
        self.scaler_fitted = False

        self._load_model()
        self._register_hooks()
        self.image_transform = self._create_transform()

    def _load_model(self):
        """Load the pre-trained backbone model."""
        logger.info(f"Loading backbone: {self.backbone_name}")

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.model = models.resnet50(weights=weights)

        elif self.backbone_name == "vgg19":
            weights = models.VGG19_Weights.IMAGENET1K_V1
            self.model = models.vgg19(weights=weights)

        elif self.backbone_name == "dinov2_vitb14":
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )

        elif self.backbone_name == "sd_vae":
            from diffusers import AutoencoderKL
            self.model = AutoencoderKL.from_pretrained(
                self.backbone_config["weights"]
            )

        elif self.backbone_name == "flux_vae":
            from diffusers import AutoencoderKL
            self.model = AutoencoderKL.from_pretrained(
                self.backbone_config["weights"],
            )

        elif self.backbone_name == "dc_ae":
            from diffusers import AutoencoderDC
            self.model = AutoencoderDC.from_pretrained(
                self.backbone_config["weights"]
            )

        elif self.backbone_name == "lpips_vgg":
            import lpips
            self.model = lpips.LPIPS(net="vgg", verbose=False)

        elif self.backbone_name == "clip_vit_base":
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained(
                self.backbone_config["weights"]
            )

        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        self.model = self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get a layer from the model by its dotted name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        layer_names = self.backbone_config["layer_names"]
        layers_to_extract = self.layer_config["layers"]

        for layer_idx in layers_to_extract:
            if layer_idx not in layer_names:
                logger.warning(f"Layer {layer_idx} not found in backbone config")
                continue

            layer_name = layer_names[layer_idx]
            try:
                layer = self._get_layer_by_name(layer_name)
                hook = FeatureHook()
                handle = layer.register_forward_hook(hook)
                self.hooks[layer_idx] = hook
                self.hook_handles.append(handle)
                logger.debug(f"Registered hook on layer {layer_idx}: {layer_name}")
            except Exception as e:
                logger.warning(f"Could not register hook on {layer_name}: {e}")

    def _create_transform(self) -> transforms.Compose:
        """Create the image preprocessing transform."""
        input_size = self.backbone_config["input_size"]
        mean = self.backbone_config["normalize_mean"]
        std = self.backbone_config["normalize_std"]

        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _process_features(self, features: torch.Tensor) -> np.ndarray:
        """Process raw layer activations into a fixed-size vector via GAP."""
        features = features.cpu()

        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected feature dimension: {features.dim()}")

        return features.numpy()

    def _aggregate_features(self, features_list: List[np.ndarray]) -> np.ndarray:
        """Aggregate features from multiple layers."""
        if len(features_list) == 1:
            return features_list[0]

        normalized = []
        for f in features_list:
            norms = np.linalg.norm(f, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normalized.append(f / norms)

        return np.concatenate(normalized, axis=1)

    def _compute_gram_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute Gram matrix and extract upper triangular elements."""
        batch_size, dim = features.shape

        max_dim = 2048
        if dim > max_dim:
            logger.debug(f"Subsampling features from {dim} to {max_dim}")
            indices = np.random.choice(dim, max_dim, replace=False)
            features = features[:, indices]
            dim = max_dim

        gram_matrices = np.einsum('bi,bj->bij', features, features)
        triu_idx = np.triu_indices(dim)
        return gram_matrices[:, triu_idx[0], triu_idx[1]]

    def _compute_gram_spatial(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute spatial Gram matrix (Gatys et al.) on feature maps.
        Computes F @ F^T on the full spatial feature map, capturing
        channel co-activation patterns across spatial positions.
        """
        features = features.float()

        if features.dim() == 4:
            B, C, H, W = features.shape
            F = features.reshape(B, C, H * W)
        elif features.dim() == 3:
            B, N, D = features.shape
            F = features.permute(0, 2, 1)
            C = D
        elif features.dim() == 2:
            return self._compute_gram_matrix(features.cpu().numpy())
        else:
            raise ValueError(f"Unexpected feature dimension: {features.dim()}")

        S = F.shape[2]
        G = torch.bmm(F, F.transpose(1, 2)) / S

        triu_mask = torch.triu(torch.ones(C, C, device=features.device, dtype=torch.bool))
        gram_features = G[:, triu_mask]

        return gram_features.cpu().numpy()

    def _compute_gram_spatial_patches(self, features: torch.Tensor, patch_size: int = 4) -> np.ndarray:
        """Compute local Gram matrices on spatial patches."""
        features = features.float()

        if features.dim() != 4:
            logger.warning(f"Patches only supported for 4D features, got {features.dim()}D. Using global Gram.")
            return self._compute_gram_spatial(features)

        B, C, H, W = features.shape

        if H % patch_size != 0 or W % patch_size != 0:
            logger.warning(
                f"Feature map {H}x{W} not divisible by patch_size={patch_size}. Using global Gram."
            )
            return self._compute_gram_spatial(features)

        n_patches_h = H // patch_size
        n_patches_w = W // patch_size
        n_patches = n_patches_h * n_patches_w

        patches = features.reshape(B, C, n_patches_h, patch_size, n_patches_w, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B * n_patches, C, patch_size * patch_size)

        S_patch = patch_size * patch_size
        G = torch.bmm(patches, patches.transpose(1, 2)) / S_patch

        triu_mask = torch.triu(torch.ones(C, C, device=features.device, dtype=torch.bool))
        gram_features = G[:, triu_mask]

        return gram_features.cpu().numpy()

    def _apply_pca(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        pca_dim = self.transform_config.get("pca_dim", 10)

        if fit or not self.pca_fitted:
            logger.info(f"Fitting PCA with {pca_dim} components on {features.shape}")
            n_components = min(pca_dim, features.shape[0], features.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(features)
            self.pca_fitted = True
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        return self.pca.transform(features)

    def transform_features(
        self, features: np.ndarray, fit_pca: bool = False
    ) -> np.ndarray:
        """Apply configured transformations (normalization, PCA) to features."""
        from sklearn.preprocessing import StandardScaler

        use_gram = self.transform_config.get("use_gram", False)
        use_pca = self.transform_config.get("use_pca", False)

        if use_gram:
            if fit_pca and not self.scaler_fitted:
                self.scaler = StandardScaler()
                features = self.scaler.fit_transform(features)
                self.scaler_fitted = True
                logger.info(f"Fitted scaler on Gram features: mean={self.scaler.mean_.mean():.2e}, std={self.scaler.scale_.mean():.2e}")
            elif self.scaler_fitted:
                features = self.scaler.transform(features)

        if use_pca:
            features = self._apply_pca(features, fit=fit_pca)

        return features

    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from a batch of preprocessed images."""
        images = images.to(self.device)

        for hook in self.hooks.values():
            hook.clear()

        with torch.no_grad():
            if self.backbone_name in ("sd_vae", "flux_vae", "dc_ae"):
                _ = self.model.encoder(images)
            elif self.backbone_name == "lpips_vgg":
                scaled = self.model.scaling_layer(images)
                _ = self.model.net(scaled)
            else:
                _ = self.model(images)

        use_gram = self.transform_config.get("use_gram", False)
        gram_patches = self.transform_config.get("gram_patches", False)
        patch_size = self.transform_config.get("patch_size", 4)

        features_list = []
        for layer_idx in sorted(self.hooks.keys()):
            hook = self.hooks[layer_idx]
            if hook.features is not None:
                if use_gram:
                    if gram_patches:
                        processed = self._compute_gram_spatial_patches(hook.features, patch_size)
                    else:
                        processed = self._compute_gram_spatial(hook.features)
                else:
                    processed = self._process_features(hook.features)
                features_list.append(processed)
            else:
                logger.warning(f"No features captured from layer {layer_idx}")

        if not features_list:
            raise RuntimeError("No features were captured from any layer")

        return self._aggregate_features(features_list)

    def extract(
        self,
        images: Union[List[str], List[Image.Image], torch.Tensor],
        fit_transform: bool = False,
        batch_size: int = None,
    ) -> np.ndarray:
        """
        Extract features from a list of images.

        Args:
            images: List of image paths, PIL images, or tensor batch
            fit_transform: Whether to fit PCA on this data (for reference set)
            batch_size: Batch size for processing

        Returns:
            Feature array (N, D)
        """
        batch_size = batch_size or CONFIG["batch_size"]

        if isinstance(images, torch.Tensor):
            all_features = []
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i+batch_size]
                features = self.extract_batch(batch)
                all_features.append(features)
            raw_features = np.concatenate(all_features, axis=0)
            return self.transform_features(raw_features, fit_pca=fit_transform)

        if isinstance(images[0], str):
            dataset = ImageDataset(images, transform=self.image_transform)
        else:
            dataset = [(self.image_transform(img), str(i)) for i, img in enumerate(images)]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=True,
        )

        all_features = []
        total_batches = len(dataloader)

        for batch_idx, (batch_images, _) in enumerate(dataloader):
            if batch_idx % CONFIG["log_progress_every"] == 0:
                logger.info(f"Extracting features: batch {batch_idx+1}/{total_batches}")

            features = self.extract_batch(batch_images)
            all_features.append(features)

        raw_features = np.concatenate(all_features, axis=0)
        logger.info(f"Extracted raw features: shape = {raw_features.shape}")

        transformed = self.transform_features(raw_features, fit_pca=fit_transform)
        logger.info(f"Final feature shape: {transformed.shape}")

        return transformed

    def extract_raw_activations(
        self,
        images: Union[List[str], List[Image.Image], torch.Tensor],
        batch_size: int = None,
    ) -> np.ndarray:
        """Extract raw hook activations (no Gram, no GAP, no PCA)."""
        batch_size = batch_size or min(CONFIG["batch_size"], 16)

        if isinstance(images, torch.Tensor):
            all_activations = []
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size]
                act = self._extract_raw_batch(batch)
                all_activations.append(act)
            return np.concatenate(all_activations, axis=0)

        if isinstance(images[0], str):
            dataset = ImageDataset(images, transform=self.image_transform)
        else:
            dataset = [
                (self.image_transform(img), str(i))
                for i, img in enumerate(images)
            ]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=True,
        )

        all_activations = []
        total_batches = len(dataloader)

        for batch_idx, (batch_images, _) in enumerate(dataloader):
            if batch_idx % CONFIG["log_progress_every"] == 0:
                logger.info(
                    f"Extracting raw activations: batch {batch_idx+1}/{total_batches}"
                )
            act = self._extract_raw_batch(batch_images)
            all_activations.append(act)

        result = np.concatenate(all_activations, axis=0)
        logger.info(f"Extracted raw activations: shape={result.shape}")
        return result

    def _extract_raw_batch(self, images: torch.Tensor) -> np.ndarray:
        """Forward pass on a single batch, return raw hook output."""
        images = images.to(self.device)

        for hook in self.hooks.values():
            hook.clear()

        with torch.no_grad():
            if self.backbone_name in ("sd_vae", "flux_vae", "dc_ae"):
                _ = self.model.encoder(images)
            elif self.backbone_name == "lpips_vgg":
                scaled = self.model.scaling_layer(images)
                _ = self.model.net(scaled)
            else:
                _ = self.model(images)

        for layer_idx in sorted(self.hooks.keys()):
            hook = self.hooks[layer_idx]
            if hook.features is not None:
                return hook.features.cpu().float().numpy()

        raise RuntimeError("No features were captured from any layer")

    def cleanup(self):
        """Remove hooks and free resources."""
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()
