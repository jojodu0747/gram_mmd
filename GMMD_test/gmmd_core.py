"""
gmmd_core.py — Gram-MMD core engine (self-contained, no external gram_mmd package).

Provides:
  - BACKBONE_CONFIGS   : architecture definitions
  - FeatureExtractor   : loads a backbone ONCE, hooks all requested layers,
                         extracts Gram features for all layers in one forward pass
  - _sq_dists          : GPU-accelerated pairwise squared distance matrix
  - median_gamma       : median heuristic for gamma
  - IMG_EXTS / list_images : image file utilities
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# BACKBONE CONFIGURATIONS
# ──────────────────────────────────────────────────────────────────────────────

BACKBONE_CONFIGS = {
    "sd_vae": {
        "weights":        "stabilityai/sd-vae-ft-mse",
        "loader":         "diffusers_kl",
        "input_size":     256,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std":  [0.5, 0.5, 0.5],
        "layer_names": {
            0:  "encoder.conv_in",
            1:  "encoder.down_blocks.0.resnets.0",
            2:  "encoder.down_blocks.0.resnets.1",
            3:  "encoder.down_blocks.0.downsamplers.0",
            4:  "encoder.down_blocks.1.resnets.0",
            5:  "encoder.down_blocks.1.resnets.1",
            6:  "encoder.down_blocks.1.downsamplers.0",
            7:  "encoder.down_blocks.2.resnets.0",
            8:  "encoder.down_blocks.2.resnets.1",
            9:  "encoder.down_blocks.2.downsamplers.0",
            10: "encoder.down_blocks.3.resnets.0",
            11: "encoder.down_blocks.3.resnets.1",
            12: "encoder.mid_block.resnets.0",
            13: "encoder.mid_block.attentions.0",
            14: "encoder.mid_block.resnets.1",
            15: "encoder.conv_norm_out",
            16: "encoder.conv_out",
        },
    },
    "flux_vae": {
        "weights":        "diffusers/FLUX.1-vae",
        "loader":         "diffusers_kl",
        "input_size":     512,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std":  [0.5, 0.5, 0.5],
        "layer_names": {
            0:  "encoder.conv_in",
            1:  "encoder.down_blocks.0.resnets.0",
            2:  "encoder.down_blocks.0.resnets.1",
            3:  "encoder.down_blocks.0.downsamplers.0",
            4:  "encoder.down_blocks.1.resnets.0",
            5:  "encoder.down_blocks.1.resnets.1",
            6:  "encoder.down_blocks.1.downsamplers.0",
            7:  "encoder.down_blocks.2.resnets.0",
            8:  "encoder.down_blocks.2.resnets.1",
            9:  "encoder.down_blocks.2.downsamplers.0",
            10: "encoder.down_blocks.3.resnets.0",
            11: "encoder.down_blocks.3.resnets.1",
            12: "encoder.mid_block.resnets.0",
            13: "encoder.mid_block.attentions.0",
            14: "encoder.mid_block.resnets.1",
            15: "encoder.conv_norm_out",
            16: "encoder.conv_out",
        },
    },
    "dc_ae": {
        "weights":        "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers",
        "loader":         "diffusers_dc",
        "input_size":     512,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std":  [0.5, 0.5, 0.5],
        "layer_names": {
            0:  "encoder.conv_in",
            1:  "encoder.down_blocks.0.0",
            2:  "encoder.down_blocks.0.1",
            3:  "encoder.down_blocks.0.2",
            4:  "encoder.down_blocks.1.0",
            5:  "encoder.down_blocks.1.1",
            6:  "encoder.down_blocks.1.2",
            7:  "encoder.down_blocks.2.0",
            8:  "encoder.down_blocks.2.1",
            9:  "encoder.down_blocks.2.2",
            10: "encoder.down_blocks.3.0",
            11: "encoder.down_blocks.3.1",
            12: "encoder.down_blocks.3.2",
            13: "encoder.down_blocks.3.3",
            14: "encoder.down_blocks.4.0",
            15: "encoder.down_blocks.4.1",
            16: "encoder.down_blocks.4.2",
            17: "encoder.down_blocks.4.3",
            18: "encoder.down_blocks.5.0",
            19: "encoder.down_blocks.5.1",
            20: "encoder.down_blocks.5.2",
            21: "encoder.conv_out",
        },
    },
    "dinov2_vitb14": {
        "weights":        None,
        "loader":         "torch_hub_dino",
        "input_size":     518,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std":  [0.229, 0.224, 0.225],
        "layer_names": {
            0:  "patch_embed",
            1:  "blocks.0",   2:  "blocks.1",   3:  "blocks.2",
            4:  "blocks.3",   5:  "blocks.4",   6:  "blocks.5",
            7:  "blocks.6",   8:  "blocks.7",   9:  "blocks.8",
            10: "blocks.9",   11: "blocks.10",  12: "blocks.11",
            13: "norm",
        },
    },
    "vgg19": {
        "weights":        "IMAGENET1K_V1",
        "loader":         "torchvision_vgg19",
        "input_size":     224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std":  [0.229, 0.224, 0.225],
        "layer_names": {
            0:  "features.0",   1:  "features.2",   2:  "features.5",
            3:  "features.7",   4:  "features.10",  5:  "features.12",
            6:  "features.14",  7:  "features.16",  8:  "features.19",
            9:  "features.21",  10: "features.23",  11: "features.25",
            12: "features.28",  13: "features.30",  14: "features.32",
            15: "features.34",  16: "classifier.0", 17: "classifier.3",
        },
    },
    "lpips_vgg": {
        "weights":        "vgg",
        "loader":         "lpips",
        "input_size":     224,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std":  [0.5, 0.5, 0.5],
        "layer_names": {
            0:  "net.slice1.0",   1:  "net.slice1.2",
            2:  "net.slice2.5",   3:  "net.slice2.7",
            4:  "net.slice3.10",  5:  "net.slice3.12",
            6:  "net.slice3.14",  7:  "net.slice4.17",
            8:  "net.slice4.19",  9:  "net.slice4.21",
            10: "net.slice5.24",  11: "net.slice5.26",
            12: "net.slice5.28",
        },
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def list_images(folder) -> List[str]:
    """Return sorted list of image paths in a folder."""
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


# ──────────────────────────────────────────────────────────────────────────────
# DATASET & HOOK
# ──────────────────────────────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except Exception as exc:
            logger.warning(f"Cannot load {path}: {exc}")
            return torch.zeros(3, 224, 224), path


class _FeatureHook:
    def __init__(self):
        self.features: Optional[torch.Tensor] = None

    def __call__(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.features = output.detach()
        elif isinstance(output, tuple):
            self.features = output[0].detach()

    def clear(self):
        self.features = None


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR  (model loaded once, all layers hooked simultaneously)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Loads a backbone model ONCE and hooks all requested layers.
    A single forward pass captures activations for every layer simultaneously,
    so the model is never reloaded between layers.

    Usage:
        extractor = FeatureExtractor("sd_vae", layers=[7, 8], device="cuda")
        feats = extractor.extract_all(image_paths)
        # feats = {7: np.ndarray (N, D7), 8: np.ndarray (N, D8)}
        extractor.cleanup()
    """

    def __init__(
        self,
        backbone: str,
        layers: List[int],
        device: str = "cuda",
        batch_size: int = 16,
        use_std: bool = True,
    ):
        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Available: {list(BACKBONE_CONFIGS.keys())}"
            )
        self._cfg = BACKBONE_CONFIGS[backbone]
        self.backbone_name = backbone
        self.layers_list = layers
        self.device = device
        self.batch_size = batch_size
        self.use_std = use_std

        for l in layers:
            if l not in self._cfg["layer_names"]:
                raise ValueError(
                    f"Layer {l} not defined for backbone '{backbone}'. "
                    f"Available: {sorted(self._cfg['layer_names'].keys())}"
                )

        self.hooks: Dict[int, _FeatureHook] = {l: _FeatureHook() for l in layers}
        self._handles = []
        self.model = None
        self.scalers: Dict[int, object] = {}   # one StandardScaler per layer
        self.scalers_fitted: bool = False

        self._load_model()
        self._register_hooks()
        self.transform = self._make_transform()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        loader = self._cfg["loader"]
        weights = self._cfg["weights"]
        logger.info(f"Loading backbone '{self.backbone_name}' (loader: {loader})")

        if loader == "diffusers_kl":
            from diffusers import AutoencoderKL
            self.model = AutoencoderKL.from_pretrained(weights)

        elif loader == "diffusers_dc":
            from diffusers import AutoencoderDC
            self.model = AutoencoderDC.from_pretrained(weights)

        elif loader == "torch_hub_dino":
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )

        elif loader == "torchvision_vgg19":
            from torchvision import models
            self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        elif loader == "lpips":
            import lpips
            self.model = lpips.LPIPS(net="vgg", verbose=False)

        else:
            raise ValueError(f"Unknown loader: {loader}")

        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ── Hook registration (all layers at once) ────────────────────────────────

    def _get_layer(self, name: str) -> nn.Module:
        parts = name.split(".")
        m = self.model
        for part in parts:
            if part.isdigit():
                m = m[int(part)] if not hasattr(m, part) else getattr(m, part)
            else:
                m = getattr(m, part)
        return m

    def _register_hooks(self):
        for l in self.layers_list:
            layer_name = self._cfg["layer_names"][l]
            module = self._get_layer(layer_name)
            handle = module.register_forward_hook(self.hooks[l])
            self._handles.append(handle)
            logger.info(f"  Hook on layer {l}: {layer_name}")

    # ── Transform ─────────────────────────────────────────────────────────────

    def _make_transform(self) -> transforms.Compose:
        sz = self._cfg["input_size"]
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self._cfg["normalize_mean"],
                std=self._cfg["normalize_std"],
            ),
        ])

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, images: torch.Tensor):
        for hook in self.hooks.values():
            hook.clear()
        loader = self._cfg["loader"]
        with torch.no_grad():
            if loader in ("diffusers_kl", "diffusers_dc"):
                _ = self.model.encoder(images)
            elif loader == "lpips":
                scaled = self.model.scaling_layer(images)
                _ = self.model.net(scaled)
            else:
                _ = self.model(images)

    # ── Gram matrix ───────────────────────────────────────────────────────────

    def _gram_spatial(self, feat: torch.Tensor) -> np.ndarray:
        """Upper-triangular Gram matrix per image → (B, C*(C+1)/2)."""
        feat = feat.float()
        if feat.dim() == 4:
            B, C, H, W = feat.shape
            F = feat.reshape(B, C, H * W)
        elif feat.dim() == 3:
            B, N, D = feat.shape
            F = feat.permute(0, 2, 1)
            C = D
        elif feat.dim() == 2:
            B, C = feat.shape
            F = feat.unsqueeze(2)
        else:
            raise ValueError(f"Unexpected feature dimension: {feat.dim()}")

        S = F.shape[2]
        G = torch.bmm(F, F.transpose(1, 2)) / S          # (B, C, C)
        triu = torch.triu(torch.ones(C, C, device=feat.device, dtype=torch.bool))
        return G[:, triu].cpu().numpy()                   # (B, C*(C+1)/2)

    # ── Public interface ──────────────────────────────────────────────────────

    def extract_all(
        self,
        image_paths: List[str],
        fit_scaler: bool = False,
    ) -> Dict[int, np.ndarray]:
        """
        Extract Gram-matrix features for all hooked layers in a single pass.

        Args:
            image_paths: list of image file paths.
            fit_scaler:  if True AND use_std=True, fit a StandardScaler on this
                         set and apply it (use for the anchor set).
                         If False and scalers_fitted, the previously fitted
                         scaler is applied automatically (use for query sets).

        Returns:
            dict {layer_idx: np.ndarray of shape (N, D)}
        """
        dataset = _ImageDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(self.device == "cuda"),
            persistent_workers=True,
        )

        # Accumulate per layer
        accum: Dict[int, list] = {l: [] for l in self.layers_list}

        n_batches = len(loader)
        for i, (batch, _) in enumerate(loader):
            batch = batch.to(self.device)
            self._forward(batch)

            for l in self.layers_list:
                feat = self.hooks[l].features
                if feat is None:
                    raise RuntimeError(
                        f"No features captured at layer {l} of '{self.backbone_name}'. "
                        "Check that the layer index is valid."
                    )
                accum[l].append(self._gram_spatial(feat))

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                logger.info(f"  batch {i+1}/{n_batches}")

        result: Dict[int, np.ndarray] = {}
        for l, feats in accum.items():
            arr = np.concatenate(feats, axis=0).astype(np.float32)

            if self.use_std:
                if fit_scaler:
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler()
                    arr = sc.fit_transform(arr).astype(np.float32)
                    self.scalers[l] = sc
                    logger.info(
                        f"  layer {l}: scaler fitted  "
                        f"mean={sc.mean_.mean():.2e}  std={sc.scale_.mean():.2e}"
                    )
                elif self.scalers_fitted and l in self.scalers:
                    arr = self.scalers[l].transform(arr).astype(np.float32)

            result[l] = arr
            logger.info(f"  layer {l}: {arr.shape}")

        if fit_scaler and self.use_std:
            self.scalers_fitted = True

        return result

    def cleanup(self):
        """Remove hooks and free GPU memory."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# DISTANCE & GAMMA UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _sq_dists(A: np.ndarray, B: np.ndarray, device: str) -> np.ndarray:
    """
    Pairwise squared Euclidean distances ||A_i - B_j||².
    Returns shape (N_A, N_B). Uses GPU for speed when available.
    """
    a = torch.tensor(A, dtype=torch.float32, device=device)
    b = torch.tensor(B, dtype=torch.float32, device=device)
    a_sq = (a * a).sum(1, keepdim=True)
    b_sq = (b * b).sum(1, keepdim=True).t()
    D = torch.clamp(a_sq + b_sq - 2.0 * (a @ b.t()), min=0.0)
    return D.cpu().numpy()


def median_gamma(D_rr: np.ndarray) -> float:
    """Median heuristic: γ = 1 / (2 · median(non-zero squared distances))."""
    nonzero = D_rr[D_rr > 0]
    if len(nonzero) == 0:
        return 1.0
    return float(1.0 / (2.0 * float(np.median(nonzero))))
