"""
Image degradation generator module.
Generates sequences of degraded images with increasing severity levels.
"""

import logging
from typing import Dict, List, Union

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torchvision import transforms

from gram_mmd.config import CONFIG

logger = logging.getLogger(__name__)


class DegradationGenerator:
    """
    Generates various types of image degradations with configurable severity levels.

    Supported: Gaussian blur, blur + contrast, Gaussian noise, aliasing, contrast.
    """

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG["degradations"]
        self.rng = np.random.RandomState(CONFIG["random_seed"])

    def apply_gaussian_blur(self, image: Union[Image.Image, np.ndarray], sigma: float) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def apply_contrast(self, image: Union[Image.Image, np.ndarray], factor: float) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        return ImageEnhance.Contrast(image).enhance(factor)

    def apply_blur_and_contrast(
        self, image: Union[Image.Image, np.ndarray],
        blur_sigma: float, contrast_factor: float,
    ) -> Image.Image:
        blurred = self.apply_gaussian_blur(image, blur_sigma)
        return self.apply_contrast(blurred, contrast_factor)

    def apply_gaussian_noise(self, image: Union[Image.Image, np.ndarray], std: float) -> Image.Image:
        if isinstance(image, Image.Image):
            image = np.array(image)
        noise = self.rng.normal(0, std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def apply_aliasing(self, image: Union[Image.Image, np.ndarray], factor: int) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        original_size = image.size
        small_size = (max(1, int(original_size[0] // factor)), max(1, int(original_size[1] // factor)))
        downsampled = image.resize(small_size, Image.Resampling.NEAREST)
        return downsampled.resize(original_size, Image.Resampling.NEAREST)

    def apply_degradation(
        self, image: Union[Image.Image, np.ndarray],
        degradation_type: str, level: int,
    ) -> Image.Image:
        """Apply a specific degradation at a given severity level."""
        if degradation_type not in self.config:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

        deg_config = self.config[degradation_type]

        if degradation_type == "blur":
            levels = deg_config["levels"]
            level = min(level, len(levels) - 1)
            return self.apply_gaussian_blur(image, levels[level])

        elif degradation_type == "blur_contrast":
            blur_levels = deg_config["blur_levels"]
            contrast_factors = deg_config["contrast_factors"]
            n = min(len(blur_levels), len(contrast_factors))
            level = min(level, n - 1)
            return self.apply_blur_and_contrast(image, blur_levels[level], contrast_factors[level])

        elif degradation_type == "noise":
            levels = deg_config["std_levels"]
            level = min(level, len(levels) - 1)
            return self.apply_gaussian_noise(image, levels[level])

        elif degradation_type == "aliasing":
            factors = deg_config["factors"]
            level = min(level, len(factors) - 1)
            return self.apply_aliasing(image, factors[level])

        elif degradation_type == "contrast":
            factors = deg_config["contrast_factors"]
            level = min(level, len(factors) - 1)
            return self.apply_contrast(image, factors[level])

        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

    def get_num_levels(self, degradation_type: str) -> int:
        """Get the number of severity levels for a degradation type."""
        if degradation_type not in self.config:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

        deg_config = self.config[degradation_type]

        if degradation_type == "blur":
            return len(deg_config["levels"])
        elif degradation_type == "blur_contrast":
            return max(len(deg_config["blur_levels"]), len(deg_config["contrast_factors"]))
        elif degradation_type == "noise":
            return len(deg_config["std_levels"])
        elif degradation_type == "aliasing":
            return len(deg_config["factors"])
        elif degradation_type == "contrast":
            return len(deg_config["contrast_factors"])
        else:
            return 0

    def generate_degradation_sequence(
        self, image: Union[Image.Image, np.ndarray], degradation_type: str,
    ) -> List[Image.Image]:
        """Generate a sequence of degraded images with increasing severity."""
        num_levels = self.get_num_levels(degradation_type)
        return [self.apply_degradation(image, degradation_type, level) for level in range(num_levels)]

    def generate_all_degradations(
        self, image: Union[Image.Image, np.ndarray],
    ) -> Dict[str, List[Image.Image]]:
        """Generate all degradation types and levels for an image."""
        return {
            deg_type: self.generate_degradation_sequence(image, deg_type)
            for deg_type in self.config.keys()
        }


class BatchDegradationGenerator:
    """Efficient batch processing of degradations."""

    def __init__(self, config: Dict = None, num_workers: int = 8):
        self.generator = DegradationGenerator(config)
        self.config = self.generator.config
        self.num_workers = num_workers

    def _process_single_image(self, args):
        img, degradation_type, level = args
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        return self.generator.apply_degradation(img, degradation_type, level)

    def process_image_batch(
        self, images: List[Union[str, Image.Image]],
        degradation_type: str, level: int,
    ) -> List[Image.Image]:
        """Apply degradation to a batch of images in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        args_list = [(img, degradation_type, level) for img in images]
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._process_single_image, args_list))
        return results

    def get_degradation_info(self, degradation_type: str, level: int) -> Dict:
        """Get human-readable information about a degradation level."""
        deg_config = self.config[degradation_type]

        if degradation_type == "blur":
            sigma = deg_config["levels"][min(level, len(deg_config["levels"]) - 1)]
            return {"type": "gaussian_blur", "sigma": sigma}
        elif degradation_type == "blur_contrast":
            blur_levels = deg_config["blur_levels"]
            contrast_factors = deg_config["contrast_factors"]
            return {
                "type": "blur_plus_contrast",
                "blur_sigma": blur_levels[min(level, len(blur_levels) - 1)],
                "contrast_factor": contrast_factors[min(level, len(contrast_factors) - 1)],
            }
        elif degradation_type == "noise":
            std = deg_config["std_levels"][min(level, len(deg_config["std_levels"]) - 1)]
            return {"type": "gaussian_noise", "std": std}
        elif degradation_type == "aliasing":
            factor = deg_config["factors"][min(level, len(deg_config["factors"]) - 1)]
            return {"type": "downsample_upsample", "factor": factor}
        return {}


def apply_transform_to_tensor(image: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    """Apply torchvision transform to a PIL image."""
    return transform(image)


def degraded_images_to_tensor(images: List[Image.Image], transform: transforms.Compose) -> torch.Tensor:
    """Convert list of degraded PIL images to a batched tensor."""
    return torch.stack([transform(img) for img in images])
