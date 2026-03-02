"""
Configuration file for gram-mmd pipeline.

PIPELINE OVERVIEW:
==================
1. Load N_R reference images from dataset
2. Extract features from a specific backbone model and layer
3. Compute Gram matrix for each image -> Distribution D_R (N_R vectors)
4. Load N_E evaluation images
5. Apply progressive degradations -> Distributions D_E_k, compute MMD distance with D_R
6. Evaluate monotonicity using Spearman correlation
7. Save results to CSV with all parameters
"""

import os
from pathlib import Path

# Base directory: current working directory (not package location)
BASE_DIR = Path(os.environ.get("GRAM_MMD_BASE_DIR", Path.cwd()))

# ============================================================================
# DATASET PATHS (configurable via env vars or direct override)
# ============================================================================
VISDRONE_DATASET_PATH = str(
    BASE_DIR / "dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/images/"
)
KITTI_DATASET_PATH = str(BASE_DIR / "dataset/kitti/images/")
VIRTUAL_KITTI_DATASET_PATH = str(BASE_DIR / "dataset/virtual_kitti/")
KITTI_TRACKING_DATASET_PATH = str(BASE_DIR / "dataset/kitti_tracking/")

CONFIG = {
    # ========================================================================
    # DATASET CONFIGURATION
    # ========================================================================
    "dataset_path": VISDRONE_DATASET_PATH,
    "results_path": str(BASE_DIR / "results/"),

    # N_R: Number of reference images (real distribution)
    "n_images_inference": 100,

    # N_E: Number of evaluation images (to be degraded)
    "n_images_evaluation": 20,

    "random_seed": 42,
    "batch_size": 64,
    "image_size": 256,  # 256 for SD VAE, 224 for VGG/LPIPS, 518 for DinoV2

    # ========================================================================
    # FEATURE EXTRACTION + GRAM MATRIX
    # ========================================================================
    "backbone": "sd_vae",

    # Experiment configurations: 4 backbones x 6 layers = 24 configs
    "experiment_configs": [
        # SD VAE (17 layers, 0-16)
        {"backbone": "sd_vae", "layer": 0},
        {"backbone": "sd_vae", "layer": 3},
        {"backbone": "sd_vae", "layer": 6},
        {"backbone": "sd_vae", "layer": 9},
        {"backbone": "sd_vae", "layer": 12},
        {"backbone": "sd_vae", "layer": 16},
        # DinoV2 ViT-B/14 (14 layers, 0-13)
        {"backbone": "dinov2_vitb14", "layer": 0},
        {"backbone": "dinov2_vitb14", "layer": 2},
        {"backbone": "dinov2_vitb14", "layer": 5},
        {"backbone": "dinov2_vitb14", "layer": 8},
        {"backbone": "dinov2_vitb14", "layer": 11},
        {"backbone": "dinov2_vitb14", "layer": 13},
        # VGG19 (18 layers, 0-17)
        {"backbone": "vgg19", "layer": 0},
        {"backbone": "vgg19", "layer": 3},
        {"backbone": "vgg19", "layer": 7},
        {"backbone": "vgg19", "layer": 11},
        {"backbone": "vgg19", "layer": 15},
        {"backbone": "vgg19", "layer": 17},
        # LPIPS VGG (13 layers, 0-12)
        {"backbone": "lpips_vgg", "layer": 0},
        {"backbone": "lpips_vgg", "layer": 3},
        {"backbone": "lpips_vgg", "layer": 7},
        {"backbone": "lpips_vgg", "layer": 11},
        {"backbone": "lpips_vgg", "layer": 13},
        {"backbone": "lpips_vgg", "layer": 15},
    ],

    # Legacy layer configs
    "layer_configs": [
        {"name": "single_layer_9", "layers": [9]},
    ],

    # Gram matrix computation
    "feature_transforms": [
        {
            "name": "gram_spatial",
            "use_gram": True,
            "use_pca": False,
            "gram_patches": False,
            "gram_averaging": "spatial",
        },
    ],

    # ========================================================================
    # DISTANCE METRICS
    # ========================================================================
    "distance_metrics": [
        {"name": "mmd", "kernel": "rbf", "gamma": None},  # Auto gamma via median heuristic
    ],

    # ========================================================================
    # DEGRADATION CONFIGURATIONS
    # ========================================================================
    "degradations": {
        "blur": {
            "type": "gaussian_blur",
            "levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                       1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        },
        "blur_contrast": {
            "type": "blur_plus_contrast",
            "blur_levels":      [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
                                  0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0],
            "contrast_factors": [0.98, 0.97, 0.96, 0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83,
                                  0.80, 0.78, 0.75, 0.72, 0.70, 0.67, 0.65, 0.62, 0.60, 0.55],
        },
        "noise": {
            "type": "gaussian_noise",
            "std_levels": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12,
                           14, 16, 18, 20, 22, 24, 26, 28, 30, 34],
        },
        "aliasing": {
            "type": "downsample_upsample",
            "factors": [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7,
                        1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0],
        },
        "contrast": {
            "type": "contrast_only",
            "contrast_factors": [0.98, 0.96, 0.94, 0.92, 0.90, 0.87, 0.84, 0.81, 0.78, 0.75,
                                  0.72, 0.69, 0.66, 0.63, 0.60, 0.57, 0.54, 0.50, 0.42, 0.30],
        },
    },

    # Logging
    "log_level": "INFO",
    "log_progress_every": 10,
}

# Backbone-specific configurations
BACKBONE_CONFIGS = {
    "resnet50": {
        "model_name": "resnet50",
        "weights": "IMAGENET1K_V2",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "total_layers": 10,
        "extractable_layers": {
            1: "conv1", 2: "bn1", 3: "relu", 4: "maxpool",
            5: "layer1", 6: "layer2", 7: "layer3", 8: "layer4",
            9: "avgpool", 10: "fc",
        },
        "layer_names": {
            7: "layer3", 8: "layer4", 9: "avgpool", 10: "fc",
            11: "layer4.0", 12: "layer4.1", 13: "layer4.2",
            14: "layer3.5", 15: "layer3.4",
        },
    },
    "vgg19": {
        "model_name": "vgg19",
        "weights": "IMAGENET1K_V1",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "total_layers": 19,
        "layer_names": {
            0: "features.0",   1: "features.2",   2: "features.5",
            3: "features.7",   4: "features.10",  5: "features.12",
            6: "features.14",  7: "features.16",  8: "features.19",
            9: "features.21",  10: "features.23", 11: "features.25",
            12: "features.28", 13: "features.30", 14: "features.32",
            15: "features.34", 16: "classifier.0", 17: "classifier.3",
        },
    },
    "dinov2_vitb14": {
        "model_name": "dinov2_vitb14",
        "weights": None,
        "input_size": 518,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "total_layers": 14,
        "layer_names": {
            0: "patch_embed",
            1: "blocks.0",  2: "blocks.1",  3: "blocks.2",
            4: "blocks.3",  5: "blocks.4",  6: "blocks.5",
            7: "blocks.6",  8: "blocks.7",  9: "blocks.8",
            10: "blocks.9", 11: "blocks.10", 12: "blocks.11",
            13: "norm",
        },
    },
    "sd_vae": {
        "model_name": "sd_vae",
        "weights": "stabilityai/sd-vae-ft-mse",
        "input_size": 256,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "total_layers": 17,
        "layer_names": {
            0: "encoder.conv_in",
            1: "encoder.down_blocks.0.resnets.0",
            2: "encoder.down_blocks.0.resnets.1",
            3: "encoder.down_blocks.0.downsamplers.0",
            4: "encoder.down_blocks.1.resnets.0",
            5: "encoder.down_blocks.1.resnets.1",
            6: "encoder.down_blocks.1.downsamplers.0",
            7: "encoder.down_blocks.2.resnets.0",
            8: "encoder.down_blocks.2.resnets.1",
            9: "encoder.down_blocks.2.downsamplers.0",
            10: "encoder.down_blocks.3.resnets.0",
            11: "encoder.down_blocks.3.resnets.1",
            12: "encoder.mid_block.resnets.0",
            13: "encoder.mid_block.attentions.0",
            14: "encoder.mid_block.resnets.1",
            15: "encoder.conv_norm_out",
            16: "encoder.conv_out",
        },
    },
    "lpips_vgg": {
        "model_name": "lpips_vgg",
        "weights": "vgg",
        "input_size": 224,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "total_layers": 13,
        "layer_names": {
            0: "net.slice1.0",   1: "net.slice1.2",
            2: "net.slice2.5",   3: "net.slice2.7",
            4: "net.slice3.10",  5: "net.slice3.12",
            6: "net.slice3.14",  7: "net.slice4.17",
            8: "net.slice4.19",  9: "net.slice4.21",
            10: "net.slice5.24", 11: "net.slice5.26",
            12: "net.slice5.28",
        },
    },
    "clip_vit_base": {
        "model_name": "clip_vit_base",
        "weights": "openai/clip-vit-base-patch32",
        "input_size": 224,
        "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
        "normalize_std": [0.26862954, 0.26130258, 0.27577711],
        "total_layers": 13,
        "layer_names": {
            0: "vision_model.embeddings",
            1: "vision_model.encoder.layers.0",
            2: "vision_model.encoder.layers.1",
            3: "vision_model.encoder.layers.2",
            4: "vision_model.encoder.layers.3",
            5: "vision_model.encoder.layers.4",
            6: "vision_model.encoder.layers.5",
            7: "vision_model.encoder.layers.6",
            8: "vision_model.encoder.layers.7",
            9: "vision_model.encoder.layers.8",
            10: "vision_model.encoder.layers.9",
            11: "vision_model.encoder.layers.10",
            12: "vision_model.encoder.layers.11",
        },
    },
}


# ============================================================================
# LAYER SELECTION FOR EXPERIMENTS
# ============================================================================
ENABLED_LAYERS = {
    "sd_vae": {
        0: True, 1: True, 2: False, 3: True, 4: False, 5: True,
        6: False, 7: True, 8: True, 9: False, 10: True, 11: True,
        12: False, 13: True, 14: False, 15: False, 16: False,
    },
    "dinov2_vitb14": {
        0: True, 1: True, 2: True, 3: True, 4: True, 5: True,
        6: True, 7: True, 8: True, 9: True, 10: True, 11: True,
        12: True, 13: True,
    },
    "vgg19": {
        0: True, 1: True, 2: True, 3: True, 4: True, 5: True,
        6: True, 7: True, 8: True, 9: True, 10: True, 11: True,
        12: True, 13: True, 14: True, 15: True, 16: True, 17: True,
    },
    "lpips_vgg": {
        0: True, 1: True, 2: True, 3: True, 4: True, 5: True,
        6: True, 7: True, 8: True, 9: True, 10: True, 11: True,
        12: True,
    },
    "resnet50": {
        7: True, 8: True, 12: True, 13: True, 14: True, 15: True,
    },
    "clip_vit_base": {
        0: True, 1: True, 2: True, 3: True, 4: True, 5: True,
        6: True, 7: True, 8: True, 9: True, 10: True, 11: True,
        12: True,
    },
}


def get_enabled_experiment_configs():
    """Get list of enabled backbone+layer configs for experiments."""
    configs = []
    for backbone, layers in ENABLED_LAYERS.items():
        for layer_idx, enabled in layers.items():
            if enabled:
                configs.append({"backbone": backbone, "layer": layer_idx})
    return configs


def get_config():
    """Return a copy of the configuration."""
    return CONFIG.copy()


def get_backbone_config(backbone_name: str):
    """Return backbone-specific configuration."""
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                        f"Available: {list(BACKBONE_CONFIGS.keys())}")
    return BACKBONE_CONFIGS[backbone_name]


def update_config(**kwargs):
    """Update configuration with new values."""
    for key, value in kwargs.items():
        if key in CONFIG:
            CONFIG[key] = value
        else:
            raise KeyError(f"Unknown config key: {key}")
