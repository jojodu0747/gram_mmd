# gram-mmd

Image quality metric based on MMD (Maximum Mean Discrepancy) computed on Gram matrices from pre-trained backbone features.

## Installation

```bash
pip install -e .
```

For CLIP-based CMMD support:

```bash
pip install -e ".[clip]"
```

## Usage

```python
from gram_mmd import FeatureExtractor, DistanceMetric, compute_mmd

# Extract Gram features from images
extractor = FeatureExtractor(
    backbone="sd_vae",
    layer_config={"name": "layer_7", "layers": [7]},
)

ref_features = extractor.extract(ref_image_paths, fit_transform=True)
test_features = extractor.extract(test_image_paths)

# Compute MMD distance
metric = DistanceMetric("mmd", features_ref=ref_features)
score = metric.compute(test_features)
```

## Backbones

- **SD-VAE** (`sd_vae`): Stable Diffusion VAE encoder (17 layers)
- **DinoV2** (`dinov2_vitb14`): ViT-B/14 (14 layers)
- **VGG19** (`vgg19`): 18 layers (16 conv + 2 FC)
- **LPIPS-VGG** (`lpips_vgg`): VGG16-based perceptual model (13 layers)
- **CLIP ViT** (`clip_vit_base`): ViT-B/32 (13 layers)
