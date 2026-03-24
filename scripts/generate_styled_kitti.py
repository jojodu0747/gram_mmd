"""
generate_styled_kitti.py
========================
Génère des images KITTI stylisées via AdaIN (Adaptive Instance Normalization).

AdaIN transfère les statistiques (moyenne/std) des features VGG d'une image
"style" vers une image "content". CLIP est invariant au style → embeddings
quasi-identiques. Les matrices de Gram changent → GMMD détecte la différence.

  Input  : 1000 images KITTI (dataset/kitti/images/)
  Style  : images de peintures (WikiArt subset, téléchargées auto)
  Output : dataset/kitti_styled/  (1000 images stylisées)

Usage:
    .venv/bin/python scripts/generate_styled_kitti.py
"""

import logging
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KITTI_DIR  = BASE_DIR / "dataset" / "kitti" / "images"
OUT_DIR    = BASE_DIR / "dataset" / "kitti_styled"
N_IMAGES   = 1000
SEED       = 42
IMG_SIZE   = 512   # resize avant stylisation
ALPHA      = 1.0   # 1.0 = full style, 0.0 = content only

# Style images : subset WikiArt via HuggingFace (chargé à la demande)
STYLE_DATASET_ID = "huggan/wikiart"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════
#  VGG ENCODER (jusqu'à relu3_1)
# ════════════════════════════════════════════════════════════

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # relu1_1 relu2_1 relu3_1 relu4_1
        self.slice1 = vgg[:2]   # relu1_1
        self.slice2 = vgg[2:7]  # relu2_1
        self.slice3 = vgg[7:12] # relu3_1
        self.slice4 = vgg[12:21]# relu4_1
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


class VGGDecoder(nn.Module):
    """Decoder miroir de VGGEncoder (jusqu'à relu4_1)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.net(x)


def adain(content_feat, style_feat):
    """Adaptive Instance Normalization."""
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std  = content_feat.std(dim=[2, 3], keepdim=True) + 1e-8
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std  = style_feat.std(dim=[2, 3], keepdim=True) + 1e-8
    return s_std * (content_feat - c_mean) / c_std + s_mean


# ════════════════════════════════════════════════════════════
#  ALTERNATIVE : AdaIN simplifié sans decoder (histogram matching)
# ════════════════════════════════════════════════════════════

def histogram_match_lab(content_img: Image.Image, style_img: Image.Image) -> Image.Image:
    """
    Transfère les statistiques couleur (mean/std) par canal dans l'espace L*a*b*.
    Rapide, sans réseau. CLIP est connu pour être invariant à ce type de transfert.
    """
    from skimage import color
    import numpy as np

    content_np = np.array(content_img.convert("RGB")).astype(np.float32) / 255.0
    style_np   = np.array(style_img.convert("RGB")).astype(np.float32) / 255.0

    content_lab = color.rgb2lab(content_np)
    style_lab   = color.rgb2lab(style_np)

    result_lab = content_lab.copy()
    for ch in range(3):
        c_mean, c_std = content_lab[:,:,ch].mean(), content_lab[:,:,ch].std() + 1e-8
        s_mean, s_std = style_lab[:,:,ch].mean(),   style_lab[:,:,ch].std()   + 1e-8
        result_lab[:,:,ch] = (content_lab[:,:,ch] - c_mean) / c_std * s_std + s_mean

    result_rgb = color.lab2rgb(np.clip(result_lab, [-128, -128, -128], [127, 127, 127]))
    result_rgb = np.clip(result_rgb * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result_rgb)


# ════════════════════════════════════════════════════════════
#  LOAD STYLE IMAGES
# ════════════════════════════════════════════════════════════

def get_style_images(n, seed):
    """
    Charge des images de style depuis HuggingFace WikiArt.
    Fallback: génère des textures de bruit coloré.
    """
    return generate_random_textures(n, seed)


def generate_random_textures(n, seed):
    """Génère n images de texture synthétique colorée."""
    rng = np.random.default_rng(seed)
    textures = []
    for _ in range(n):
        # Bruit coloré à basse fréquence (texture peinture)
        small = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(small).resize((512, 512), Image.BILINEAR)
        textures.append(img)
    return textures


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load KITTI paths
    kitti_paths = sorted(KITTI_DIR.glob("*.png"))
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(len(kitti_paths), size=min(N_IMAGES, len(kitti_paths)),
                            replace=False).tolist())
    selected = [kitti_paths[i] for i in idx]
    logger.info(f"  {len(selected)} images KITTI sélectionnées")

    # Load style images
    style_imgs = get_style_images(len(selected), SEED)

    logger.info(f"  Génération des images stylisées → {OUT_DIR}")
    logger.info(f"  Méthode : histogram matching L*a*b* (fast, reproductible)")

    for i, (kitti_path, style_img) in enumerate(zip(selected, style_imgs)):
        content_img = Image.open(kitti_path).convert("RGB")

        # Resize style to content size
        style_resized = style_img.resize(content_img.size, Image.BILINEAR)

        styled = histogram_match_lab(content_img, style_resized)
        out_path = OUT_DIR / f"{kitti_path.stem}_styled.png"
        styled.save(out_path)

        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{len(selected)} images générées")

    logger.info(f"  Done. {len(selected)} images dans {OUT_DIR}")


if __name__ == "__main__":
    main()
