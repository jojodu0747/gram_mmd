"""
generate_sd_styled.py
=====================
Génère des images stylisées via Stable Diffusion img2img.

SD img2img ajoute du bruit à l'image puis la débruite avec un prompt.
  - strength ∈ [0, 1] : 0 = image originale, 1 = génération pure
  - strength ~0.45 : préserve la structure/sémantique, change les textures

CLIP est largement invariant à strength ≤ 0.5 → CMMD devrait inverser.
Les features Gram (SD-VAE) capturent les textures → GMMD ne devrait pas inverser.

Usage:
    .venv/bin/python scripts/generate_sd_styled.py --dataset kitti --n 1000
    .venv/bin/python scripts/generate_sd_styled.py --dataset visdrone --n 1000
    .venv/bin/python scripts/generate_sd_styled.py --dataset kitti --strength 0.6 --n 1000
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIRS = {
    "kitti":    BASE_DIR / "dataset" / "kitti" / "images",
    "visdrone": BASE_DIR / "dataset" / "VisDrone2019-DET-train" / "VisDrone2019-DET-train" / "images",
    "tracking": BASE_DIR / "dataset" / "data_tracking_image_2" / "training" / "image_02",
}

# Prompts génériques qui ne changent pas la sémantique mais guident la texture
PROMPTS = {
    "kitti":    "a photo of a street scene, detailed, realistic",
    "visdrone": "aerial photograph, drone view, detailed, realistic",
    "tracking": "a photo of a street scene, detailed, realistic",
}

SEED = 42


def get_paths(dataset, n, seed):
    d = DATASET_DIRS[dataset]
    if dataset == "tracking":
        paths = sorted(d.rglob("*.png"))
    else:
        paths = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(idx)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kitti", choices=list(DATASET_DIRS.keys()))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--strength", type=float, default=0.45,
                        help="SD strength: 0=original, 1=full generation. ~0.45 recommended.")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    out_dir = BASE_DIR / "dataset" / f"{args.dataset}_sd_styled_s{int(args.strength*100)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset  : {args.dataset}")
    logger.info(f"Strength : {args.strength}")
    logger.info(f"Output   : {out_dir}")
    logger.info(f"Model    : {args.model}")

    # Load SD pipeline
    from diffusers import StableDiffusionImg2ImgPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device   : {device}")

    logger.info("Loading SD pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    prompt = PROMPTS.get(args.dataset, "a photo, detailed, realistic")
    paths = get_paths(args.dataset, args.n, SEED)
    logger.info(f"Images   : {len(paths)}")

    generator = torch.Generator(device=device).manual_seed(SEED)

    for i in range(0, len(paths), args.batch_size):
        batch_paths = paths[i:i+args.batch_size]
        # Vérifier quelles images sont déjà générées
        to_process = []
        for p in batch_paths:
            out_path = out_dir / f"{p.stem}_sd.png"
            if not out_path.exists():
                to_process.append(p)

        if not to_process:
            continue

        imgs = [Image.open(str(p)).convert("RGB").resize((512, 512)) for p in to_process]

        with torch.autocast(device):
            results = pipe(
                prompt=[prompt] * len(imgs),
                image=imgs,
                strength=args.strength,
                guidance_scale=7.5,
                num_inference_steps=20,
                generator=generator,
            ).images

        for p, result in zip(to_process, results):
            out_path = out_dir / f"{p.stem}_sd.png"
            result.save(out_path)

        done = min(i + args.batch_size, len(paths))
        logger.info(f"  {done}/{len(paths)} images générées")

    logger.info(f"Done. {len(paths)} images → {out_dir}")


if __name__ == "__main__":
    main()
