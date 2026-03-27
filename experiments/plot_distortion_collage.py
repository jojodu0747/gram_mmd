#!/usr/bin/env python3
"""
plot_distortion_collage.py
==========================
Collage 5 lignes × 10 colonnes :
  - Lignes  : 5 types de dégradation sélectionnés parmi les 20 de sweep_synthetic_kadid.py
  - Colonnes: niveaux 1 à 10
  - Image   : 1 référence COCO (même pour toutes les lignes)

Utilise apply_distortion() importé depuis sweep_synthetic_kadid.py.

Usage :
    .venv/bin/python experiments/plot_distortion_collage.py
    .venv/bin/python experiments/plot_distortion_collage.py --ref dataset/coco_5000/00042.jpg
    .venv/bin/python experiments/plot_distortion_collage.py --types 1,9,13,14,19
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# ── Importer apply_distortion depuis sweep_synthetic_kadid ──────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "sweep_synthetic_kadid",
    Path(__file__).parent / "sweep_synthetic_kadid.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
apply_distortion = _mod.apply_distortion

# ── Noms des 20 types (index 1-based) ───────────────────────────────────────
DIST_NAMES = {
    1:  "Gaussian noise",
    2:  "Multiplicative noise",
    3:  "Brighten",
    4:  "Darken",
    5:  "Jitter",
    6:  "Patches",
    7:  "Pixelate",
    8:  "Quantization",
    9:  "Fog",
    10: "Color cast cool",
    11: "Chrom. aberration",
    12: "Sparse sampling",
    13: "JPEG",
    14: "Gaussian blur",
    15: "Lens blur",
    16: "Motion blur",
    17: "Tilt-stretch",
    18: "Vignette",
    19: "Contrast compress",
    20: "Non-uniform blur",
}

# 10 dégradations par défaut
DEFAULT_TYPES  = [1, 5, 6, 7, 8, 9, 12, 13, 14, 20]
DEFAULT_LEVELS = [1, 5, 10]   # colonnes affichées
THUMB_SIZE     = 200   # pixels (carré)
PAD            = 6     # pixels entre vignettes
LABEL_H        = 26    # hauteur de la barre de titre de colonne
ROW_LABEL_W    = 170   # largeur de la barre de titre de ligne


def make_collage(ref_paths: list[str], dist_types: list[int], out_path: str,
                 levels: list[int] = None):
    """ref_paths : une image par ligne (len == len(dist_types)).
    levels : liste des niveaux à afficher (défaut: DEFAULT_LEVELS)."""
    assert len(ref_paths) == len(dist_types)
    if levels is None:
        levels = DEFAULT_LEVELS
    n_rows = len(dist_types)
    n_cols = len(levels)

    # Dimensions totales
    total_w = ROW_LABEL_W + n_cols * (THUMB_SIZE + PAD) + PAD
    total_h = LABEL_H     + n_rows * (THUMB_SIZE + PAD) + PAD
    canvas  = Image.new("RGB", (total_w, total_h), (240, 240, 240))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font_col = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_row = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font_col = ImageFont.load_default()
        font_row = font_col

    # Bande grise foncée en haut
    draw.rectangle([0, 0, total_w, LABEL_H], fill=(60, 60, 60))
    for col_idx, lvl in enumerate(levels):
        x_center = ROW_LABEL_W + PAD + col_idx * (THUMB_SIZE + PAD) + THUMB_SIZE // 2
        draw.text((x_center, LABEL_H // 2), f"Lvl {lvl}",
                  fill=(255, 255, 255), font=font_col, anchor="mm")

    # ── Lignes : dégradations ────────────────────────────────────────────────
    for row_idx, (dist_type, ref_path) in enumerate(zip(dist_types, ref_paths)):
        ref_img = Image.open(ref_path).convert("RGB")
        y_top   = LABEL_H + PAD + row_idx * (THUMB_SIZE + PAD)

        # Bande grise gauche (nom de la dégradation)
        draw.rectangle([0, y_top, ROW_LABEL_W, y_top + THUMB_SIZE], fill=(80, 80, 80))
        label = f"T{dist_type:02d}  {DIST_NAMES[dist_type]}"
        draw.text((ROW_LABEL_W // 2, y_top + THUMB_SIZE // 2), label,
                  fill=(255, 255, 255), font=font_row, anchor="mm")

        for col_idx, level in enumerate(levels):
            x_left = ROW_LABEL_W + PAD + col_idx * (THUMB_SIZE + PAD)

            degraded = apply_distortion(ref_img, dist_type, level, img_seed=0)
            thumb    = degraded.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            canvas.paste(thumb, (x_left, y_top))

    canvas.save(out_path, dpi=(150, 150))
    print(f"Collage sauvegardé : {out_path}")
    print(f"Taille : {canvas.width}×{canvas.height} px")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",  type=int, default=42,
                   help="Graine aléatoire pour la sélection des 5 images (défaut: 42)")
    p.add_argument("--refs",  type=str, default=None,
                   help="10 noms de fichiers séparés par virgule (dans coco_5000/)")
    p.add_argument("--types", type=str, default=None,
                   help="Types séparés par virgule, ex: '1,5,6,7,8,9,12,13,14,20' (défaut: 10 représentatifs)")
    p.add_argument("--out",   type=str, default=None,
                   help="Fichier de sortie (défaut: results/collage_distortions.png)")
    return p.parse_args()


def main():
    args = parse_args()

    coco_dir  = BASE_DIR / "dataset" / "coco_5000"
    coco_imgs = sorted(coco_dir.glob("*.jpg"))
    if not coco_imgs:
        raise FileNotFoundError("Aucune image COCO trouvée dans dataset/coco_5000/")

    n_rows = len(DEFAULT_TYPES) if not args.types else len(args.types.split(","))

    if args.refs:
        ref_paths = [str(coco_dir / name.strip()) for name in args.refs.split(",")]
        assert len(ref_paths) == n_rows, f"--refs attend exactement {n_rows} noms"
    else:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(coco_imgs), size=n_rows, replace=False)
        ref_paths = [str(coco_imgs[i]) for i in sorted(chosen)]

    for i, p in enumerate(ref_paths):
        print(f"Ligne {i+1} : {Path(p).name}")

    # Types de dégradation
    if args.types:
        dist_types = [int(x) for x in args.types.split(",")]
        assert len(dist_types) == 5, "Exactly 5 types required"
    else:
        dist_types = DEFAULT_TYPES

    print(f"Dégradations : {[(t, DIST_NAMES[t]) for t in dist_types]}")

    # Sortie
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(out_dir / "collage_distortions.png")

    make_collage(ref_paths, dist_types, out_path, levels=DEFAULT_LEVELS)


if __name__ == "__main__":
    main()
