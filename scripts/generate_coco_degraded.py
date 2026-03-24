"""
Génère un dataset COCO-dégradé avec les 25 types de dégradations de KADID-10k,
chacun à 5 niveaux de sévérité croissants.

Structure de sortie :
    dataset/coco_degraded/
    ├── images/         # images dégradées  COCO_{ref:05d}_{type:02d}_{level}.png
    ├── references/     # images de référence originales
    └── metadata.csv    # ref_img, dist_img, dist_type, dist_name, level, params

Usage :
    .venv/bin/python scripts/generate_coco_degraded.py
    .venv/bin/python scripts/generate_coco_degraded.py --n-ref 100 --workers 4

Paramètres des distorsions : basés sur le papier KADID-10k (Lin et al., 2019).
Quelques valeurs sont approximées là où les suppléments du papier ne sont pas
disponibles — elles sont marquées [approx] dans les commentaires.
"""

import argparse
import csv
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
COCO_DIR  = BASE_DIR / "dataset" / "coco_5000"
OUT_DIR   = BASE_DIR / "dataset" / "coco_degraded"

ANCHOR_SEED = 42
N_ANCHOR    = 1000      # à exclure du set de référence
N_REF       = 200       # images de référence pour les dégradations
REF_SEED    = 123       # seed séparé pour sélectionner les refs
TARGET_SIZE = 512       # taille de recadrage (carré)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Définition des 25 distorsions KADID
# ===========================================================================
# Format : (nom, [(param_label, valeur), ...]) — 5 entrées = 5 niveaux
# Les valeurs de niveau 1 → 5 correspondent à une sévérité croissante.

DISTORTIONS = {
    # ----- Groupe 1 : Flou -----
    1:  ("gaussian_blur",
         [("sigma", v) for v in [0.495, 0.99, 1.485, 1.98, 2.475]]),
         # KADID exact : uniform steps de 0.495

    2:  ("lens_blur",
         [("radius", v) for v in [3, 5, 7, 9, 11]]),
         # Filtre disque (PSF circulaire) — rayon en pixels [approx]

    3:  ("motion_blur",
         [("kernel_size", v) for v in [3, 7, 13, 19, 25]]),
         # Flou linéaire directionnel 45° [approx]

    # ----- Groupe 2 : Couleur -----
    4:  ("color_diffusion",
         [("kernel_size", v) for v in [3, 7, 13, 21, 33]]),
         # Lissage spatial dans l'espace LAB (diffusion couleur) [approx]

    5:  ("color_shift",
         [("shift_r", v) for v in [10, 20, 35, 50, 75]]),
         # Décalage R+, G−, B+ par niveaux [approx]

    6:  ("color_quantization",
         [("n_colors", v) for v in [64, 32, 16, 8, 4]]),
         # Quantification palette N couleurs

    7:  ("color_saturation_1",
         [("factor", v) for v in [0.8, 0.6, 0.4, 0.2, 0.0]]),
         # Désaturation (0 = niveaux de gris)

    8:  ("color_saturation_2",
         [("factor", v) for v in [1.5, 2.0, 3.0, 5.0, 8.0]]),
         # Sursaturation

    # ----- Groupe 3 : Compression -----
    9:  ("jpeg2000",
         [("rate", v) for v in [5, 15, 40, 100, 200]]),
         # JPEG2000 — rate = facteur de compression (plus haut = pire)

    10: ("jpeg",
         [("quality", v) for v in [24, 18, 12, 8, 4]]),
         # JPEG — qualité PIL (valeur KADID exacte)

    # ----- Groupe 4 : Bruit -----
    11: ("white_noise",
         [("sigma", v) for v in [2, 7, 15, 25, 40]]),
         # Bruit gaussien uniforme [approx — KADID utilise σ en [0,1] normalisé]

    12: ("color_noise",
         [("sigma", v) for v in [2, 7, 15, 25, 40]]),
         # Bruit gaussien par canal indépendant [approx]

    13: ("impulse_noise",
         [("fraction", v) for v in [0.02, 0.05, 0.10, 0.20, 0.35]]),
         # Bruit sel-et-poivre — fraction de pixels corrompus [approx]

    14: ("multiplicative_noise",
         [("sigma", v) for v in [0.05, 0.10, 0.20, 0.35, 0.50]]),
         # Bruit multiplicatif : pixel × (1 + N(0,σ)) [approx]

    15: ("denoise",
         [("sigma", v) for v in [2, 5, 10, 20, 40]]),
         # Sur-débruitage (flou gaussien fort) [approx]

    # ----- Groupe 5 : Luminosité -----
    16: ("brighten",
         [("factor", v) for v in [1.1, 1.2, 1.3, 1.4, 1.5]]),
         # Eclaircissement — facteur PIL Brightness

    17: ("darken",
         [("factor", v) for v in [0.9, 0.8, 0.7, 0.6, 0.5]]),
         # Assombrissement — facteur PIL Brightness

    18: ("mean_shift",
         [("shift", v) for v in [10, 25, 50, 75, 100]]),
         # Décalage global de luminosité (ajout constante) [approx]

    # ----- Groupe 6 : Divers -----
    19: ("jitter",
         [("magnitude", v) for v in [1, 2, 4, 6, 8]]),
         # Déplacement aléatoire des pixels (translation locale) [approx]

    20: ("noneccentricity_patch",
         [("n_patches", v) for v in [5, 10, 20, 40, 80]]),
         # Copie de patches internes — crée des répétitions anormales [approx]

    21: ("pixelate",
         [("block_size", v) for v in [2, 4, 8, 16, 32]]),
         # Pixelisation par blocs carrés

    22: ("quantization",
         [("bits", v) for v in [6, 5, 4, 3, 2]]),
         # Réduction de profondeur de bit par canal

    23: ("color_block",
         [("block_size", v) for v in [16, 32, 64, 96, 128]]),
         # Patches de couleur uniforme aléatoire superposés [approx]

    24: ("high_sharpen",
         [("factor", v) for v in [2.0, 4.0, 6.0, 9.0, 15.0]]),
         # Sur-accentuation (unsharp masking fort)

    25: ("contrast_change",
         [("factor", v) for v in [0.8, 0.6, 0.45, 0.30, 0.15]]),
         # Réduction de contraste — facteur PIL Contrast
}


# ===========================================================================
# Implémentations des distorsions
# ===========================================================================

def _to_pil(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8))
    return img


def _to_arr(img):
    if isinstance(img, Image.Image):
        return np.array(img, dtype=np.uint8)
    return img.astype(np.uint8)


def apply_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_lens_blur(img: Image.Image, radius: int) -> Image.Image:
    """Filtre disque (PSF circulaire) via convolution numpy."""
    from scipy.ndimage import convolve
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
    disk /= disk.sum()
    arr = np.array(img, dtype=np.float32)
    out = np.stack([convolve(arr[:, :, c], disk) for c in range(3)], axis=2)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_motion_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    """Flou de mouvement directionnel à 45°."""
    from scipy.ndimage import convolve
    kernel = np.eye(kernel_size, dtype=np.float32) / kernel_size
    arr = np.array(img, dtype=np.float32)
    out = np.stack([convolve(arr[:, :, c], kernel) for c in range(3)], axis=2)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_color_diffusion(img: Image.Image, kernel_size: int) -> Image.Image:
    """Lissage spatial dans l'espace LAB."""
    from scipy.ndimage import uniform_filter
    lab = img.convert("LAB")
    arr = np.array(lab, dtype=np.float32)
    out = np.stack([uniform_filter(arr[:, :, c], size=kernel_size) for c in range(3)], axis=2)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGB")


def apply_color_shift(img: Image.Image, shift_r: int) -> Image.Image:
    """Décalage asymétrique des canaux : R+shift, G-shift//2, B+shift//3."""
    arr = np.array(img, dtype=np.int16)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shift_r,          0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] - shift_r // 2,     0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + shift_r // 3,     0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_color_quantization(img: Image.Image, n_colors: int) -> Image.Image:
    quantized = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    return quantized.convert("RGB")


def apply_color_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def apply_jpeg2000(img: Image.Image, rate: float) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[rate])
    buf.seek(0)
    return Image.open(buf).copy().convert("RGB")


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy().convert("RGB")


def apply_white_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_color_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.stack([rng.normal(0, sigma, arr.shape[:2]) for _ in range(3)], axis=2).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_impulse_noise(img: Image.Image, fraction: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img, dtype=np.uint8).copy()
    n_pixels = arr.shape[0] * arr.shape[1]
    n_corrupt = int(n_pixels * fraction)
    coords = rng.choice(n_pixels, size=n_corrupt, replace=False)
    rows, cols = np.unravel_index(coords, (arr.shape[0], arr.shape[1]))
    salt = rng.random(n_corrupt) > 0.5
    arr[rows[salt],  cols[salt]]  = 255
    arr[rows[~salt], cols[~salt]] = 0
    return Image.fromarray(arr)


def apply_multiplicative_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr * (1.0 + noise), 0, 255).astype(np.uint8))


def apply_denoise(img: Image.Image, sigma: float) -> Image.Image:
    """Sur-débruitage simulé par flou gaussien fort."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_brighten(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_darken(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_mean_shift(img: Image.Image, shift: int) -> Image.Image:
    arr = np.array(img, dtype=np.int16)
    return Image.fromarray(np.clip(arr + shift, 0, 255).astype(np.uint8))


def apply_jitter(img: Image.Image, magnitude: int, rng: np.random.Generator) -> Image.Image:
    """Déplacement aléatoire des pixels (map_coordinates)."""
    from scipy.ndimage import map_coordinates
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape[:2]
    dy = rng.integers(-magnitude, magnitude + 1, size=(H, W))
    dx = rng.integers(-magnitude, magnitude + 1, size=(H, W))
    rows = np.clip(np.arange(H)[:, None] + dy, 0, H - 1)
    cols = np.clip(np.arange(W)[None, :] + dx, 0, W - 1)
    out = np.stack([arr[:, :, c][rows, cols] for c in range(3)], axis=2)
    return Image.fromarray(out.astype(np.uint8))


def apply_noneccentricity_patch(img: Image.Image, n_patches: int, rng: np.random.Generator) -> Image.Image:
    """Copie de patches 32×32 depuis le centre vers des positions aléatoires."""
    arr = np.array(img, dtype=np.uint8).copy()
    H, W = arr.shape[:2]
    patch_size = 32
    cx, cy = W // 2 - patch_size // 2, H // 2 - patch_size // 2
    source_patch = arr[cy:cy + patch_size, cx:cx + patch_size].copy()
    for _ in range(n_patches):
        ty = rng.integers(0, H - patch_size)
        tx = rng.integers(0, W - patch_size)
        arr[ty:ty + patch_size, tx:tx + patch_size] = source_patch
    return Image.fromarray(arr)


def apply_pixelate(img: Image.Image, block_size: int) -> Image.Image:
    W, H = img.size
    small = img.resize((max(1, W // block_size), max(1, H // block_size)), Image.Resampling.NEAREST)
    return small.resize((W, H), Image.Resampling.NEAREST)


def apply_quantization(img: Image.Image, bits: int) -> Image.Image:
    """Réduction de profondeur de bit par canal."""
    levels = 2 ** bits
    arr = np.array(img, dtype=np.float32)
    quantized = np.floor(arr / 256.0 * levels) / levels * 255.0
    return Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))


def apply_color_block(img: Image.Image, block_size: int, rng: np.random.Generator) -> Image.Image:
    """Patches de couleur uniforme aléatoire superposés."""
    arr = np.array(img, dtype=np.uint8).copy()
    H, W = arr.shape[:2]
    n_blocks = max(1, (H * W) // (block_size * block_size) // 10)
    for _ in range(n_blocks):
        ty = rng.integers(0, max(1, H - block_size))
        tx = rng.integers(0, max(1, W - block_size))
        color = rng.integers(0, 256, size=3).astype(np.uint8)
        arr[ty:ty + block_size, tx:tx + block_size] = color
    return Image.fromarray(arr)


def apply_high_sharpen(img: Image.Image, factor: float) -> Image.Image:
    """Unsharp masking fort."""
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
    arr = np.array(img, dtype=np.float32)
    arr_b = np.array(blurred, dtype=np.float32)
    sharpened = arr + factor * (arr - arr_b)
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))


def apply_contrast_change(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


# ===========================================================================
# Dispatcher : appelle la bonne fonction selon dist_type et level
# ===========================================================================

def apply_distortion(img: Image.Image, dist_type: int, level_idx: int,
                     rng: np.random.Generator) -> Image.Image:
    """
    dist_type : 1-25 (numérotation KADID)
    level_idx : 0-4  (niveau 0 = le plus léger)
    """
    _, params = DISTORTIONS[dist_type]
    param_label, param_value = params[level_idx]

    if dist_type == 1:
        return apply_gaussian_blur(img, param_value)
    elif dist_type == 2:
        return apply_lens_blur(img, param_value)
    elif dist_type == 3:
        return apply_motion_blur(img, param_value)
    elif dist_type == 4:
        return apply_color_diffusion(img, param_value)
    elif dist_type == 5:
        return apply_color_shift(img, param_value)
    elif dist_type == 6:
        return apply_color_quantization(img, param_value)
    elif dist_type == 7:
        return apply_color_saturation(img, param_value)
    elif dist_type == 8:
        return apply_color_saturation(img, param_value)
    elif dist_type == 9:
        return apply_jpeg2000(img, param_value)
    elif dist_type == 10:
        return apply_jpeg(img, param_value)
    elif dist_type == 11:
        return apply_white_noise(img, param_value, rng)
    elif dist_type == 12:
        return apply_color_noise(img, param_value, rng)
    elif dist_type == 13:
        return apply_impulse_noise(img, param_value, rng)
    elif dist_type == 14:
        return apply_multiplicative_noise(img, param_value, rng)
    elif dist_type == 15:
        return apply_denoise(img, param_value)
    elif dist_type == 16:
        return apply_brighten(img, param_value)
    elif dist_type == 17:
        return apply_darken(img, param_value)
    elif dist_type == 18:
        return apply_mean_shift(img, param_value)
    elif dist_type == 19:
        return apply_jitter(img, param_value, rng)
    elif dist_type == 20:
        return apply_noneccentricity_patch(img, param_value, rng)
    elif dist_type == 21:
        return apply_pixelate(img, param_value)
    elif dist_type == 22:
        return apply_quantization(img, param_value)
    elif dist_type == 23:
        return apply_color_block(img, param_value, rng)
    elif dist_type == 24:
        return apply_high_sharpen(img, param_value)
    elif dist_type == 25:
        return apply_contrast_change(img, param_value)
    else:
        raise ValueError(f"dist_type inconnu : {dist_type}")


# ===========================================================================
# Sélection des images de référence
# ===========================================================================

def select_reference_images(n_ref: int) -> list:
    """
    Retourne n_ref chemins COCO non inclus dans l'ancrage (seed=ANCHOR_SEED).
    """
    all_imgs = sorted(
        list(COCO_DIR.glob("*.jpg")) + list(COCO_DIR.glob("*.png"))
    )
    all_imgs = [str(p) for p in all_imgs]
    n_total = len(all_imgs)

    # Reproduire exactement la sélection de l'ancrage
    anchor_rng = np.random.default_rng(ANCHOR_SEED)
    anchor_idx = set(
        anchor_rng.choice(n_total, size=N_ANCHOR, replace=False).tolist()
    )

    non_anchor = [p for i, p in enumerate(all_imgs) if i not in anchor_idx]
    logger.info(f"COCO total : {n_total}  |  ancrage exclus : {N_ANCHOR}  |  "
                f"non-ancrage disponible : {len(non_anchor)}")

    if n_ref > len(non_anchor):
        raise ValueError(f"Demande {n_ref} refs mais seulement {len(non_anchor)} disponibles")

    ref_rng = np.random.default_rng(REF_SEED)
    idx = sorted(ref_rng.choice(len(non_anchor), size=n_ref, replace=False).tolist())
    selected = [non_anchor[i] for i in idx]
    logger.info(f"Images de référence sélectionnées : {len(selected)}")
    return selected


# ===========================================================================
# Chargement + resize carré
# ===========================================================================

def load_and_resize(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # Crop carré centré, puis resize
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.Resampling.LANCZOS)


# ===========================================================================
# Traitement d'une image de référence
# ===========================================================================

def process_reference(ref_idx: int, ref_path: str,
                       img_dir: Path, ref_dir: Path,
                       size: int) -> list:
    """
    Génère les 25×5 = 125 images dégradées pour une image de référence.
    Retourne une liste de lignes CSV.
    """
    rows = []
    ref_name = f"COCO_{ref_idx + 1:05d}.png"
    ref_out  = ref_dir / ref_name

    try:
        img = load_and_resize(ref_path, size)
    except Exception as e:
        logger.warning(f"[ref {ref_idx + 1}] Impossible de charger {ref_path}: {e}")
        return rows

    # Sauvegarder la référence une seule fois
    if not ref_out.exists():
        img.save(ref_out, format="PNG")

    # Un RNG par référence (reproductible)
    rng = np.random.default_rng(REF_SEED + ref_idx * 1000)

    for dist_type in range(1, 26):
        dist_name, params = DISTORTIONS[dist_type]
        for level_idx, (param_label, param_value) in enumerate(params):
            level = level_idx + 1  # 1-5 comme KADID
            dist_fname = f"COCO_{ref_idx + 1:05d}_{dist_type:02d}_{level}.png"
            dist_out = img_dir / dist_fname

            if dist_out.exists():
                rows.append({
                    "ref_img":   ref_name,
                    "dist_img":  dist_fname,
                    "dist_type": dist_type,
                    "dist_name": dist_name,
                    "level":     level,
                    "param":     param_label,
                    "value":     str(param_value),
                })
                continue

            try:
                degraded = apply_distortion(img, dist_type, level_idx, rng)
                degraded.save(dist_out, format="PNG")
                rows.append({
                    "ref_img":   ref_name,
                    "dist_img":  dist_fname,
                    "dist_type": dist_type,
                    "dist_name": dist_name,
                    "level":     level,
                    "param":     param_label,
                    "value":     str(param_value),
                })
            except Exception as e:
                logger.warning(f"  [ref {ref_idx + 1} type {dist_type} level {level}] ERREUR : {e}")

    return rows


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-ref",    type=int, default=N_REF,
                   help=f"Nombre d'images de référence (défaut={N_REF})")
    p.add_argument("--size",     type=int, default=TARGET_SIZE,
                   help=f"Taille carrée de sortie (défaut={TARGET_SIZE})")
    p.add_argument("--workers",  type=int, default=4,
                   help="Threads parallèles (défaut=4)")
    p.add_argument("--dist-types", type=str, default=None,
                   help="Sous-ensemble de types ex: '1,2,10,11' (défaut=tous)")
    return p.parse_args()


def main():
    args = parse_args()

    img_dir = OUT_DIR / "images"
    ref_dir = OUT_DIR / "references"
    img_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "metadata.csv"

    dist_types_to_run = list(range(1, 26))
    if args.dist_types:
        dist_types_to_run = [int(x) for x in args.dist_types.split(",")]
        # Filtrer les distorsions non demandées
        for k in list(DISTORTIONS.keys()):
            if k not in dist_types_to_run:
                pass  # garder pour le CSV mais pas générer

    ref_paths = select_reference_images(args.n_ref)
    total_imgs = len(ref_paths) * len(dist_types_to_run) * 5

    logger.info(f"Dataset : {len(ref_paths)} refs × {len(dist_types_to_run)} types × 5 niveaux "
                f"= {total_imgs} images")
    logger.info(f"Sortie  : {OUT_DIR}")
    logger.info(f"Taille  : {args.size}×{args.size}  |  Threads : {args.workers}")

    all_rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_reference, i, path, img_dir, ref_dir, args.size
            ): i
            for i, path in enumerate(ref_paths)
        }
        with tqdm(total=len(ref_paths), desc="Références", unit="ref") as pbar:
            for fut in as_completed(futures):
                try:
                    rows = fut.result()
                    all_rows.extend(rows)
                except Exception as e:
                    logger.error(f"Erreur ref {futures[fut]} : {e}")
                pbar.update(1)

    # Trier par (ref_img, dist_type, level) comme KADID dmos.csv
    all_rows.sort(key=lambda r: (r["ref_img"], r["dist_type"], r["level"]))

    # Écrire le CSV
    fieldnames = ["ref_img", "dist_img", "dist_type", "dist_name", "level", "param", "value"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"CSV écrit : {csv_path}  ({len(all_rows)} lignes)")
    logger.info(f"Images dans : {img_dir}")
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
