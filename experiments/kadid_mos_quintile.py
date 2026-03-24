"""
kadid_mos_quintile.py
=====================
Protocole MOS-global sur KADID-10k.

Toutes les 10 125 images dégradées sont triées globalement par DMOS croissant
(meilleure qualité → pire qualité), puis découpées en 405 groupes consécutifs
de 25 images chacun (25 × 405 = 10 125).

Chaque groupe contient un mélange de types de dégradations et d'images de
référence, mais toutes les images d'un groupe ont un DMOS similaire.

Pour chaque groupe g (g = 1..405) :
  - Calculer le 2-sample MMD² biaisé entre le groupe et les 81 références
  - Tester si MMD²(g) augmente avec g (c'est-à-dire avec le DMOS)

2-sample MMD² biaisé :
  MMD²(Q, R) = K̄_QQ + K̄_RR - 2 · K̄_QR
  avec K(a,b) = exp(-γ · ||a-b||²)
  K̄_RR est constant pour tous les groupes → précalculé une seule fois.

Métriques sur la séquence de 405 valeurs :
  - strict_mono_rate  : fraction des 404 paires consécutives où MMD² croît
  - kendall_tau       : corrélation Kendall τ entre rang_groupe et MMD²
  - spearman_rho      : corrélation Spearman ρ entre rang_groupe et MMD²

Configurations testées :
  - SD-VAE   L11  γ = 1e-4
  - DC-AE    L09  γ = 5e-6
  - DINOv2   L11  γ = 2.8e-5
  - CMMD     CLIP ViT-L/14@336, γ = 0.005

Usage :
    .venv/bin/python experiments/kadid_mos_quintile.py
    .venv/bin/python experiments/kadid_mos_quintile.py --skip-cmmd
    .venv/bin/python experiments/kadid_mos_quintile.py --device cpu
"""

import argparse
import csv
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE_DIR  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from gram_mmd.config import get_backbone_config
from gram_mmd.features import FeatureHook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

KADID_DIR  = BASE_DIR / "dataset" / "kadid10k"
IMAGES_DIR = KADID_DIR / "images"
DMOS_CSV   = KADID_DIR / "dmos.csv"
COCO_DIR   = BASE_DIR / "dataset" / "coco_5000"
N_ANCHOR   = 1000   # nombre d'images COCO utilisées comme ancrage
OUT_DIR    = BASE_DIR / "results" / "kadid_mos_coco1000_g27"

N_TOTAL     = 10125   # 81 × 25 × 5
N_GROUPS    = 27      # groupes globaux triés par DMOS  (10125 / 375 = 27)
N_PER_GROUP = 375     # images par groupe

SELECTED_CONFIGS = [
    {"backbone": "sd_vae",        "layer": 11, "gamma": 1e-4,   "batch_size": 16},
    {"backbone": "dc_ae",         "layer": 9,  "gamma": 5e-6,   "batch_size": 8},
    {"backbone": "dinov2_vitb14", "layer": 11, "gamma": 2.8e-5, "batch_size": 8},
]

MAX_CHANNELS  = 512
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CMMD_GAMMA    = 0.005
BATCH_CLIP    = 32


# ============================================================
# Dataset util
# ============================================================

class _ImageDataset(Dataset):
    def __init__(self, paths: list, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.paths[idx]
        except Exception as e:
            logger.warning(f"Cannot load {self.paths[idx]}: {e}")
            h = w = self.transform.transforms[0].size
            if isinstance(h, (list, tuple)):
                h, w = h
            return torch.zeros(3, h, w), self.paths[idx]


# ============================================================
# KADID data loading — tri global par DMOS
# ============================================================

def load_kadid_global_groups() -> tuple:
    """
    Lit dmos.csv, trie TOUTES les images dégradées par DMOS croissant,
    découpe en 405 groupes consécutifs de 25 images.
    L'ancrage est constitué de N_ANCHOR images COCO (externes, propres).

    Returns:
        anchor_paths : list de N_ANCHOR chemins COCO (ancrage externe)
        groups       : list de 405 listes de 25 chemins (dégradées, triées globalement)
        dmos_means   : np.ndarray (405,) — DMOS moyen par groupe (info)
    """
    all_pairs = []    # (dmos, dist_img)

    with open(DMOS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            all_pairs.append((float(r["dmos"]), r["dist_img"]))

    assert len(all_pairs) == N_TOTAL, (
        f"Expected {N_TOTAL} entries, got {len(all_pairs)}"
    )

    # Tri global par DMOS croissant
    all_pairs.sort(key=lambda x: x[0])

    # Ancrage : N_ANCHOR premières images COCO (triées par nom)
    coco_paths = sorted(COCO_DIR.glob("*.jpg"))
    assert len(coco_paths) >= N_ANCHOR, (
        f"Pas assez d'images COCO : {len(coco_paths)} < {N_ANCHOR}"
    )
    anchor_paths = [str(p) for p in coco_paths[:N_ANCHOR]]

    groups     = []
    dmos_means = np.zeros(N_GROUPS, dtype=np.float32)

    for g in range(N_GROUPS):
        chunk = all_pairs[g * N_PER_GROUP : (g + 1) * N_PER_GROUP]
        dmos_means[g] = float(np.mean([c[0] for c in chunk]))
        groups.append([str(IMAGES_DIR / c[1]) for c in chunk])

    logger.info(
        f"KADID global: {N_TOTAL} images → {N_GROUPS} groupes × {N_PER_GROUP} images"
    )
    logger.info(
        f"  DMOS range: {dmos_means[0]:.3f} (G1) → {dmos_means[-1]:.3f} (G{N_GROUPS})"
    )
    logger.info(f"  Ancrage : {N_ANCHOR} images COCO (externe, propres)")

    return anchor_paths, groups, dmos_means


# ============================================================
# Feature extractor
# ============================================================

class SingleLayerExtractor:
    def __init__(self, backbone_name: str, layer_idx: int, device: str,
                 batch_size: int, max_channels: int = MAX_CHANNELS):
        self.backbone_name = backbone_name
        self.layer_idx     = layer_idx
        self.device        = device
        self.batch_size    = batch_size
        self.max_channels  = max_channels
        self.cfg           = get_backbone_config(backbone_name)
        self.model         = None
        self.hook          = FeatureHook()
        self.hook_handle   = None
        self.scaler        = None
        self._load_model()
        self._register_hook()
        self.transform = self._make_transform()

    def _load_model(self):
        name = self.backbone_name
        cfg  = self.cfg
        logger.info(f"  Loading {name} ...")
        if name in ("sd_vae", "flux_vae"):
            from diffusers import AutoencoderKL
            self.model = AutoencoderKL.from_pretrained(cfg["weights"])
        elif name == "dc_ae":
            from diffusers import AutoencoderDC
            self.model = AutoencoderDC.from_pretrained(cfg["weights"])
        elif name == "dinov2_vitb14":
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )
        elif name == "vgg19":
            from torchvision import models as tv
            self.model = tv.vgg19(weights=tv.VGG19_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Backbone not supported: {name}")
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _get_module(self, dotted: str):
        m = self.model
        for part in dotted.split("."):
            try:
                m = getattr(m, part)
            except AttributeError:
                m = m[int(part)]
        return m

    def _register_hook(self):
        layer_name = self.cfg["layer_names"][self.layer_idx]
        mod = self._get_module(layer_name)
        self.hook_handle = mod.register_forward_hook(self.hook)
        logger.info(f"  Hook: L{self.layer_idx} → {layer_name}")

    def _make_transform(self):
        cfg = self.cfg
        return transforms.Compose([
            transforms.Resize((cfg["input_size"], cfg["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["normalize_mean"],
                                 std=cfg["normalize_std"]),
        ])

    def _forward(self, imgs: torch.Tensor):
        imgs = imgs.to(self.device)
        self.hook.features = None
        with torch.no_grad():
            if self.backbone_name in ("sd_vae", "flux_vae", "dc_ae"):
                _ = self.model.encoder(imgs)
            else:
                _ = self.model(imgs)

    def _gram(self, feat: torch.Tensor) -> np.ndarray:
        feat = feat.float()
        if feat.dim() == 4:
            B, C, H, W = feat.shape
            if C > self.max_channels:
                gen = torch.Generator(device="cpu").manual_seed(42)
                idx = torch.randperm(C, generator=gen)[:self.max_channels]
                feat = feat[:, idx.to(feat.device)]
                C = self.max_channels
            F = feat.reshape(B, C, H * W)
        elif feat.dim() == 3:
            B, N, D = feat.shape
            if D > self.max_channels:
                gen = torch.Generator(device="cpu").manual_seed(42)
                idx = torch.randperm(D, generator=gen)[:self.max_channels]
                feat = feat[:, :, idx.to(feat.device)]
                D = self.max_channels
            F = feat.permute(0, 2, 1)
            C = D
        elif feat.dim() == 2:
            return feat.cpu().numpy().astype(np.float32)
        else:
            return None

        S = F.shape[2]
        G = torch.bmm(F, F.transpose(1, 2)) / S
        triu = torch.triu(torch.ones(C, C, device=feat.device, dtype=torch.bool))
        return G[:, triu].cpu().numpy().astype(np.float32)

    def _extract_batch(self, imgs: torch.Tensor) -> np.ndarray | None:
        self._forward(imgs)
        feat = self.hook.features
        if feat is None:
            return None
        return self._gram(feat)

    def extract_all(self, paths: list) -> np.ndarray:
        ds     = _ImageDataset(paths, self.transform)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=(self.device == "cuda"),
            persistent_workers=True,
        )
        parts = []
        for imgs, _ in loader:
            g = self._extract_batch(imgs)
            if g is not None:
                parts.append(g)
        return np.concatenate(parts, axis=0)

    def fit_scaler(self, ref_feats: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        ref_scaled = self.scaler.fit_transform(ref_feats).astype(np.float32)
        logger.info(
            f"  Scaler fitted — mean={self.scaler.mean_.mean():.3e}  "
            f"std={self.scaler.scale_.mean():.3e}"
        )
        return ref_scaled

    def get_layer_name(self) -> str:
        return self.cfg["layer_names"][self.layer_idx]

    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# 2-sample biased MMD²
# ============================================================

def mmd2_biased(Q: np.ndarray, R: np.ndarray, gamma: float,
                K_RR_mean: float) -> float:
    """
    2-sample biased MMD²(Q, R) = K̄_QQ + K̄_RR - 2·K̄_QR
    K̄_RR est fourni précalculé (constant pour tous les groupes).
    """
    Q_t = torch.tensor(Q)
    R_t = torch.tensor(R)

    q_sq = (Q_t * Q_t).sum(1)
    r_sq = (R_t * R_t).sum(1)

    D2_QQ = torch.clamp(q_sq[:, None] + q_sq[None, :] - 2.0 * (Q_t @ Q_t.T), min=0.0)
    K_QQ_mean = float(torch.exp(-gamma * D2_QQ).mean())

    D2_QR = torch.clamp(q_sq[:, None] + r_sq[None, :] - 2.0 * (Q_t @ R_t.T), min=0.0)
    K_QR_mean = float(torch.exp(-gamma * D2_QR).mean())

    return K_QQ_mean + K_RR_mean - 2.0 * K_QR_mean


# ============================================================
# Monotonicity metrics — séquence globale de 405 valeurs
# ============================================================

def compute_metrics(mmd2_vector: np.ndarray) -> dict:
    """
    Args:
        mmd2_vector : (405,) — MMD²[g] pour g = 0..404
                      g=0 = meilleure qualité (DMOS le plus bas)
                      g=404 = pire qualité (DMOS le plus haut)
    Returns:
        dict avec spearman_rho, strict_mono_rate, kendall_tau_mean
    """
    n = len(mmd2_vector)
    ranks = np.arange(1, n + 1)

    # Spearman ρ global
    rho, _ = spearmanr(ranks, mmd2_vector)

    # Monotonie stricte : fraction des 404 paires consécutives croissantes
    diffs = np.diff(mmd2_vector)
    strict_mono_rate = float(np.mean(diffs > 0))

    # Kendall τ global
    tau, _ = kendalltau(ranks, mmd2_vector)

    return {
        "spearman_rho":     float(rho),
        "strict_mono_rate": strict_mono_rate,
        "kendall_tau_mean": float(tau) if np.isfinite(tau) else 0.0,
        "n_groups":         n,
    }


# ============================================================
# CMMD (CLIP ViT-L/14@336)
# ============================================================

def run_cmmd_global(ref_paths: list, groups: list, device: str) -> np.ndarray:
    """
    Embedding CLIP des N_ANCHOR images COCO (ancrage) + 405 groupes.
    Retourne mmd2_vector (405,).
    """
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    logger.info(f"  Loading {CLIP_MODEL_ID} ...")
    model = (
        CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
        .to(device).eval()
    )
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)

    def _embed(paths: list) -> np.ndarray:
        parts = []
        for i in range(0, len(paths), BATCH_CLIP):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH_CLIP]]
            pv = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                parts.append(
                    model(pixel_values=pv).image_embeds.float().cpu().numpy()
                )
        return np.concatenate(parts, axis=0).astype(np.float32)

    logger.info("  Embedding references ...")
    R = _embed(ref_paths)
    r_sq = np.einsum("ij,ij->i", R, R)
    K_RR = np.exp(-CMMD_GAMMA * (r_sq[:, None] + r_sq[None, :] - 2.0 * (R @ R.T)))
    K_RR_mean = float(K_RR.mean())
    logger.info(f"  K̄_RR = {K_RR_mean:.6f}")

    mmd2_vector = np.zeros(len(groups), dtype=np.float64)
    t0 = time.time()

    for g, group_paths in enumerate(groups):
        Q = _embed(group_paths)
        q_sq = np.einsum("ij,ij->i", Q, Q)
        K_QQ_mean = float(np.exp(-CMMD_GAMMA * (
            q_sq[:, None] + q_sq[None, :] - 2.0 * (Q @ Q.T)
        )).mean())
        K_QR_mean = float(np.exp(-CMMD_GAMMA * (
            q_sq[:, None] + r_sq[None, :] - 2.0 * (Q @ R.T)
        )).mean())
        mmd2_vector[g] = K_QQ_mean + K_RR_mean - 2.0 * K_QR_mean

        if (g + 1) % 50 == 0 or (g + 1) == len(groups):
            logger.info(f"    CMMD group {g+1}/{len(groups)} — {time.time()-t0:.1f}s")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return mmd2_vector


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cmmd", action="store_true")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("KADID-10k — Protocole MOS global (2-sample MMD²)")
    logger.info(f"  {N_TOTAL} images → {N_GROUPS} groupes × {N_PER_GROUP} (tri global DMOS)")
    logger.info(f"  Ancrage : {N_ANCHOR} images COCO (externe)")
    logger.info(f"  Device : {args.device}")
    logger.info("=" * 70)

    ref_paths, groups, dmos_means = load_kadid_global_groups()

    results = []   # list of dicts for CSV output

    # ── Gram-MMD configs ──────────────────────────────────────
    for cfg in SELECTED_CONFIGS:
        bb    = cfg["backbone"]
        layer = cfg["layer"]
        gamma = cfg["gamma"]
        bs    = cfg["batch_size"]

        logger.info(f"\n{'='*60}")
        logger.info(f"  {bb}  L{layer:02d}  γ={gamma:.1e}")
        logger.info(f"{'='*60}")

        extractor = SingleLayerExtractor(
            backbone_name=bb, layer_idx=layer,
            device=args.device, batch_size=bs,
        )

        # Extraire et standardiser les features des références
        logger.info(f"  Extracting {len(ref_paths)} reference features ...")
        ref_raw    = extractor.extract_all(ref_paths)
        ref_scaled = extractor.fit_scaler(ref_raw)

        # Précalculer K̄_RR (constant pour tous les groupes)
        R_t   = torch.tensor(ref_scaled)
        r_sq  = (R_t * R_t).sum(1)
        D2_RR = torch.clamp(r_sq[:, None] + r_sq[None, :] - 2.0 * (R_t @ R_t.T), min=0.0)
        K_RR_mean = float(torch.exp(-gamma * D2_RR).mean())
        logger.info(f"  K̄_RR = {K_RR_mean:.6f}")
        del D2_RR

        # Calculer le MMD² 2-sample pour chacun des 405 groupes
        mmd2_vector = np.zeros(len(groups), dtype=np.float64)
        t0 = time.time()

        for g, group_paths in enumerate(groups):
            Q_raw    = extractor.extract_all(group_paths)
            Q_scaled = extractor.scaler.transform(Q_raw).astype(np.float32)
            mmd2_vector[g] = mmd2_biased(Q_scaled, ref_scaled, gamma, K_RR_mean)

            if (g + 1) % 50 == 0 or (g + 1) == len(groups):
                elapsed = time.time() - t0
                logger.info(
                    f"    group {g+1}/{len(groups)} — {elapsed:.1f}s  "
                    f"mmd2={mmd2_vector[g]:.4e}  dmos_mean={dmos_means[g]:.3f}"
                )

        met = compute_metrics(mmd2_vector)
        logger.info(
            f"  ρ={met['spearman_rho']:+.4f}  "
            f"mono={met['strict_mono_rate']:.3f}  "
            f"τ={met['kendall_tau_mean']:+.4f}"
        )

        results.append({
            "backbone": bb, "layer": layer, "gamma": gamma,
            "layer_name": extractor.get_layer_name(),
            **met,
        })

        # Sauvegarder le vecteur MMD²
        tag      = f"{bb}_L{layer:02d}_g{gamma:.0e}"
        npy_path = OUT_DIR / f"mmd2_vector_{tag}_{ts}.npy"
        np.save(npy_path, mmd2_vector)
        logger.info(f"  MMD² vector → {npy_path}")

        extractor.cleanup()

    # ── CMMD ─────────────────────────────────────────────────
    if not args.skip_cmmd:
        logger.info(f"\n{'='*60}")
        logger.info("  CMMD  (CLIP ViT-L/14@336)")
        logger.info(f"{'='*60}")

        mmd2_cmmd = run_cmmd_global(ref_paths, groups, args.device)
        met_cmmd  = compute_metrics(mmd2_cmmd)

        logger.info(
            f"  CMMD  ρ={met_cmmd['spearman_rho']:+.4f}  "
            f"mono={met_cmmd['strict_mono_rate']:.3f}  "
            f"τ={met_cmmd['kendall_tau_mean']:+.4f}"
        )

        results.append({
            "backbone": "cmmd", "layer": 0, "gamma": CMMD_GAMMA,
            "layer_name": "CLIP ViT-L/14@336",
            **met_cmmd,
        })

        tag      = f"cmmd_L00_g{CMMD_GAMMA:.0e}"
        npy_path = OUT_DIR / f"mmd2_vector_{tag}_{ts}.npy"
        np.save(npy_path, mmd2_cmmd)
        logger.info(f"  MMD² vector → {npy_path}")

    # ── Résumé ────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS — KADID-10k MOS global  (2-sample MMD², {N_GROUPS} groupes × {N_PER_GROUP}, ancrage COCO {N_ANCHOR})")
    logger.info(f"{'Backbone':<18} {'Layer':<6} {'γ':<10} {'Spearman ρ':>10}  "
                f"{'Strict mono':>11}  {'Kendall τ':>9}")
    logger.info("-" * 70)
    for r in sorted(results, key=lambda x: -x["spearman_rho"]):
        logger.info(
            f"  {r['backbone']:<16} L{r['layer']:<4} "
            f"{r['gamma']:<10.1e} {r['spearman_rho']:>+10.4f}  "
            f"{r['strict_mono_rate']:>11.3f}  {r['kendall_tau_mean']:>+9.4f}"
        )
    logger.info("=" * 70)

    logger.info(f"\nAncrage : {N_ANCHOR} images COCO")
    logger.info("\nDMOS moyen par groupe (10 premiers et 10 derniers) :")
    for g in list(range(10)) + list(range(N_GROUPS - 10, N_GROUPS)):
        logger.info(f"  G{g+1:03d}: {dmos_means[g]:.3f}")

    # ── CSV ───────────────────────────────────────────────────
    csv_path = OUT_DIR / f"global_results_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["backbone", "layer", "layer_name", "gamma",
                  "spearman_rho", "strict_mono_rate", "kendall_tau_mean", "n_groups"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    logger.info(f"\nCSV → {csv_path}")

    logger.info(f"\nDone. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
