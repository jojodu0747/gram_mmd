==============================================================================
  GMMD_test — Gram-MMD Image Quality Distance Tool
==============================================================================

WHAT IS GRAM-MMD?
-----------------
Gram-MMD is a perceptual image quality metric based on two ideas:

  1. GRAM MATRIX (Gatys et al., 2016)
     For an image passed through a deep network, we capture the activation
     map F of shape (C, H, W) at a chosen layer. The Gram matrix is:

         G = F @ F^T / (H*W)    →  shape (C, C)

     The upper-triangular elements of G form a vector g ∈ R^(C*(C+1)/2).
     This vector encodes the *texture* and *style* of the image at that
     feature level, independently of spatial arrangement.

  2. MAXIMUM MEAN DISCREPANCY — MMD (Gretton et al., 2012)
     MMD measures the distance between two distributions using an RBF kernel:

         k(a, b) = exp(-γ · ||a - b||²)

     Two usage modes are supported:

     a) POINT vs DISTRIBUTION  (each image in evaluation_set/ scored individually)

         MMD²(x, R) = mean k(r_i, r_j)  +  k(x, x)  -  2/n · Σ_i k(x, r_i)
                                              ↑ = 1 for RBF kernel

         Interpretation: score ≈ 0 when x looks like the anchor distribution.
         Score increases as x becomes more different (degraded, out-of-domain…).

     b) DISTRIBUTION vs DISTRIBUTION  (each folder in evaluation_set/ scored)

         MMD²(Q, R) = mean k(r_i,r_j)  +  mean k(q_i,q_j)  -  2·mean k(r_i,q_j)

         Interpretation: distance between two sets of images.


HOW TO SET UP
-------------
1. Create a virtual environment and install dependencies (once):

       python -m venv venv
       source venv/bin/activate          # Linux/Mac
       # venv\Scripts\activate           # Windows

       pip install -r requirements.txt

   For GPU support, install the appropriate CUDA version of PyTorch instead:
       https://pytorch.org/get-started/locally/
   Example (CUDA 12.1):
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
       pip install -r requirements.txt

2. Put your ANCHOR images in:

       dataset/anchor_set/

   These represent your reference / "good quality" distribution.
   Use 50–500 images. Accepted formats: jpg, jpeg, png, bmp, tiff, webp.

3. Put what you want to evaluate in:

       dataset/evaluation_set/

   TWO MODES depending on what you put there:

   MODE A — Point vs Distribution (one score per image):
     Put image files directly in evaluation_set/.
     Example:
       evaluation_set/
         img_001.jpg
         img_002.jpg
         img_003.png

   MODE B — Distribution vs Distribution (one score per folder):
     Put sub-folders, each containing a set of images.
     Example:
       evaluation_set/
         level_1/   (10 images)
         level_2/   (10 images)
         level_3/   (10 images)

   The mode is detected automatically: if evaluation_set/ contains
   sub-directories → Mode B; if only files → Mode A.

4. Configure config.py (see next section).

5. Run:

       python run.py

   Results are saved to results.csv.


HOW TO CONFIGURE (config.py)
-----------------------------
Open config.py. The key settings are:

  BACKBONES   — dict with one entry per backbone.
                Set  enabled: True   to activate it.
                Set  layer: <int>    to select the encoder layer.

  GAMMAS      — list of γ values for the RBF kernel.
                Use None for automatic median heuristic.

  DEVICE      — "cuda" (GPU) or "cpu".

RECOMMENDED SETTINGS (validated on COCO degradation benchmarks):

    "sd_vae"  →  layer: 7,  gamma: 2.8e-5

  This gives Spearman ρ ≈ 0.78 on blur/noise/JPEG distortions.
  SD-VAE (Stable Diffusion VAE, Rombach et al. 2022) is a convolutional
  autoencoder trained on natural images with a strong texture-aware encoder.

MULTIPLE BACKBONES / GAMMAS:
  If you enable several backbones or list several gammas, the output CSV
  will contain one row per (evaluation_unit, backbone, gamma) combination.
  Example with 2 backbones × 3 gammas × 50 images = 300 rows in the CSV.

  This lets you compare which configuration is most sensitive to your
  specific quality criterion without re-running feature extraction.


OUTPUT FORMAT (results.csv)
-----------------------------
Columns:
  name        — image filename (Mode A) or folder name (Mode B)
  mode        — "point" or "distribution"
  backbone    — backbone name used
  layer       — layer index used
  gamma       — γ value actually used (numerical, even if None was specified)
  mmd_score   — MMD² value (non-negative float)

Higher MMD score = more different from the anchor distribution.


AVAILABLE BACKBONES
--------------------
Backbone         Hugging Face / source               Recommended layer
--------         ---------------------               -----------------
sd_vae           stabilityai/sd-vae-ft-mse           7   (512ch, 64×64)
flux_vae         diffusers/FLUX.1-vae                7   (512ch, 64×64)
dc_ae            mit-han-lab/dc-ae-f64c128-*         10  (512ch)
dinov2_vitb14    facebookresearch/dinov2 (hub)        9   (ViT block 8)
vgg19            torchvision IMAGENET1K_V1            8   (features.19)
lpips_vgg        lpips package (vgg)                  5   (slice3.12)

Models are downloaded automatically on first use and cached in
~/.cache/huggingface/ (HuggingFace models) or ~/.cache/torch/ (others).
Set environment variable HF_HOME to redirect the cache if needed.


TIPS
----
- Start with sd_vae + gamma=2.8e-5. It is fast and well-validated.
- More anchor images = more stable MMD estimates. Below 50, variance is high.
- In Mode B, each folder should contain at least 20 images for reliable MMD.
- For CPU-only machines, set DEVICE="cpu" and reduce batch sizes to 4-8.
- If you get GPU out-of-memory errors, halve the batch_size in config.py.


REFERENCES
----------
- Gatys et al. (2016). "A Neural Algorithm of Artistic Style." CVPR.
- Gretton et al. (2012). "A Kernel Two-Sample Test." JMLR.
- Rombach et al. (2022). "High-Resolution Image Synthesis with Latent
  Diffusion Models." CVPR.
==============================================================================
