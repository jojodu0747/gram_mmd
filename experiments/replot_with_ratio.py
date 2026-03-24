"""
replot_with_ratio.py
====================
Replot dc_ae L12 et dc_ae L14 depuis le JSON sauvegardé.
Génère 4 variantes par config (ancienne + 3 nouvelles) sans écraser les anciens plots.

Variantes :
  ratio_*          : original (ratio explose à droite)
  ratio_clip3_*    : ratio Gram clippé à [0, 3]
  ratio_logy_*     : ratio Gram en log scale Y
  ratio_zoom_*     : zoom γ/100 → 10×γ_med uniquement
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "experiments/results/gamma_sweep_top_configs/gamma_sweep_top_configs_20260323_095513.json"
OUT_DIR   = BASE_DIR / "experiments/results/gamma_sweep_top_configs"
CMMD_GAMMA = 0.005

CONFIGS_TO_PLOT = [
    ("dc_ae", 12, "Kendall #3 (τ=0.872)"),
    ("dc_ae", 14, "Spearman #7 (ρ=0.902)"),
]


def plot_with_ratio(gammas_gram, mmd_vk_gram, mmd_sc_gram,
                    gammas_clip, mmd_vk_clip, mmd_sc_clip,
                    cmmd_fixed_vk, cmmd_fixed_sc,
                    g_med_gram, g_med_clip,
                    config_label, ranking_label, out_path):

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"MMD² sweep — KITTI 1k anchor | VKITTI2 vs Stanford Cars\n"
                 f"{config_label}  ({ranking_label})",
                 fontsize=12, fontweight="bold")

    # ── Panel 1 : Gram-MMD courbes ────────────────────────────────
    ax = axes[0]
    ax.semilogx(gammas_gram, mmd_vk_gram, "b-o", markersize=3, label="VKITTI2 (synthetic)")
    ax.semilogx(gammas_gram, mmd_sc_gram,  "r-s", markersize=3, label="Stanford Cars (real)")
    inv_mask = mmd_vk_gram < mmd_sc_gram
    if inv_mask.any():
        ax.axvspan(gammas_gram[inv_mask].min(), gammas_gram[inv_mask].max(),
                   alpha=0.12, color="red", label="zone inversion")
    ax.fill_between(gammas_gram,
                    np.minimum(mmd_vk_gram, mmd_sc_gram),
                    np.maximum(mmd_vk_gram, mmd_sc_gram),
                    alpha=0.15, color="orange" if inv_mask.any() else "green")
    ax.axvline(g_med_gram, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"γ_med={g_med_gram:.1e}")
    ax.axvline(g_med_gram * 0.1, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.annotate("γ/10", xy=(g_med_gram * 0.1, ax.get_ylim()[1] * 0.05),
                fontsize=7, color="gray")
    ax.set_xlabel("γ  (log scale)")
    ax.set_ylabel("MMD² (unbiased)")
    ax.set_title(f"Gram-MMD ({config_label})")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel 2 : Gram-MMD ratio ──────────────────────────────────
    ax = axes[1]
    ratio_gram = np.where(mmd_sc_gram > 1e-15, mmd_vk_gram / mmd_sc_gram, np.nan)
    ax.semilogx(gammas_gram, ratio_gram, "k-o", markersize=3, label="ratio VKITTI2 / Stanford")
    ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="seuil inversion (ratio=1)")
    ax.fill_between(gammas_gram, ratio_gram, 1.0,
                    where=(ratio_gram < 1.0), alpha=0.2, color="red", label="zone inversée")
    ax.fill_between(gammas_gram, ratio_gram, 1.0,
                    where=(ratio_gram >= 1.0), alpha=0.1, color="green")
    ax.axvline(g_med_gram, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"γ_med={g_med_gram:.1e}")
    # Annoter valeur du ratio à γ_med et γ/10
    for mul, name in [(1.0, "γ_med"), (0.1, "γ/10")]:
        g_ref = g_med_gram * mul
        idx = np.argmin(np.abs(gammas_gram - g_ref))
        r = ratio_gram[idx]
        if not np.isnan(r):
            ax.annotate(f"{r:.3f}", xy=(gammas_gram[idx], r),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=8, ha="center", color="navy",
                        arrowprops=dict(arrowstyle="-", color="navy", lw=0.5))
    ax.set_xlabel("γ  (log scale)")
    ax.set_ylabel("MMD²(VKITTI2) / MMD²(Stanford)")
    ax.set_title(f"Ratio Gram-MMD — {config_label}\n(< 1.0 = inversion)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel 3 : CMMD courbes + ratio ────────────────────────────
    ax = axes[2]
    ax2 = ax.twinx()
    ax.semilogx(gammas_clip, mmd_vk_clip, "b-o", markersize=2, alpha=0.6, label="VKITTI2 (CMMD)")
    ax.semilogx(gammas_clip, mmd_sc_clip,  "r-s", markersize=2, alpha=0.6, label="Stanford (CMMD)")
    ratio_clip = np.where(mmd_sc_clip > 1e-15, mmd_vk_clip / mmd_sc_clip, np.nan)
    ax2.semilogx(gammas_clip, ratio_clip, "k--", linewidth=1.5, alpha=0.8, label="ratio VKITTI2/Stanford")
    ax2.axhline(1.0, color="red", linewidth=1.5, linestyle=":", alpha=0.8)
    # γ=0.005 fixe
    if gammas_clip.min() <= CMMD_GAMMA <= gammas_clip.max():
        ax.axvline(CMMD_GAMMA, color="purple", linestyle=":", linewidth=1.5,
                   label=f"γ=0.005 (CMMD fixe)")
        ax.plot(CMMD_GAMMA, cmmd_fixed_vk, "b*", markersize=10)
        ax.plot(CMMD_GAMMA, cmmd_fixed_sc, "r*", markersize=10)
        idx_f = np.argmin(np.abs(gammas_clip - CMMD_GAMMA))
        ax2.annotate(f"ratio={ratio_clip[idx_f]:.3f}", xy=(CMMD_GAMMA, ratio_clip[idx_f]),
                     xytext=(10, 0), textcoords="offset points",
                     fontsize=8, color="purple",
                     arrowprops=dict(arrowstyle="-", color="purple", lw=0.5))
    ax.axvline(g_med_clip, color="darkgreen", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"γ_med CLIP={g_med_clip:.1e}")
    ax.set_xlabel("γ  (log scale)")
    ax.set_ylabel("MMD² (unbiased)", color="gray")
    ax2.set_ylabel("Ratio VKITTI2 / Stanford", color="black")
    ax.set_title("CMMD (CLIP ViT-L/14)\n(droite: ratio, rouge=inversion)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_variant(gammas_gram, mmd_vk_gram, mmd_sc_gram,
                 gammas_clip, mmd_vk_clip, mmd_sc_clip,
                 cmmd_fixed_vk, cmmd_fixed_sc,
                 g_med_gram, g_med_clip,
                 config_label, ranking_label, out_path,
                 ratio_ylim=None, ratio_logy=False, gamma_xlim=None):
    """Variante paramétrable du plot_with_ratio."""

    # Optionnel : zoom sur plage gamma
    if gamma_xlim is not None:
        gmin, gmax = gamma_xlim
        mask_g = (gammas_gram >= gmin) & (gammas_gram <= gmax)
        gammas_gram  = gammas_gram[mask_g]
        mmd_vk_gram  = mmd_vk_gram[mask_g]
        mmd_sc_gram  = mmd_sc_gram[mask_g]
        mask_c = (gammas_clip >= gmin) & (gammas_clip <= gmax)
        gammas_clip = gammas_clip[mask_c]
        mmd_vk_clip = mmd_vk_clip[mask_c]
        mmd_sc_clip = mmd_sc_clip[mask_c]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"MMD² sweep — KITTI 1k anchor | VKITTI2 vs Stanford Cars\n"
                 f"{config_label}  ({ranking_label})",
                 fontsize=12, fontweight="bold")

    # Panel 1 : Gram courbes
    ax = axes[0]
    ax.semilogx(gammas_gram, mmd_vk_gram, "b-o", markersize=3, label="VKITTI2 (synthetic)")
    ax.semilogx(gammas_gram, mmd_sc_gram,  "r-s", markersize=3, label="Stanford Cars (real)")
    inv_mask = mmd_vk_gram < mmd_sc_gram
    if inv_mask.any():
        ax.axvspan(gammas_gram[inv_mask].min(), gammas_gram[inv_mask].max(),
                   alpha=0.12, color="red", label="zone inversion")
    ax.fill_between(gammas_gram,
                    np.minimum(mmd_vk_gram, mmd_sc_gram),
                    np.maximum(mmd_vk_gram, mmd_sc_gram),
                    alpha=0.15, color="orange" if inv_mask.any() else "green")
    ax.axvline(g_med_gram, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"γ_med={g_med_gram:.1e}")
    g10 = g_med_gram * 0.1
    if gammas_gram.min() <= g10 <= gammas_gram.max():
        ax.axvline(g10, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate("γ/10", xy=(g10, ax.get_ylim()[1] * 0.05), fontsize=7, color="gray")
    ax.set_xlabel("γ  (log scale)")
    ax.set_ylabel("MMD² (unbiased)")
    ax.set_title(f"Gram-MMD ({config_label})")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2 : ratio Gram
    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_gram = np.where(mmd_sc_gram > 1e-15, mmd_vk_gram / mmd_sc_gram, np.nan)

    if ratio_logy:
        ax.semilogy(gammas_gram, ratio_gram, "k-o", markersize=3, label="ratio VKITTI2 / Stanford")
        ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="seuil inversion")
        ax.set_xscale("log")
    else:
        ax.semilogx(gammas_gram, ratio_gram, "k-o", markersize=3, label="ratio VKITTI2 / Stanford")
        ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="seuil inversion")

    if ratio_ylim is not None:
        ax.set_ylim(ratio_ylim)

    ax.fill_between(gammas_gram, np.clip(ratio_gram, 0, ratio_ylim[1] if ratio_ylim else 1e9), 1.0,
                    where=(ratio_gram < 1.0), alpha=0.2, color="red", label="zone inversée")
    ax.fill_between(gammas_gram, np.clip(ratio_gram, 0, ratio_ylim[1] if ratio_ylim else 1e9), 1.0,
                    where=(ratio_gram >= 1.0), alpha=0.1, color="green")

    ax.axvline(g_med_gram, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"γ_med={g_med_gram:.1e}")

    for mul, name in [(1.0, "γ_med"), (0.1, "γ/10")]:
        g_ref = g_med_gram * mul
        if gammas_gram.min() <= g_ref <= gammas_gram.max():
            idx = np.argmin(np.abs(gammas_gram - g_ref))
            r = ratio_gram[idx]
            if not np.isnan(r):
                ax.annotate(f"{r:.3f}", xy=(gammas_gram[idx], r),
                            xytext=(0, 10), textcoords="offset points",
                            fontsize=8, ha="center", color="navy",
                            arrowprops=dict(arrowstyle="-", color="navy", lw=0.5))

    ax.set_xlabel("γ  (log scale)")
    ax.set_ylabel("ratio MMD²(VKITTI2) / MMD²(Stanford)")
    logy_tag = " [log Y]" if ratio_logy else (f" [ylim {ratio_ylim}]" if ratio_ylim else "")
    ax.set_title(f"Ratio Gram-MMD — {config_label}{logy_tag}\n(< 1.0 = inversion)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 3 : CMMD + ratio
    ax = axes[2]
    ax2 = ax.twinx()
    ax.semilogx(gammas_clip, mmd_vk_clip, "b-o", markersize=2, alpha=0.6, label="VKITTI2 (CMMD)")
    ax.semilogx(gammas_clip, mmd_sc_clip,  "r-s", markersize=2, alpha=0.6, label="Stanford (CMMD)")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_clip = np.where(mmd_sc_clip > 1e-15, mmd_vk_clip / mmd_sc_clip, np.nan)
    ax2.semilogx(gammas_clip, ratio_clip, "k--", linewidth=1.5, alpha=0.8, label="ratio")
    ax2.axhline(1.0, color="red", linewidth=1.5, linestyle=":", alpha=0.8)

    if gammas_clip.min() <= CMMD_GAMMA <= gammas_clip.max():
        ax.axvline(CMMD_GAMMA, color="purple", linestyle=":", linewidth=1.5,
                   label=f"γ=0.005 (CMMD fixe)")
        ax.plot(CMMD_GAMMA, cmmd_fixed_vk, "b*", markersize=10)
        ax.plot(CMMD_GAMMA, cmmd_fixed_sc, "r*", markersize=10)
        idx_f = np.argmin(np.abs(gammas_clip - CMMD_GAMMA))
        if not np.isnan(ratio_clip[idx_f]):
            ax2.annotate(f"ratio={ratio_clip[idx_f]:.3f}", xy=(CMMD_GAMMA, ratio_clip[idx_f]),
                         xytext=(10, 0), textcoords="offset points",
                         fontsize=8, color="purple",
                         arrowprops=dict(arrowstyle="-", color="purple", lw=0.5))

    ax.axvline(g_med_clip, color="darkgreen", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"γ_med CLIP={g_med_clip:.1e}")
    ax.set_xlabel("γ  (log scale)")
    ax.set_ylabel("MMD² (unbiased)", color="gray")
    ax2.set_ylabel("Ratio VKITTI2 / Stanford", color="black")
    ax.set_title("CMMD (CLIP ViT-L/14)\n(droite: ratio, rouge=inversion)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)

    gammas_clip = np.array(d["clip_sweep"]["gammas"])
    mmd_vk_clip = np.array(d["clip_sweep"]["mmd_vkitti"])
    mmd_sc_clip = np.array(d["clip_sweep"]["mmd_stanford"])
    g_med_clip  = d["gamma_med_clip"]
    cmmd_fixed_vk = d["cmmd_fixed_vkitti"]
    cmmd_fixed_sc = d["cmmd_fixed_stanford"]

    for backbone, layer, ranking in CONFIGS_TO_PLOT:
        key = f"{backbone}_L{layer}"
        if key not in d["gram_configs"]:
            print(f"Config {key} not found in JSON!")
            continue
        cfg = d["gram_configs"][key]
        gammas_gram  = np.array(cfg["gammas"])
        mmd_vk_gram  = np.array(cfg["mmd_vkitti"])
        mmd_sc_gram  = np.array(cfg["mmd_stanford"])
        g_med_gram   = cfg["gamma_med"]
        config_label = f"{backbone} L{layer}"

        safe = ranking.replace(" ", "_").replace("#", "").replace("(", "").replace(")", "").replace("=", "").replace(".", "")

        common = dict(
            gammas_clip=gammas_clip, mmd_vk_clip=mmd_vk_clip, mmd_sc_clip=mmd_sc_clip,
            cmmd_fixed_vk=cmmd_fixed_vk, cmmd_fixed_sc=cmmd_fixed_sc,
            g_med_gram=g_med_gram, g_med_clip=g_med_clip,
            config_label=config_label, ranking_label=ranking,
        )

        # Variante 1 : ratio Y clippé à [0, 3]
        plot_variant(gammas_gram.copy(), mmd_vk_gram.copy(), mmd_sc_gram.copy(),
                     **common,
                     out_path=OUT_DIR / f"ratio_clip3_{safe}_{backbone}_L{layer}.png",
                     ratio_ylim=(0, 3))

        # Variante 2 : ratio Y en log scale
        plot_variant(gammas_gram.copy(), mmd_vk_gram.copy(), mmd_sc_gram.copy(),
                     **common,
                     out_path=OUT_DIR / f"ratio_logy_{safe}_{backbone}_L{layer}.png",
                     ratio_logy=True)

        # Variante 3 : zoom γ/100 → 10×γ_med
        plot_variant(gammas_gram.copy(), mmd_vk_gram.copy(), mmd_sc_gram.copy(),
                     **common,
                     out_path=OUT_DIR / f"ratio_zoom_{safe}_{backbone}_L{layer}.png",
                     ratio_ylim=(0, 3),
                     gamma_xlim=(g_med_gram * 0.01, g_med_gram * 10))


if __name__ == "__main__":
    main()
