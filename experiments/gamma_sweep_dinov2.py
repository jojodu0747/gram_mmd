"""
Analyse fine de ρ(γ) pour DINOv2 ViT-B/14 — 1000 gammas sur grille log.

Utilise les matrices D_rr et D_rq déjà cachées (pas de GPU nécessaire).
Vectorisé : calcul kernel par batch de gammas, Spearman rapide via rankdata.

Produit :
  results/sweep_coco_degraded/plots/
    08_dinov2_rho_vs_gamma.png   — courbes ρ(γ) par layer
    09_dinov2_heatmap.png        — heatmap ρ(layer × γ)
    10_dinov2_pvalue.png         — p-values + tests Bonferroni
    dinov2_gamma_sweep.csv       — tableau complet

Usage :
    .venv/bin/python experiments/gamma_sweep_dinov2.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, t as t_dist

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

CACHE_DIR    = BASE_DIR / "results" / "sweep_coco_degraded" / "cache"
COCO_DEG_DIR = BASE_DIR / "dataset" / "coco_degraded"
OUT_DIR      = BASE_DIR / "results" / "sweep_coco_degraded" / "plots"
N_REFS       = 200
DIST_TYPES   = [1, 10, 11, 21, 25]

LAYER_NAMES = {
    0:  "patch_embed",
    1:  "blocks.0",
    2:  "blocks.1",
    3:  "blocks.2",
    4:  "blocks.3",
    5:  "blocks.4",
    6:  "blocks.5",
    7:  "blocks.6",
    8:  "blocks.7",
    9:  "blocks.8",
    10: "blocks.9",
    11: "blocks.10",
    12: "blocks.11",
    13: "norm",
}


# ===========================================================================
# Chargement métadonnées (identique à gamma_sweep_sdvae.py)
# ===========================================================================

def load_levels():
    meta_path = COCO_DEG_DIR / "metadata.csv"
    dist_filter = set(DIST_TYPES)
    sorted_types = sorted(dist_filter)
    type_to_rank = {t: i for i, t in enumerate(sorted_types)}
    n_types = len(sorted_types)

    rows = []
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    all_refs, seen = [], set()
    for r in rows:
        if r["ref_img"] not in seen:
            all_refs.append(r["ref_img"])
            seen.add(r["ref_img"])

    selected_refs = set(all_refs[:N_REFS])
    ref_to_idx = {r: i for i, r in enumerate(all_refs[:N_REFS])}

    levels, seq_ids = [], []
    for r in rows:
        if r["ref_img"] not in selected_refs:
            continue
        dist_type = int(r["dist_type"])
        if dist_type not in dist_filter:
            continue
        ref_idx = ref_to_idx[r["ref_img"]]
        level = int(r["level"])
        seq_id = ref_idx * n_types + type_to_rank[dist_type]
        levels.append(level)
        seq_ids.append(seq_id)

    return np.array(levels, dtype=np.int32), np.array(seq_ids, dtype=np.int32)


# ===========================================================================
# Spearman rapide
# ===========================================================================

def spearman_pval(rho: float, n: int) -> float:
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 else np.nan
    t_stat = rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2)
    return float(2 * t_dist.sf(abs(t_stat), df=n - 2))


def fast_spearman_batch(scores_batch, levels_rank_c, levels_rank_std):
    B, N = scores_batch.shape
    rhos = np.empty(B)
    pvals = np.empty(B)
    for i in range(B):
        r = rankdata(scores_batch[i]).astype(np.float64)
        r -= r.mean()
        r_std = np.sqrt((r ** 2).sum())
        if r_std < 1e-12:
            rhos[i] = np.nan
            pvals[i] = np.nan
        else:
            rho = float((r * levels_rank_c).sum() / (r_std * levels_rank_std))
            rhos[i] = rho
            pvals[i] = spearman_pval(rho, N)
    return rhos, pvals


# ===========================================================================
# Structure séquences
# ===========================================================================

def _precompute_seq_structure(levels, seq_ids):
    unique_seqs = np.unique(seq_ids)
    seq_len = None
    rows = []
    for sid in unique_seqs:
        mask = (seq_ids == sid)
        lv = levels[mask]
        idx = np.where(mask)[0]
        order = np.argsort(lv)
        rows.append(idx[order])
        if seq_len is None:
            seq_len = len(order)
    rows = [r for r in rows if len(r) == seq_len]
    sorted_idx = np.array(rows, dtype=np.intp)
    return sorted_idx, len(rows), seq_len


def _strict_mono_and_tau_batch(scores_batch, sorted_idx):
    B, _ = scores_batch.shape
    n_seqs, seq_len = sorted_idx.shape
    C = seq_len * (seq_len - 1) // 2

    scores_sorted = scores_batch[:, sorted_idx]
    diffs = np.diff(scores_sorted, axis=2)
    strict = np.all(diffs > 0, axis=2)
    strict_rates = strict.mean(axis=1).astype(np.float64)

    tau_vals = np.zeros(B, dtype=np.float64)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            conc = scores_sorted[:, :, i] < scores_sorted[:, :, j]
            disc = scores_sorted[:, :, i] > scores_sorted[:, :, j]
            tau_vals += (conc.sum(axis=1) - disc.sum(axis=1))
    tau_vals /= (n_seqs * C)

    return strict_rates, tau_vals


# ===========================================================================
# Sweep principal
# ===========================================================================

def sweep_gammas(layers, gammas, levels, seq_ids, batch_size=100):
    N = len(levels)
    lr_c = rankdata(levels).astype(np.float64)
    lr_c -= lr_c.mean()
    lr_std = np.sqrt((lr_c ** 2).sum())

    print("  Pré-calcul structure séquences…", flush=True)
    sorted_idx, n_seqs, seq_len = _precompute_seq_structure(levels, seq_ids)
    print(f"  {n_seqs} séquences × {seq_len} niveaux")

    results = {}
    G = len(gammas)

    for l in layers:
        drr_path = CACHE_DIR / f"dinov2_vitb14_l{l:02d}_Drr_ref200_std.npy"
        drq_path = CACHE_DIR / f"dinov2_vitb14_l{l:02d}_Drq_r200_t1-10-11-21-25_std.npy"

        if not drr_path.exists() or not drq_path.exists():
            print(f"  [skip] layer {l} — cache manquant")
            continue

        D_rr = np.load(drr_path).astype(np.float64)
        D_rq = np.load(drq_path).astype(np.float64)
        print(f"  layer {l:02d}  D_rr{D_rr.shape}  D_rq{D_rq.shape}", flush=True)

        all_rhos   = np.empty(G)
        all_pvals  = np.empty(G)
        all_strict = np.empty(G)
        all_tau    = np.empty(G)

        for start in range(0, G, batch_size):
            end = min(start + batch_size, G)
            g_batch = gammas[start:end]
            B = len(g_batch)

            K_rr_mean = np.array([np.exp(-g * D_rr).mean() for g in g_batch])

            exp_neg = np.exp(-g_batch[:, None, None] * D_rq[None])  # (B, n_r, n_q)
            K_rq_col = exp_neg.mean(axis=1)                          # (B, n_q)
            scores_batch = 1.0 + K_rr_mean[:, None] - 2.0 * K_rq_col

            rhos_b, pvals_b = fast_spearman_batch(scores_batch, lr_c, lr_std)
            all_rhos[start:end]  = rhos_b
            all_pvals[start:end] = pvals_b

            strict_b, tau_b = _strict_mono_and_tau_batch(scores_batch, sorted_idx)
            all_strict[start:end] = strict_b
            all_tau[start:end]    = tau_b

        results[l] = {
            "rho":    all_rhos,
            "pval":   all_pvals,
            "strict": all_strict,
            "tau":    all_tau,
        }

    return results


# ===========================================================================
# Figure 1 — Courbes ρ(γ) par layer
# ===========================================================================

def plot_rho_vs_gamma(results, gammas, out_dir):
    layers = sorted(results.keys())
    n_layers = len(layers)
    cmap = plt.get_cmap("plasma", n_layers)

    median_gammas = {}
    for l in layers:
        drr_path = CACHE_DIR / f"dinov2_vitb14_l{l:02d}_Drr_ref200_std.npy"
        if drr_path.exists():
            D_rr = np.load(drr_path).astype(np.float64)
            nonzero = D_rr[D_rr > 0]
            if len(nonzero) > 0:
                median_gammas[l] = float(1.0 / (2.0 * np.median(nonzero)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f9f9f9")

    best_per_layer = {}
    for idx, l in enumerate(layers):
        rhos = results[l]["rho"]
        valid = np.isfinite(rhos)
        if not valid.any():
            continue
        ax.plot(gammas[valid], rhos[valid],
                color=cmap(idx), linewidth=1.4, alpha=0.85,
                label=f"L{l:02d} {LAYER_NAMES.get(l, '')}")
        best_idx = int(np.nanargmax(rhos))
        best_per_layer[l] = (gammas[best_idx], float(rhos[best_idx]))
        ax.scatter([gammas[best_idx]], [rhos[best_idx]],
                   color=cmap(idx), s=40, zorder=5,
                   edgecolors="black", linewidths=0.4)

        if l in median_gammas:
            gi = int(np.argmin(np.abs(gammas - median_gammas[l])))
            rho_med = float(rhos[gi]) if np.isfinite(rhos[gi]) else np.nan
            if np.isfinite(rho_med):
                ax.scatter([gammas[gi]], [rho_med],
                           color=cmap(idx), s=70, zorder=6,
                           marker="^", edgecolors="black", linewidths=0.6)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("γ (log scale)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "DINOv2 ViT-B/14 — Spearman ρ as a function of γ  (1 000 values, log grid)\n"
        "COCO-degraded  ·  anchor = 200 clean refs  ·  standardisation ON",
        fontsize=13, fontweight="bold",
    )

    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=6, markeredgecolor="black", markeredgewidth=0.4,
               label="Optimal γ (best ρ)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=7, markeredgecolor="black", markeredgewidth=0.6,
               label="γ = 1/(2·median(D_rr))"),
    ]
    layer_legend = ax.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.85)
    ax.add_artist(layer_legend)
    ax.legend(handles=extra_handles, fontsize=8, loc="upper left",
              framealpha=0.9, bbox_to_anchor=(0.0, 1.0))

    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    out_path = out_dir / "08_dinov2_rho_vs_gamma.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return best_per_layer


# ===========================================================================
# Figure 2 — Heatmap
# ===========================================================================

def plot_heatmap(results, gammas, out_dir):
    layers = sorted(results.keys())
    n_layers = len(layers)
    G = len(gammas)

    n_display = 100
    step = max(1, G // n_display)
    g_idx = np.arange(0, G, step)
    g_display = gammas[g_idx]

    matrix = np.full((n_layers, len(g_idx)), np.nan)
    for i, l in enumerate(layers):
        matrix[i] = results[l]["rho"][g_idx]

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-0.1, vmax=1.0, origin="upper")
    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.02, pad=0.02)

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l:02d} {LAYER_NAMES.get(l, '')}" for l in layers], fontsize=8)

    x_step = max(1, len(g_idx) // 10)
    ax.set_xticks(range(0, len(g_idx), x_step))
    ax.set_xticklabels(
        [f"{g_display[j]:.1e}" for j in range(0, len(g_idx), x_step)],
        rotation=45, ha="right", fontsize=7.5,
    )
    ax.set_xlabel("γ", fontsize=11)
    ax.set_title(
        "DINOv2 ViT-B/14 — Heatmap ρ(layer × γ)  ·  1 000 gammas\n"
        "COCO-degraded  ·  anchor = 200 clean refs  ·  standardisation ON",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "09_dinov2_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure 3 — p-values
# ===========================================================================

def plot_pvalues(results, gammas, out_dir, n_tests_total):
    layers = sorted(results.keys())
    n_layers = len(layers)
    cmap = plt.get_cmap("plasma", n_layers)
    N = 5000

    fig = plt.figure(figsize=(20, 18))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.38)
    ax_curve = fig.add_subplot(gs[0])
    ax_tbl   = fig.add_subplot(gs[1])

    ax_curve.set_facecolor("#f9f9f9")
    for idx, l in enumerate(layers):
        pvals = results[l]["pval"]
        rhos  = results[l]["rho"]
        valid = np.isfinite(pvals) & (pvals > 0)
        if not valid.any():
            continue
        logp = -np.log10(np.clip(pvals[valid], 1e-300, 1.0))
        ax_curve.plot(gammas[valid], logp,
                      color=cmap(idx), linewidth=1.8, alpha=0.85,
                      label=f"L{l:02d} {LAYER_NAMES.get(l, '')}")
        best_idx = int(np.nanargmax(rhos))
        if valid[best_idx]:
            lp_best = float(-np.log10(max(float(pvals[best_idx]), 1e-300)))
            ax_curve.scatter([gammas[best_idx]], [lp_best],
                             color=cmap(idx), s=55, zorder=6,
                             edgecolors="black", linewidths=0.5)

    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("γ  (log scale)", fontsize=13)
    ax_curve.set_ylabel("−log₁₀(p-value)", fontsize=13)
    ax_curve.set_title(
        "−log₁₀(p-value) of Spearman ρ  vs  γ  —  all layers\n"
        "Dots = optimal γ (best ρ) per layer",
        fontsize=12, fontweight="bold",
    )
    ax_curve.legend(fontsize=9, ncol=2, loc="lower left", framealpha=0.9)
    ax_curve.grid(True, which="both", alpha=0.25)
    ax_curve.tick_params(axis="both", labelsize=11)

    ax_tbl.axis("off")
    table_data = []
    for l in layers:
        rhos_l  = results[l]["rho"]
        pvals_l = results[l]["pval"]
        best_idx   = int(np.nanargmax(rhos_l))
        rho_best   = float(rhos_l[best_idx])
        pval_best  = float(pvals_l[best_idx])
        gamma_best = float(gammas[best_idx])

        if np.isfinite(rho_best) and abs(rho_best) < 1.0:
            t_stat = rho_best * np.sqrt(N - 2) / np.sqrt(1 - rho_best**2)
        else:
            t_stat = np.inf

        pval_str = f"{pval_best:.2e}" if np.isfinite(pval_best) and pval_best > 0 else "< 1e-300"
        table_data.append([
            f"L{l:02d}", LAYER_NAMES.get(l, ""),
            f"{gamma_best:.3e}", f"{rho_best:+.4f}",
            f"{t_stat:.1f}" if np.isfinite(t_stat) else "∞",
            pval_str,
        ])

    cols = ["Layer", "Layer name", "Optimal γ", "ρ max", "t-stat", "p-value"]
    tbl = ax_tbl.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(range(len(cols)))
    for (row, col), cell in tbl.get_celld().items():
        cell.set_height(0.052)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)

    ax_tbl.set_title(
        f"Spearman significance tests — best γ per layer\n"
        f"N = {N:,} images  (200 refs × 5 distortion types × 5 severity levels)",
        fontsize=12, fontweight="bold", pad=16,
    )
    fig.suptitle(
        "DINOv2 ViT-B/14 — Statistical Significance of Spearman ρ  (1 000-gamma sweep)\n"
        "COCO-degraded dataset  ·  anchor = 200 clean refs  ·  standardisation ON",
        fontsize=12, fontweight="bold", y=1.02,
    )

    out_path = out_dir / "10_dinov2_pvalue.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Export CSV
# ===========================================================================

def export_csv(results, gammas, out_dir):
    rows = []
    for l in sorted(results.keys()):
        r = results[l]
        for gi in range(len(gammas)):
            def fmt(v):
                return float(v) if np.isfinite(v) else ""
            rows.append({
                "layer":        l,
                "layer_name":   LAYER_NAMES.get(l, ""),
                "gamma":        float(gammas[gi]),
                "spearman_rho": fmt(r["rho"][gi]),
                "pval":         fmt(r["pval"][gi]),
                "strict_mono":  fmt(r["strict"][gi]),
                "kendall_tau":  fmt(r["tau"][gi]),
            })
    out_path = out_dir / "dinov2_gamma_sweep.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    import time
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gammas = np.logspace(-9, -1, 1000)
    layers = list(range(14))

    print("Chargement métadonnées…")
    levels, seq_ids = load_levels()
    print(f"  {len(levels)} images  ·  {len(np.unique(seq_ids))} séquences")

    print(f"\nSweep 1 000 gammas × {len(layers)} layers (vectorisé, batch=50)…")
    t0 = time.time()
    results = sweep_gammas(layers, gammas, levels, seq_ids, batch_size=50)
    print(f"  Terminé en {time.time() - t0:.1f}s")

    n_valid = len(results)
    n_tests = n_valid * len(gammas)
    print(f"  {n_valid} layers valides · {n_tests:,} tests au total")

    print("\n--- Figure 1 : ρ(γ) ---")
    best_per_layer = plot_rho_vs_gamma(results, gammas, OUT_DIR)

    print("\n--- Figure 2 : Heatmap ---")
    plot_heatmap(results, gammas, OUT_DIR)

    print("\n--- Figure 3 : p-values ---")
    plot_pvalues(results, gammas, OUT_DIR, n_tests_total=n_tests)

    print("\n--- Export CSV ---")
    export_csv(results, gammas, OUT_DIR)

    print("\n" + "=" * 68)
    print(f"  {'Layer':<6} {'Nom':<18} {'γ optimal':>10}  {'ρ max':>8}  {'p-value':>12}")
    print("  " + "-" * 62)
    for l, (g_opt, rho_opt) in sorted(best_per_layer.items(), key=lambda x: -x[1][1]):
        pv = float(results[l]["pval"][int(np.nanargmax(results[l]["rho"]))])
        pv_str = f"{pv:.2e}" if np.isfinite(pv) and pv > 0 else "~0"
        print(f"  L{l:02d}   {LAYER_NAMES.get(l, ''):<18} {g_opt:>10.2e}  {rho_opt:>+8.4f}  {pv_str:>12}")
    print("=" * 68)


if __name__ == "__main__":
    main()
