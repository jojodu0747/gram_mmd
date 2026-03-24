"""Regenerate scatter plot from existing CSV with GMMD prefix and legend at bottom right."""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATH = Path("/home/jnapolitano/Documents/metrique/results/kadid_coco_distdist/summary_20260313_123143.csv")
OUT_DIR  = Path("/home/jnapolitano/Documents/metrique/results/kadid_coco_distdist/plots")
CMMD_GAMMA = 0.005

BACKBONE_COLORS = {
    "SD-VAE L11": "#1976D2",
    "DC-AE L8":   "#388E3C",
    "DINOv2 L3":  "#F57C00",
    "CMMD":       "#E91E63",
}
GAMMA_MARKERS = {
    "div10":  "v",
    "median": "o",
    "10x":    "s",
    "fixed":  "D",
}
GAMMA_LABELS = {
    "div10":  "γ/10",
    "median": "γ_med",
    "10x":    "10×γ_med",
    "fixed":  f"CMMD γ={CMMD_GAMMA}",
}

def load_csv():
    results = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bname = row["backbone"]
            g_tag = row["gamma_tag"]
            # Build label with GMMD prefix for non-CMMD entries
            if bname == "CMMD":
                label = f"CMMD γ={CMMD_GAMMA}"
            else:
                label = f"GMMD {bname} {GAMMA_LABELS[g_tag]}"
            results.append({
                "label":         label,
                "backbone_name": bname,
                "gamma_tag":     g_tag,
                "stats": {
                    "spearman":    float(row["spearman"]),
                    "kendall":     float(row["kendall"]),
                    "mono_strict": float(row["mono_strict"]),
                },
            })
    return results

ARROW_PROPS = dict(arrowstyle="-", color="gray", lw=0.7)

# Per-label annotation: (dx_pts, dy_pts, ha, use_arrow)
# Cluster labels use arrows so text can be placed far from the crowded area
LABEL_OFFSETS = {
    # isolated — simple offset, no arrow
    "GMMD SD-VAE L11 γ/10":     ( 8,   4,  "left",   False),
    "GMMD SD-VAE L11 γ_med":    ( 8,   4,  "left",   False),
    "CMMD γ=0.005":              ( 8,   4,  "left",   False),
    # cluster: spread with arrows
    # DC-AE γ_med is rightmost → right, no arrow needed
    "GMMD DC-AE L8 γ_med":       ( 8,   4,  "left",   False),
    # SD-VAE 10x — send label to the left-middle (avoid top cutoff)
    "GMMD SD-VAE L11 10×γ_med": (-85,  0,  "right",  True),
    # DC-AE 10x — right, slightly below DC-AE γ_med
    "GMMD DC-AE L8 10×γ_med":   ( 50,  15, "left",   True),
    # DINOv2 γ_med — left-mid
    "GMMD DINOv2 L3 γ_med":     (-85,  10, "right",  True),
    # DC-AE γ/10 — right of cluster
    "GMMD DC-AE L8 γ/10":       ( 50,  10, "left",   True),
    # DINOv2 10×γ_med — lower right
    "GMMD DINOv2 L3 10×γ_med":  ( 50, -25, "left",   True),
    # DINOv2 γ/10 — leftmost, send left-low
    "GMMD DINOv2 L3 γ/10":      (-85, -15, "right",  True),
}

def plot_scatter(all_results):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#f9f9f9")

    for entry in all_results:
        bname  = entry["backbone_name"]
        g_tag  = entry["gamma_tag"]
        sp     = abs(entry["stats"]["spearman"])
        kt     = abs(entry["stats"]["kendall"])
        color  = BACKBONE_COLORS.get(bname, "#9C27B0")
        marker = GAMMA_MARKERS.get(g_tag, "o")
        ax.scatter(kt, sp, color=color, marker=marker, s=110,
                   edgecolors="black", linewidths=0.7, zorder=5)
        dx, dy, ha, arrow = LABEL_OFFSETS.get(entry["label"], (8, 4, "left", False))
        ap = ARROW_PROPS if arrow else None
        ax.annotate(entry["label"], (kt, sp),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.5, ha=ha, zorder=6,
                    arrowprops=ap)

    color_handles = [
        Line2D([0],[0], marker="o", color=c, linestyle="None", markersize=8,
               markeredgecolor="black", markeredgewidth=0.6, label=n)
        for n, c in BACKBONE_COLORS.items()
        if any(e["backbone_name"] == n for e in all_results)
    ]
    marker_handles = [
        Line2D([0],[0], marker=m, color="gray", linestyle="None", markersize=8,
               markeredgecolor="black", markeredgewidth=0.5, label=GAMMA_LABELS[t])
        for t, m in GAMMA_MARKERS.items()
        if any(e["gamma_tag"] == t for e in all_results)
    ]
    # γ legend: bottom right ; Backbone legend: upper left
    leg_gamma = ax.legend(handles=marker_handles, fontsize=8, loc="lower right",
                          title="γ", framealpha=0.9)
    ax.add_artist(leg_gamma)
    ax.legend(handles=color_handles, fontsize=8, loc="upper left",
              title="Backbone", framealpha=0.9)

    ax.set_xlabel("|Kendall τ|", fontsize=12)
    ax.set_ylabel("|Spearman ρ|", fontsize=12)
    ax.set_title(
        "KADID-10k  ·  1000 COCO anchor  ·  dist-vs-dist MMD²\n"
        "|Spearman ρ| vs |Kendall τ|",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = OUT_DIR / "01_scatter_spearman_vs_kendall_tau.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_csv()
    plot_scatter(results)
