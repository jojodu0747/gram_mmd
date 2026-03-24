#!/usr/bin/env python3
"""Plot results from sweep summary_all.csv"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path("/home/jnapolitano/Documents/metrique/results/synthetic_kadid_sweep/summary_all.csv")
OUT_DIR  = Path("/home/jnapolitano/Documents/metrique/results/synthetic_kadid_sweep/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_ORDER = ["sd_vae", "dc_ae", "lpips_vgg", "vgg19", "dinov2_vitb14"]
BACKBONE_LABELS = {"sd_vae":"SD-VAE","dc_ae":"DC-AE","lpips_vgg":"LPIPS-VGG","vgg19":"VGG-19","dinov2_vitb14":"DINOv2-B"}
BACKBONE_COLORS = {"sd_vae":"#1976D2","dc_ae":"#388E3C","lpips_vgg":"#9C27B0","vgg19":"#F57C00","dinov2_vitb14":"#E91E63"}

def load():
    rows=[]
    with open(CSV_PATH,newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"backbone":r["backbone"],"layer":int(r["layer"]),"gamma_mult":float(r["gamma_mult"]),
                         "spearman":float(r["spearman"]),"kendall":float(r["kendall"]),"mono_strict":float(r["mono_strict"])})
    return rows

def best_per_layer(rows, bb, metric):
    layers = sorted({r["layer"] for r in rows if r["backbone"]==bb})
    vals=[]
    for l in layers:
        lr=[r for r in rows if r["backbone"]==bb and r["layer"]==l]
        vals.append((l, max(abs(r[metric]) for r in lr)))
    return vals

# ── Plot 1: layer profiles (3 metrics) ───────────────────────────────────
def plot_profiles(rows):
    metrics=[("spearman","|Spearman ρ|"),("kendall","|Kendall τ|"),("mono_strict","Strict mono")]
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,(met,mlabel) in zip(axes,metrics):
        for bb in BACKBONE_ORDER:
            pts=best_per_layer(rows,bb,met)
            if pts:
                ls,vs=zip(*pts)
                ax.plot(ls,vs,marker="o",ms=4,lw=1.8,color=BACKBONE_COLORS[bb],label=BACKBONE_LABELS[bb])
        ax.set_xlabel("Layer index",fontsize=10)
        ax.set_ylabel(mlabel,fontsize=10)
        ax.set_title(mlabel,fontsize=10,fontweight="bold")
        ax.grid(True,alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Synthetic KADID — metrics vs layer (best γ)\n20 COCO refs · 1000 anchor · 125 KADID distortions",fontsize=11,fontweight="bold")
    plt.tight_layout()
    out=OUT_DIR/"01_layer_profiles.png"
    fig.savefig(out,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── Plot 2: gamma sensitivity (best layer per backbone) ───────────────────
def plot_gamma(rows):
    metrics=[("spearman","|Spearman ρ|"),("kendall","|Kendall τ|"),("mono_strict","Strict mono")]
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,(met,mlabel) in zip(axes,metrics):
        for bb in BACKBONE_ORDER:
            bb_rows=[r for r in rows if r["backbone"]==bb]
            layers=sorted({r["layer"] for r in bb_rows})
            best_l=max(layers,key=lambda l:max(abs(r[met]) for r in bb_rows if r["layer"]==l))
            pts=sorted([(r["gamma_mult"],abs(r[met])) for r in bb_rows if r["layer"]==best_l])
            gm,vs=zip(*pts)
            ax.semilogx(gm,vs,marker="o",ms=4,lw=1.8,color=BACKBONE_COLORS[bb],label=f"{BACKBONE_LABELS[bb]} L{best_l}")
        ax.set_xlabel("γ multiplier",fontsize=10)
        ax.set_ylabel(mlabel,fontsize=10)
        ax.set_title(mlabel,fontsize=10,fontweight="bold")
        ax.grid(True,alpha=0.25,which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Synthetic KADID — metrics vs γ (best layer per backbone)\n20 COCO refs · 1000 anchor",fontsize=11,fontweight="bold")
    plt.tight_layout()
    out=OUT_DIR/"02_gamma_sensitivity.png"
    fig.savefig(out,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── Plot 3: heatmap spearman ──────────────────────────────────────────────
def plot_heatmap(rows, metric, mlabel, fname, vmin=None, vmax=None):
    layers_per_bb={bb:sorted({r["layer"] for r in rows if r["backbone"]==bb}) for bb in BACKBONE_ORDER}
    max_l=max(len(v) for v in layers_per_bb.values())
    mat=np.full((len(BACKBONE_ORDER),max_l),np.nan)
    best_g=np.full((len(BACKBONE_ORDER),max_l),np.nan)
    for i,bb in enumerate(BACKBONE_ORDER):
        for j,l in enumerate(layers_per_bb[bb]):
            lr=[r for r in rows if r["backbone"]==bb and r["layer"]==l]
            best=max(lr,key=lambda r:abs(r[metric]))
            mat[i,j]=abs(best[metric])
            best_g[i,j]=best["gamma_mult"]
    fig,ax=plt.subplots(figsize=(max(10,max_l*0.6),4.5))
    im=ax.imshow(mat,aspect="auto",cmap="viridis",
                 vmin=vmin if vmin is not None else np.nanmin(mat),
                 vmax=vmax if vmax is not None else np.nanmax(mat))
    plt.colorbar(im,ax=ax,label=mlabel,fraction=0.03,pad=0.02)
    vrange=np.nanmax(mat)-np.nanmin(mat)
    thresh=np.nanmin(mat)+vrange*0.55
    for i in range(len(BACKBONE_ORDER)):
        for j in range(max_l):
            if not np.isnan(mat[i,j]):
                c="white" if mat[i,j]<thresh else "black"
                ax.text(j,i,f"{mat[i,j]:.2f}\n×{best_g[i,j]:.2g}",ha="center",va="center",fontsize=5,color=c)
    ax.set_xticks(range(max_l))
    ax.set_xticklabels([str(j) for j in range(max_l)],fontsize=7)
    ax.set_yticks(range(len(BACKBONE_ORDER)))
    ax.set_yticklabels([BACKBONE_LABELS[bb] for bb in BACKBONE_ORDER],fontsize=9)
    ax.set_xlabel("Layer index",fontsize=10)
    ax.set_title(f"Synthetic KADID — {mlabel} (best γ per cell)\n20 COCO refs · 1000 anchor",fontsize=10,fontweight="bold")
    plt.tight_layout()
    out=OUT_DIR/fname
    fig.savefig(out,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── Plot 4: scatter |ρ| vs |τ|, all (bb,layer) at best gamma + star best ──
def plot_scatter(rows):
    fig,ax=plt.subplots(figsize=(8,7))
    ax.set_facecolor("#f9f9f9")
    for bb in BACKBONE_ORDER:
        bb_rows=[r for r in rows if r["backbone"]==bb]
        layers=sorted({r["layer"] for r in bb_rows})
        sps,kts=[],[]
        for l in layers:
            lr=[r for r in bb_rows if r["layer"]==l]
            best=max(lr,key=lambda r:abs(r["spearman"]))
            sps.append(abs(best["spearman"]))
            kts.append(abs(best["kendall"]))
        ax.scatter(kts,sps,color=BACKBONE_COLORS[bb],s=35,alpha=0.6,edgecolors="none",zorder=3)
        # best overall
        bo=max(bb_rows,key=lambda r:abs(r["spearman"]))
        ax.scatter(abs(bo["kendall"]),abs(bo["spearman"]),color=BACKBONE_COLORS[bb],s=200,marker="*",
                   edgecolors="black",lw=0.7,zorder=6,label=f"{BACKBONE_LABELS[bb]} L{bo['layer']} ×{bo['gamma_mult']:.2g}")
    ax.set_xlabel("|Kendall τ|",fontsize=12)
    ax.set_ylabel("|Spearman ρ|",fontsize=12)
    ax.set_title("Synthetic KADID — |ρ| vs |τ|\n★ best config per backbone · dots = all layers (best γ)",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8,loc="lower right")
    ax.grid(True,alpha=0.25)
    plt.tight_layout()
    out=OUT_DIR/"04_scatter.png"
    fig.savefig(out,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

rows=load()
print(f"Loaded {len(rows)} rows")
plot_profiles(rows)
plot_gamma(rows)
plot_heatmap(rows,"spearman","|Spearman ρ|","03_heatmap_spearman.png")
plot_heatmap(rows,"kendall","|Kendall τ|","03b_heatmap_kendall.png")
plot_heatmap(rows,"mono_strict","Strict mono","03c_heatmap_mono.png",vmin=0.4,vmax=0.65)
plot_scatter(rows)
print("All done.")
