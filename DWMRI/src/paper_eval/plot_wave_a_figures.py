#!/usr/bin/env python3
"""Wave A matplotlib figures: sigma, K-sweep, Pareto, RGS-vs-sequential."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Authoritative paper table values (D-Brain).
SIGMA = np.array([0.05, 0.10, 0.15, 0.20])
SIGMA_PSNR = {
    "MP-PCA": [33.33, 28.44, 25.54, 23.55],
    "Patch2Self": [23.74, 21.31, 21.48, 19.10],
    "MD-S2S": [16.77, 15.28, 15.10, 14.57],
    "DRCNet-Hybrid-RGS": [29.76, 26.93, 25.11, 23.52],
    "Restormer-Hybrid-RGS": [27.19, 23.43, 24.63, 23.02],
}
# FA-MAE where reported in tab:sigma_sweep (baselines omitted in table).
SIGMA_FA = {
    "MP-PCA": [0.1009, 0.0898, 0.0978, 0.0863],
    "DRCNet-Hybrid-RGS": [0.2579, 0.2575, 0.2202, 0.2211],
    "Restormer-Hybrid-RGS": [0.2170, 0.2238, 0.2452, 0.2597],
}
# From staged dti_metrics for P2S/MDS2S (optional companion panel).
SIGMA_FA_BASELINES = {
    "Patch2Self": [0.2363, 0.2393, 0.2411, 0.2424],
    "MD-S2S": [0.2309, 0.2298, 0.2391, 0.2443],
}

# K-sweep PSNR-ROI / FA-MAE from manuscript K-sweep (σ=0.1).
K = np.array([5, 10, 16, 24, 30])
K_PSNR = {
    "DRCNet-Hybrid-RGS": [26.46, 26.46, 26.93, 26.92, 26.36],
    "Restormer-Hybrid-RGS": [25.88, 26.60, 23.43, 25.83, 25.50],
}
K_FA = {
    "DRCNet-Hybrid-RGS": [0.2319, 0.2378, 0.2575, 0.2559, 0.2464],
    "Restormer-Hybrid-RGS": [0.2489, 0.2195, 0.2238, 0.2198, 0.2199],
}

# Accuracy–cost Pareto (D-Brain K=16 σ=0.1 primary + 2D/large from tab:3d_vs_2d).
PARETO = [
    # label, psnr_roi, sec_per_vol, n_params_m, family
    ("MP-PCA", 28.44, 1.0, None, "classical"),
    ("Patch2Self", 21.31, 45.0, None, "classical"),
    ("MD-S2S", 15.28, 120.0, 0.5, "classical"),
    ("DRCNet-Hybrid-RGS", 26.93, 34.1, 0.116, "hybrid3d"),
    ("Restormer-Hybrid-RGS", 23.43, 126.8, 0.178, "hybrid3d"),
    ("Plain-CNN-2D", 27.11, 4.7, 0.014, "hybrid2d"),
    ("Res-CNN-2D", 26.44, 3.9, 0.015, "hybrid2d"),
    ("Restormer-2D", 25.25, 26.7, 0.149, "hybrid2d"),
    ("Restormer3D-large", 25.94, 362.4, 2.10, "hybrid3d"),
]

COLORS = {
    "MP-PCA": "#1f77b4",
    "Patch2Self": "#ff7f0e",
    "MD-S2S": "#2ca02c",
    "DRCNet-Hybrid-RGS": "#d62728",
    "Restormer-Hybrid-RGS": "#9467bd",
}
MARKERS = {"MP-PCA": "o", "Patch2Self": "s", "MD-S2S": "^", "DRCNet-Hybrid-RGS": "D", "Restormer-Hybrid-RGS": "v"}


def plot_sigma() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=200)
    ax = axes[0]
    for name, ys in SIGMA_PSNR.items():
        ax.plot(SIGMA, ys, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=1.6)
    ax.set_xlabel(r"Rician noise level $\sigma$")
    ax.set_ylabel("PSNR-ROI (dB)")
    ax.set_xticks(SIGMA)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Image-domain robustness")

    ax = axes[1]
    for name, ys in {**SIGMA_FA, **SIGMA_FA_BASELINES}.items():
        ax.plot(
            SIGMA,
            ys,
            marker=MARKERS.get(name, "o"),
            color=COLORS.get(name, "#333333"),
            label=name,
            linewidth=1.6,
        )
    ax.set_xlabel(r"Rician noise level $\sigma$")
    ax.set_ylabel("FA-MAE")
    ax.set_xticks(SIGMA)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Tensor-domain robustness")
    fig.tight_layout()
    # Keep legacy single-panel filename as PSNR-only for existing \includegraphics,
    # plus a two-panel asset used if tex is updated.
    fig.savefig(OUT / "sigma_robustness_two_panel.png", dpi=300, bbox_inches="tight", facecolor="white")

    # Legacy single-panel PSNR (all five methods, four points).
    fig2, ax2 = plt.subplots(figsize=(6.2, 3.8), dpi=200)
    for name, ys in SIGMA_PSNR.items():
        ax2.plot(SIGMA, ys, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=1.8)
    ax2.set_xlabel(r"Rician noise level $\sigma$")
    ax2.set_ylabel("PSNR-ROI (dB)")
    ax2.set_xticks(SIGMA)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "sigma_robustness_psnr_roi.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.close(fig2)


def plot_k_sweep() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=200)
    for name in K_PSNR:
        axes[0].plot(K, K_PSNR[name], marker="o", color=COLORS[name], label=name, linewidth=1.8)
        axes[1].plot(K, K_FA[name], marker="o", color=COLORS[name], label=name, linewidth=1.8)
    axes[0].set_xlabel(r"Input subset size $K$")
    axes[0].set_ylabel("PSNR-ROI (dB)")
    axes[0].set_xticks(K)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Image-domain vs $K$")
    axes[1].set_xlabel(r"Input subset size $K$")
    axes[1].set_ylabel("FA-MAE")
    axes[1].set_xticks(K)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Tensor-domain vs $K$")
    fig.tight_layout()
    fig.savefig(OUT / "k_sweep_two_panel.png", dpi=300, bbox_inches="tight", facecolor="white")
    # Also refresh single-panel PSNR legacy asset.
    fig2, ax2 = plt.subplots(figsize=(5.8, 3.6), dpi=200)
    for name in K_PSNR:
        ax2.plot(K, K_PSNR[name], marker="o", color=COLORS[name], label=name, linewidth=1.8)
    ax2.set_xlabel(r"Input subset size $K$")
    ax2.set_ylabel("PSNR-ROI (dB)")
    ax2.set_xticks(K)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "k_sweep_psnr_roi.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.close(fig2)


def plot_pareto() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fam_style = {
        "classical": dict(marker="s", color="#4c78a8"),
        "hybrid3d": dict(marker="o", color="#e45756"),
        "hybrid2d": dict(marker="D", color="#54a24b"),
    }
    for label, psnr, sec, nparams, fam in PARETO:
        st = fam_style[fam]
        ax.scatter(sec, psnr, s=70 if nparams is None else 40 + 40 * float(nparams), **st, zorder=3)
        ax.annotate(label, (sec, psnr), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per volume (s, log scale)")
    ax.set_ylabel("PSNR-ROI (dB)")
    ax.grid(True, which="both", alpha=0.3)
    # Legend proxies
    for fam, st in fam_style.items():
        ax.scatter([], [], label={"classical": "Classical / SS baselines", "hybrid3d": "Hybrid RGS 3D", "hybrid2d": "Hybrid RGS 2D"}[fam], **st)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Accuracy–cost Pareto (D-Brain, $K=16$, $\\sigma=0.1$)")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_cost_pareto.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rgs_vs_sequential() -> None:
    """Schematic: sequential windows leave gaps; RGS covers all targets over training."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=200)
    g = 16  # illustrative shell size
    k = 5
    rng = np.random.default_rng(0)

    # Sequential: sliding windows of K consecutive indices
    ax = axes[0]
    ax.set_title("Sequential $K$-windows")
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_yticks([])
    ax.set_xlabel("Gradient index")
    coverage = np.zeros(g)
    for row, start in enumerate(range(0, g - k + 1, 2)):
        xs = list(range(start, start + k))
        coverage[xs] += 1
        target = start + k - 1
        for x in xs:
            ax.add_patch(plt.Rectangle((x - 0.4, row), 0.8, 0.7, facecolor="#a6cee3", edgecolor="#333"))
        ax.add_patch(plt.Rectangle((target - 0.4, row), 0.8, 0.7, facecolor="#e31a1c", edgecolor="#333"))
    # under-covered ends
    for i, c in enumerate(coverage):
        if c == 0:
            ax.plot(i, -0.25, "x", color="black", markersize=8)
    ax.text(0.02, 0.95, "Red = target slot\n× = never a target", transform=ax.transAxes, va="top", fontsize=7)

    # RGS: random subsets; each index is target equally often over many draws
    ax = axes[1]
    ax.set_title("Random gradient subsets (RGS)")
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_yticks([])
    ax.set_xlabel("Gradient index")
    counts = np.zeros(g)
    for row in range(5):
        idxs = rng.choice(g, size=k, replace=False)
        target = idxs[-1]
        counts[target] += 1
        for x in idxs:
            ax.add_patch(plt.Rectangle((x - 0.4, row), 0.8, 0.7, facecolor="#b2df8a", edgecolor="#333"))
        ax.add_patch(plt.Rectangle((target - 0.4, row), 0.8, 0.7, facecolor="#e31a1c", edgecolor="#333"))
    ax.text(0.02, 0.95, "Every direction can be\nthe target over training", transform=ax.transAxes, va="top", fontsize=7)

    fig.suptitle(r"Sampling coverage for fixed $K\ll G$ (schematic)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "rgs_vs_sequential_schematic.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    plot_sigma()
    plot_k_sweep()
    plot_pareto()
    plot_rgs_vs_sequential()
    for name in (
        "sigma_robustness_psnr_roi.png",
        "sigma_robustness_two_panel.png",
        "k_sweep_psnr_roi.png",
        "k_sweep_two_panel.png",
        "accuracy_cost_pareto.png",
        "rgs_vs_sequential_schematic.png",
    ):
        p = OUT / name
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
