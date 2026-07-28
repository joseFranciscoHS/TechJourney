#!/usr/bin/env python3
"""Stanford 2D-vs-3D spatial coherence panel (axial / sagittal / coronal)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "tmp" / "paper_final_k16_stanford_fixels" / "arrays"
OUT = ROOT / "paper" / "figures"

ARMS = [
    ("restormer2d", "Restormer-2D", "denoised_restormer2d.npy"),
    ("res_cnn_2d", "Res-CNN-2D", "denoised_res_cnn_2d.npy"),
    ("restormer3d", "Restormer-3D", "denoised_restormer3d.npy"),
    ("restormer3d_large", "Restormer-3D-large", "denoised_restormer3d_large.npy"),
]


def _load(arm: str, fname: str) -> np.ndarray:
    p = ARRAYS / arm / fname
    try:
        return np.load(p).astype(np.float32)
    except ValueError:
        nii = ARRAYS / arm / fname.replace(".npy", ".nii.gz")
        return np.asanyarray(nib.load(str(nii)).dataobj, dtype=np.float32)


def main() -> None:
    bvals = np.loadtxt(ARRAYS / "noisy" / "bvals")
    dwi_idx = int(np.where(bvals > 100)[0][len(np.where(bvals > 100)[0]) // 2])
    vols = [(t, _load(a, f)) for a, t, f in ARMS]
    x, y, z = [s // 2 for s in vols[0][1].shape[:3]]
    # shared clim from Restormer-3D
    ref = vols[2][1]
    lo, hi = np.percentile(ref[..., dwi_idx][np.isfinite(ref[..., dwi_idx])], [1, 99])

    views = [
        ("Axial", lambda v: np.rot90(v[:, :, z, dwi_idx])),
        ("Sagittal", lambda v: np.rot90(v[x, :, :, dwi_idx])),
        ("Coronal", lambda v: np.rot90(v[:, y, :, dwi_idx])),
    ]
    fig, axes = plt.subplots(3, len(vols), figsize=(1.8 * len(vols), 5.6), dpi=200)
    for j, (title, vol) in enumerate(vols):
        axes[0, j].set_title(title, fontsize=8)
        for i, (vname, fn) in enumerate(views):
            axes[i, j].imshow(fn(vol), cmap="gray", vmin=lo, vmax=hi)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].text(
                    -0.08,
                    0.5,
                    vname,
                    transform=axes[i, j].transAxes,
                    va="center",
                    ha="right",
                    fontsize=8,
                    rotation=90,
                )
    fig.suptitle("Stanford 2D vs 3D spatial coherence (shared clim; no GT)", fontsize=10)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    out = OUT / "stanford_2d_vs_3d_coherence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out, out.stat().st_size)


if __name__ == "__main__":
    main()
