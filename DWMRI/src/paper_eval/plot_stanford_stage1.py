#!/usr/bin/env python3
"""Stanford Stage-1 qualitative panel: DWI + FA (+ MD) for core arms.

No GT / no error maps. Caption should note qualitative / no clean reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from paper_eval.dti_metrics import compute_dti_maps

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "tmp" / "paper_final_k16_stanford_fixels" / "arrays"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Core Stage-1 arms (+ optional 2D).
ARMS = [
    ("noisy", "Noisy", "denoised_noisy.npy"),
    ("p2s", "Patch2Self", "denoised_p2s.npy"),
    ("drcnet3d", "DRCNet-Hybrid-RGS", "denoised_drcnet3d.npy"),
    ("restormer3d", "Restormer-Hybrid-RGS", "denoised_restormer3d.npy"),
    ("restormer2d", "Restormer-2D", "denoised_restormer2d.npy"),
    ("res_cnn_2d", "Res-CNN-2D", "denoised_res_cnn_2d.nii.gz"),
]


def _load_volume(arm_dir: Path, fname: str) -> np.ndarray:
    path = arm_dir / fname
    if path.suffix == ".gz" or path.name.endswith(".nii.gz"):
        return np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)
    try:
        return np.load(path).astype(np.float32)
    except ValueError:
        # Truncated npy fallback to nifti sibling.
        nii = arm_dir / fname.replace(".npy", ".nii.gz")
        if not nii.exists():
            raise
        return np.asanyarray(nib.load(str(nii)).dataobj, dtype=np.float32)


def _load_btable(arm_dir: Path):
    bvals = np.loadtxt(arm_dir / "bvals")
    bvecs = np.loadtxt(arm_dir / "bvecs")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    return bvals, bvecs


def _pick_slice(vol: np.ndarray) -> int:
    # Mid-axial slice with high energy in a mid-shell DWI volume.
    z = vol.shape[2]
    return int(z // 2)


def _pick_dwi_index(bvals: np.ndarray) -> int:
    # Prefer a mid-range DWI volume (not b0).
    dw = np.where(bvals > 100)[0]
    return int(dw[len(dw) // 2]) if len(dw) else 0


def main() -> None:
    panels = []
    bvals, bvecs = _load_btable(ARRAYS / "noisy")
    dwi_idx = _pick_dwi_index(bvals)
    z = None
    dwi_clim = None
    fa_clim = (0.0, 0.8)
    md_clim = None

    for arm, title, fname in ARMS:
        vol = _load_volume(ARRAYS / arm, fname)
        if z is None:
            z = _pick_slice(vol)
        maps = compute_dti_maps(vol, bvals, bvecs)
        dwi = vol[:, :, z, dwi_idx]
        fa = maps["fa"][:, :, z]
        md = maps["md"][:, :, z]
        if dwi_clim is None:
            lo, hi = np.percentile(dwi[np.isfinite(dwi)], [1, 99])
            dwi_clim = (float(lo), float(hi))
        if md_clim is None:
            lo, hi = np.percentile(md[np.isfinite(md)], [1, 99])
            md_clim = (float(lo), float(hi))
        panels.append((title, dwi, fa, md))

    n = len(panels)
    fig, axes = plt.subplots(3, n, figsize=(1.7 * n, 5.4), dpi=200)
    row_titles = ["DWI", "FA", "MD"]
    for j, (title, dwi, fa, md) in enumerate(panels):
        axes[0, j].imshow(np.rot90(dwi), cmap="gray", vmin=dwi_clim[0], vmax=dwi_clim[1])
        axes[1, j].imshow(np.rot90(fa), cmap="gray", vmin=fa_clim[0], vmax=fa_clim[1])
        axes[2, j].imshow(np.rot90(md), cmap="gray", vmin=md_clim[0], vmax=md_clim[1])
        axes[0, j].set_title(title, fontsize=8)
        for i in range(3):
            axes[i, j].axis("off")
    for i, rt in enumerate(row_titles):
        axes[i, 0].set_ylabel(rt, fontsize=9)
        # ylabel needs axis on; use text instead
        axes[i, 0].text(
            -0.08,
            0.5,
            rt,
            transform=axes[i, 0].transAxes,
            va="center",
            ha="right",
            fontsize=9,
            rotation=90,
        )
    fig.suptitle(
        "Stanford HARDI qualitative panel (no clean GT; shared display ranges)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.95])
    out = OUT / "stanford_stage1_dwi_fa_md.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out, out.stat().st_size)


if __name__ == "__main__":
    main()
