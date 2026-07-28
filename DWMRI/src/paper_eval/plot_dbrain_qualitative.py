#!/usr/bin/env python3
"""Compose D-Brain multi-method qualitative + FA/MD preservation panels.

Expects staged/exported volumes under
`tmp/paper_final_k16_dbrain_exports/arrays/<arm>/denoised_<arm>.npy`
with matching bvals/bvecs (see stage_dbrain_baseline_arrays + hybrid export).

Skips missing arms so partial trees still render; paper wiring should wait
until hybrids + MDS2S are present.

Usage (from DWMRI/src):
  python -m paper_eval.plot_dbrain_qualitative
  python -m paper_eval.plot_dbrain_qualitative --require-hybrids
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from paper_eval.dti_metrics import compute_dti_maps
from utils.eval_protocol import compute_roi_mask

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "tmp" / "paper_final_k16_dbrain_exports" / "arrays"
OUT = ROOT / "paper" / "figures"

# Main qualitative columns (order matters for the paper panel).
QUAL_ARMS = [
    ("noisy", "Noisy"),
    ("mppca", "MP-PCA"),
    ("p2s", "Patch2Self"),
    ("mds2s", "MD-S2S"),
    ("drcnet3d", "DRCNet-Hybrid-RGS"),
    ("restormer3d", "Restormer-Hybrid-RGS"),
    ("gt", "Clean"),
]

# FA/MD panel: reference first, then denoisers (no separate clean error column).
FAMD_ARMS = [
    ("gt", "Clean"),
    ("noisy", "Noisy"),
    ("mppca", "MP-PCA"),
    ("p2s", "Patch2Self"),
    ("mds2s", "MD-S2S"),
    ("drcnet3d", "DRCNet"),
    ("restormer3d", "Restormer"),
]


def _load_arm(arm: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    arm_dir = ARRAYS / arm
    npy = arm_dir / f"denoised_{arm}.npy"
    if not npy.exists():
        logging.warning("missing %s", npy)
        return None
    vol = np.load(npy).astype(np.float32)
    bvals = np.loadtxt(arm_dir / "bvals")
    bvecs = np.loadtxt(arm_dir / "bvecs")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    return vol, bvals, bvecs


def _pick_slice(vol: np.ndarray) -> int:
    return int(vol.shape[2] // 2)


def _pick_dwi_index(bvals: np.ndarray) -> int:
    dw = np.where(np.asarray(bvals) > 100)[0]
    return int(dw[len(dw) // 2]) if len(dw) else 0


def _zoom_box(shape_hw: Tuple[int, int]) -> Tuple[slice, slice]:
    h, w = shape_hw
    # Central WM-ish inset (~28% of FOV).
    rh, rw = max(h // 7, 12), max(w // 7, 12)
    cy, cx = h // 2, int(w * 0.42)
    return slice(cy - rh, cy + rh), slice(cx - rw, cx + rw)


def plot_qualitative(require_hybrids: bool) -> Path:
    loaded = []
    for arm, title in QUAL_ARMS:
        pack = _load_arm(arm)
        if pack is None:
            continue
        loaded.append((arm, title, pack[0], pack[1], pack[2]))

    have = {a for a, *_ in loaded}
    if require_hybrids and not {"drcnet3d", "restormer3d", "mds2s", "gt"}.issubset(have):
        raise SystemExit(
            f"require-hybrids: need drcnet3d, restormer3d, mds2s, gt; have {sorted(have)}"
        )
    if "gt" not in have or "noisy" not in have:
        raise SystemExit(f"need at least gt+noisy; have {sorted(have)}")

    gt = next(v for a, _, v, *_ in loaded if a == "gt")
    bvals = next(b for a, _, _, b, _ in loaded if a == "gt")
    z = _pick_slice(gt)
    dwi_idx = _pick_dwi_index(bvals)
    ref_slice = gt[:, :, z, dwi_idx]
    clim = tuple(np.percentile(ref_slice[np.isfinite(ref_slice)], [1, 99]))
    err_clim = None
    ys, xs = _zoom_box(ref_slice.shape)

    n = len(loaded)
    fig, axes = plt.subplots(3, n, figsize=(1.55 * n, 5.2), dpi=200)
    if n == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for j, (arm, title, vol, _, _) in enumerate(loaded):
        sl = vol[:, :, z, dwi_idx]
        err = np.abs(sl - ref_slice) if arm != "gt" else np.zeros_like(sl)
        if arm != "gt" and err_clim is None:
            # Shared error clim from noisy absolute error.
            noisy_sl = next(v for a, _, v, *_ in loaded if a == "noisy")[:, :, z, dwi_idx]
            e0 = np.abs(noisy_sl - ref_slice)
            err_clim = (0.0, float(np.percentile(e0[np.isfinite(e0)], 98)))
        axes[0, j].imshow(np.rot90(sl), cmap="gray", vmin=clim[0], vmax=clim[1])
        axes[1, j].imshow(
            np.rot90(sl[ys, xs]), cmap="gray", vmin=clim[0], vmax=clim[1]
        )
        axes[2, j].imshow(
            np.rot90(err),
            cmap="magma",
            vmin=0.0,
            vmax=(err_clim[1] if err_clim else 1.0),
        )
        axes[0, j].set_title(title, fontsize=7)
        for i in range(3):
            axes[i, j].axis("off")

    for i, lab in enumerate(["DWI", "Zoom", "|err|"]):
        axes[i, 0].text(
            -0.08,
            0.5,
            lab,
            transform=axes[i, 0].transAxes,
            va="center",
            ha="right",
            fontsize=8,
            rotation=90,
        )
    fig.suptitle(
        "D-Brain qualitative (shared clim; mid-shell DWI; σ=0.1 Rician)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.95])
    out = OUT / "dbrain_qualitative_main.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s (%d arms)", out, n)
    return out


def plot_famd(require_hybrids: bool) -> Path:
    packs = []
    for arm, title in FAMD_ARMS:
        pack = _load_arm(arm)
        if pack is None:
            continue
        packs.append((arm, title, *pack))
    have = {a for a, *_ in packs}
    if require_hybrids and not {"drcnet3d", "restormer3d", "gt"}.issubset(have):
        raise SystemExit(f"require-hybrids FA/MD missing hybrids; have {sorted(have)}")
    if "gt" not in have:
        raise SystemExit("FA/MD panel needs gt")

    gt_vol, bvals, bvecs = next((v, b, c) for a, _, v, b, c in packs if a == "gt")
    z = _pick_slice(gt_vol)
    gt_maps = compute_dti_maps(gt_vol, bvals, bvecs)
    fa_ref = gt_maps["fa"][:, :, z]
    md_ref = gt_maps["md"][:, :, z]
    fa_clim = (0.0, 0.8)
    md_clim = tuple(np.percentile(md_ref[np.isfinite(md_ref)], [1, 99]))
    fa_err_clim = None
    md_err_clim = None

    # Rows: FA, |FA err|, MD, |MD err|
    n = len(packs)
    fig, axes = plt.subplots(4, n, figsize=(1.55 * n, 6.4), dpi=200)
    if n == 1:
        axes = np.asarray(axes).reshape(4, 1)

    for j, (arm, title, vol, bv, bc) in enumerate(packs):
        maps = compute_dti_maps(vol, bv, bc)
        fa = maps["fa"][:, :, z]
        md = maps["md"][:, :, z]
        fa_err = np.abs(fa - fa_ref) if arm != "gt" else np.zeros_like(fa)
        md_err = np.abs(md - md_ref) if arm != "gt" else np.zeros_like(md)
        if arm == "noisy":
            fa_err_clim = (0.0, float(np.nanpercentile(fa_err, 98)))
            md_err_clim = (0.0, float(np.nanpercentile(md_err, 98)))
        axes[0, j].imshow(np.rot90(fa), cmap="gray", vmin=fa_clim[0], vmax=fa_clim[1])
        axes[1, j].imshow(
            np.rot90(fa_err),
            cmap="magma",
            vmin=0.0,
            vmax=(fa_err_clim[1] if fa_err_clim else 0.3),
        )
        axes[2, j].imshow(np.rot90(md), cmap="gray", vmin=md_clim[0], vmax=md_clim[1])
        axes[3, j].imshow(
            np.rot90(md_err),
            cmap="magma",
            vmin=0.0,
            vmax=(md_err_clim[1] if md_err_clim else 1e-3),
        )
        axes[0, j].set_title(title, fontsize=7)
        for i in range(4):
            axes[i, j].axis("off")

    for i, lab in enumerate(["FA", "|ΔFA|", "MD", "|ΔMD|"]):
        axes[i, 0].text(
            -0.08,
            0.5,
            lab,
            transform=axes[i, 0].transAxes,
            va="center",
            ha="right",
            fontsize=8,
            rotation=90,
        )
    fig.suptitle("D-Brain FA/MD preservation (shared scales vs clean GT)", fontsize=10)
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.95])
    out = OUT / "dbrain_fa_md_preservation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s (%d arms)", out, n)
    return out


def plot_per_gradient(require_hybrids: bool) -> Path:
    """Per-direction ROI PSNR violin/box across denoisers vs clean GT."""
    gt_pack = _load_arm("gt")
    if gt_pack is None:
        raise SystemExit("per-gradient needs gt")
    gt, bvals, _ = gt_pack
    denoisers: List[Tuple[str, str, np.ndarray]] = []
    for arm, title in [
        ("noisy", "Noisy"),
        ("mppca", "MP-PCA"),
        ("p2s", "P2S"),
        ("mds2s", "MDS2S"),
        ("drcnet3d", "DRCNet"),
        ("restormer3d", "Restormer"),
    ]:
        pack = _load_arm(arm)
        if pack is None:
            continue
        denoisers.append((arm, title, pack[0]))
    have = {a for a, _, _ in denoisers}
    if require_hybrids and not {"drcnet3d", "restormer3d"}.issubset(have):
        raise SystemExit("per-gradient require-hybrids missing")

    # ROI from clean 4D intensity (any volume above threshold).
    roi = compute_roi_mask(gt, threshold=0.02)
    if roi is None or not np.any(roi):
        raise SystemExit("empty ROI mask for per-gradient PSNR")
    dw_idx = np.where(np.asarray(bvals) > 100)[0]

    series = []
    labels = []
    for arm, title, vol in denoisers:
        psnrs = []
        for v in dw_idx:
            pred = vol[..., v][roi]
            ref = gt[..., v][roi]
            mse = float(np.mean((pred - ref) ** 2))
            if mse <= 0:
                psnrs.append(60.0)
            else:
                peak = float(np.max(ref)) if np.max(ref) > 0 else 1.0
                psnrs.append(20.0 * np.log10(peak / np.sqrt(mse)))
        series.append(psnrs)
        labels.append(title)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    parts = ax.violinplot(series, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.55)
    ax.boxplot(series, widths=0.18, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    ax.set_ylabel("PSNR-ROI (dB)")
    ax.set_title("D-Brain per-gradient PSNR-ROI (σ=0.1)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "dbrain_per_gradient_psnr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-hybrids",
        action="store_true",
        help="Fail if Hybrid RGS / MDS2S volumes are missing",
    )
    parser.add_argument(
        "--skip-famd",
        action="store_true",
        help="Skip FA/MD panel (DTI fit is slower)",
    )
    parser.add_argument(
        "--skip-per-gradient",
        action="store_true",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_qualitative(args.require_hybrids)
    if not args.skip_famd:
        plot_famd(args.require_hybrids)
    if not args.skip_per_gradient:
        plot_per_gradient(args.require_hybrids)


if __name__ == "__main__":
    main()
