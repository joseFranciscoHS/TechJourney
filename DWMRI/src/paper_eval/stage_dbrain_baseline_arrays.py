#!/usr/bin/env python3
"""Stage existing D-Brain gt/noisy/MP-PCA/P2S into the qualitative export tree.

Writes physical-intensity 4D volumes under
`tmp/paper_final_k16_dbrain_exports/arrays/<arm>/` using the same
`denoised_<arm>.npy` + bvals/bvecs convention as hybrid exports.

Usage (from DWMRI/src, CPU-ok):
  python -m paper_eval.stage_dbrain_baseline_arrays
  python -m paper_eval.stage_dbrain_baseline_arrays --arms gt,noisy,mppca,p2s
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from paper_eval.export_denoised import save_arm
from utils.data import DBrainDataLoader

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "tmp" / "paper_final_shared_npy"
DEFAULT_OUT = ROOT / "tmp" / "paper_final_k16_dbrain_exports" / "arrays"
P2S_NII = (
    ROOT
    / "tmp"
    / "paper_final_k16_out"
    / "p2s"
    / "output"
    / "dbrain"
    / "bvalue_2500"
    / "noise_sigma_0.1"
    / "backend_dipy_model_ols"
    / "denoised_patch2self.nii.gz"
)
MPPCA_NPY = (
    ROOT / "tmp" / "paper_final_k16_out" / "baselines" / "mppca" / "dbrain" / "denoised.npy"
)

TX, TY, TZ = 128, 128, 96
NB0 = 6
TAKE_VOLUMES = 66


def _gradient_table():
    nii = "/teamspace/s3_folders/dwmri-dataset/D_BRAIN_b2500_6_60_14_HCP_nless.nii"
    bvecs = "/teamspace/s3_folders/dwmri-dataset/D_BRAIN_b2500_6_60_HCP_b_matrix.txt"
    loader = DBrainDataLoader(nii_path=nii, bvecs_path=bvecs, bvalue=2500, noise_sigma=0.1)
    gtab = loader.load_gradient_table()
    bvals = np.asarray(gtab.bvals)[:TAKE_VOLUMES]
    bvecs_arr = np.asarray(gtab.bvecs)[:TAKE_VOLUMES]
    # Identity affine matching the shared-crop origin (export_meta crop starts at 0).
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.4
    return bvals, bvecs_arr, affine


def _invert_shared(norm_01: np.ndarray, norm_params: np.ndarray) -> np.ndarray:
    """Invert per-volume min-max used by paper_final_shared_npy."""
    out = np.empty_like(norm_01, dtype=np.float32)
    for v in range(norm_01.shape[-1]):
        lo, hi = float(norm_params[v, 0]), float(norm_params[v, 1])
        out[..., v] = norm_01[..., v] * (hi - lo) + lo
    return out


def stage_gt(out_root: Path, bvals, bvecs, affine) -> None:
    gt01 = np.load(SHARED / "gt_full_xyzv.npy")
    params = np.load(SHARED / "norm_params.npy")
    assert gt01.shape == (TX, TY, TZ, TAKE_VOLUMES), gt01.shape
    vol = _invert_shared(gt01, params[:TAKE_VOLUMES])
    save_arm(str(out_root / "gt"), "gt", vol, affine, bvals, bvecs, NB0)


def stage_noisy(out_root: Path, bvals, bvecs, affine) -> None:
    """Noisy DWIs + clean b0s (matches hybrid reconstruct evaluation layout)."""
    gt01 = np.load(SHARED / "gt_full_xyzv.npy")
    noisy_dwi01 = np.load(SHARED / "noisy_dwi_xyzv.npy")
    params = np.load(SHARED / "norm_params.npy")
    assert noisy_dwi01.shape == (TX, TY, TZ, TAKE_VOLUMES - NB0), noisy_dwi01.shape
    b0 = _invert_shared(gt01[..., :NB0], params[:NB0])
    dwi = _invert_shared(noisy_dwi01, params[NB0:TAKE_VOLUMES])
    vol = np.concatenate([b0, dwi], axis=-1).astype(np.float32)
    save_arm(str(out_root / "noisy"), "noisy", vol, affine, bvals, bvecs, NB0)


def stage_mppca(out_root: Path, bvals, bvecs, affine) -> None:
    """MP-PCA DWI-only array + clean b0s from shared GT."""
    gt01 = np.load(SHARED / "gt_full_xyzv.npy")
    params = np.load(SHARED / "norm_params.npy")
    dwi = np.load(MPPCA_NPY).astype(np.float32)
    assert dwi.shape == (TX, TY, TZ, TAKE_VOLUMES - NB0), dwi.shape
    # Existing baseline MP-PCA is already in physical intensity (not [0,1]).
    # Confirm by comparing dynamic range to denormed GT DWI.
    gt_dwi = _invert_shared(gt01[..., NB0:], params[NB0:TAKE_VOLUMES])
    if float(np.nanmax(dwi)) <= 1.5:
        logging.warning("MP-PCA looks normalized; denormalizing with shared params")
        dwi = _invert_shared(dwi, params[NB0:TAKE_VOLUMES])
    else:
        # Sanity: scales should be in the same ballpark as GT DWI.
        logging.info(
            "MP-PCA physical range [%.3g, %.3g] vs GT DWI [%.3g, %.3g]",
            float(np.nanmin(dwi)),
            float(np.nanmax(dwi)),
            float(np.nanmin(gt_dwi)),
            float(np.nanmax(gt_dwi)),
        )
    b0 = _invert_shared(gt01[..., :NB0], params[:NB0])
    vol = np.concatenate([b0, dwi], axis=-1).astype(np.float32)
    save_arm(str(out_root / "mppca"), "mppca", vol, affine, bvals, bvecs, NB0)


def stage_p2s(out_root: Path, bvals, bvecs, affine) -> None:
    raw = np.asanyarray(nib.load(str(P2S_NII)).dataobj, dtype=np.float32)
    # P2S NIfTI is full Z=97; crop to shared take_z=96.
    vol = raw[:TX, :TY, :TZ, :TAKE_VOLUMES]
    assert vol.shape == (TX, TY, TZ, TAKE_VOLUMES), vol.shape
    params = np.load(SHARED / "norm_params.npy")
    if float(np.nanmax(vol)) <= 1.5:
        logging.info("P2S looks normalized [0,1]; denormalizing with shared params")
        vol = _invert_shared(vol, params[:TAKE_VOLUMES])
    save_arm(str(out_root / "p2s"), "p2s", vol, affine, bvals, bvecs, NB0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT,
        help="arrays root (contains <arm>/ subdirs)",
    )
    parser.add_argument(
        "--arms",
        default="gt,noisy,mppca,p2s",
        help="comma-separated subset of gt,noisy,mppca,p2s",
    )
    args = parser.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    bvals, bvecs, affine = _gradient_table()
    dispatch = {
        "gt": stage_gt,
        "noisy": stage_noisy,
        "mppca": stage_mppca,
        "p2s": stage_p2s,
    }
    for arm in arms:
        if arm not in dispatch:
            raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(dispatch)}")
        logging.info("Staging D-Brain arm %s -> %s", arm, args.out_root / arm)
        dispatch[arm](args.out_root, bvals, bvecs, affine)
    logging.info("Done. Staged arms: %s", ", ".join(arms))


if __name__ == "__main__":
    main()
