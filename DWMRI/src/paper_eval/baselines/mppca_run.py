import argparse
import logging
import os

import numpy as np
from dipy.denoise.localpca import mppca

from paper_eval.dti_metrics import (
    bvals_bvecs_truncated_from_dbrain_matrix,
    save_dti_metrics,
    try_compute_dti_errors,
)
from utils.metrics import compute_metrics, save_metrics

logging.basicConfig(level=logging.INFO)


def run_mppca(
    noisy_xyzv: np.ndarray,
    gt_xyzv: np.ndarray,
    out_dir: str,
    patch_radius: int = 2,
    *,
    bvecs_path: str | None = None,
    bvalue: float = 2500.0,
    metrics_roi_threshold: float | None = 0.02,
):
    denoised, sigma = mppca(noisy_xyzv, patch_radius=patch_radius, return_sigma=True)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "denoised.npy"), denoised)
    np.save(os.path.join(out_dir, "sigma.npy"), sigma)
    metrics = compute_metrics(gt_xyzv, denoised)
    save_metrics(metrics, out_dir, filename="metrics.json")
    thr = 0.02 if metrics_roi_threshold is None else metrics_roi_threshold
    roi_mask = (gt_xyzv > thr).any(axis=-1)
    metrics_roi = compute_metrics(gt_xyzv, denoised, mask=roi_mask)
    save_metrics(metrics_roi, out_dir, filename="metrics_roi.json")

    if bvecs_path:
        nv = int(denoised.shape[-1])
        bvals, bvecs = bvals_bvecs_truncated_from_dbrain_matrix(
            bvecs_path, bvalue=bvalue, n_volumes=nv
        )
        dti = try_compute_dti_errors(
            denoised,
            gt_xyzv,
            bvals,
            bvecs,
            roi_threshold=metrics_roi_threshold,
        )
    else:
        dti = {
            "fa_mae": None,
            "md_mae": None,
            "ad_mae": None,
            "rd_mae": None,
            "dti_skipped_reason": "no_bvecs_path",
        }
    save_dti_metrics(dti, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MP-PCA baseline")
    parser.add_argument("--noisy", required=True, help="Path to noisy .npy (X,Y,Z,V)")
    parser.add_argument("--gt", required=True, help="Path to GT .npy (X,Y,Z,V)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument(
        "--bvecs-path",
        default=None,
        help="HCP-style b-matrix text file (same as DBrainDataLoader); enables DTI MAE",
    )
    parser.add_argument("--bvalue", type=float, default=2500.0)
    parser.add_argument(
        "--metrics-roi-threshold",
        type=float,
        default=0.02,
        help="ROI for DTI MAE (voxels where GT > threshold on any channel); use negative to disable",
    )
    args = parser.parse_args()

    noisy = np.load(args.noisy)
    gt = np.load(args.gt)
    roi_thr = (
        None
        if args.metrics_roi_threshold < 0
        else args.metrics_roi_threshold
    )
    run_mppca(
        noisy,
        gt,
        args.out_dir,
        patch_radius=args.patch_radius,
        bvecs_path=args.bvecs_path,
        bvalue=args.bvalue,
        metrics_roi_threshold=roi_thr,
    )
