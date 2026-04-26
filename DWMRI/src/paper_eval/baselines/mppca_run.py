import argparse
import os

import numpy as np
from dipy.denoise.localpca import mppca

from paper_eval.dti_metrics import save_dti_metrics
from utils.metrics import compute_metrics, save_metrics


def run_mppca(noisy_xyzv: np.ndarray, gt_xyzv: np.ndarray, out_dir: str, patch_radius: int = 2):
    denoised, sigma = mppca(noisy_xyzv, patch_radius=patch_radius, return_sigma=True)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "denoised.npy"), denoised)
    np.save(os.path.join(out_dir, "sigma.npy"), sigma)
    metrics = compute_metrics(gt_xyzv, denoised)
    save_metrics(metrics, out_dir, filename="metrics.json")
    roi_mask = (gt_xyzv > 0.02).any(axis=-1)
    metrics_roi = compute_metrics(gt_xyzv, denoised, mask=roi_mask)
    save_metrics(metrics_roi, out_dir, filename="metrics_roi.json")
    save_dti_metrics({"fa_mae": None, "md_mae": None, "ad_mae": None, "rd_mae": None}, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MP-PCA baseline")
    parser.add_argument("--noisy", required=True, help="Path to noisy .npy (X,Y,Z,V)")
    parser.add_argument("--gt", required=True, help="Path to GT .npy (X,Y,Z,V)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--patch-radius", type=int, default=2)
    args = parser.parse_args()

    noisy = np.load(args.noisy)
    gt = np.load(args.gt)
    run_mppca(noisy, gt, args.out_dir, patch_radius=args.patch_radius)
