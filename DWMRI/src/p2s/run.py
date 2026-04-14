"""
Patch2Self pipeline for full DWI reconstruction.

Applies DIPY's classical Patch2Self denoiser to a full 4D DWI dataset.
b0 volumes (bvals <= b0_threshold) are kept untouched; only diffusion-weighted
volumes (bvals > b0_threshold) are denoised (b0_denoising=False).
Supports Stanford (DIPY built-in HARDI) and dBrain datasets.

Usage (programmatic):
    from p2s.run import main
    main(dataset="stanford", reconstruct=True, generate_images=True)
    main(dataset="dbrain", reconstruct=True, generate_images=True)

Usage (CLI):
    python -m p2s.run          # defaults to dbrain
"""

import logging
import os
import time

import numpy as np
from dipy.denoise.patch2self import patch2self
from dipy.io.image import save_nifti, load_nifti
from dipy.data import get_fnames

from utils import setup_logging
from utils.data import DBrainDataLoader, StanfordDataLoader
from utils.metrics import (
    compute_metrics,
    save_metrics,
    fully_compare_volumes,
    visualize_single_volume,
)
from utils.utils import load_config


def _resolve_nii_path(settings_data):
    """
    Resolve NIfTI path for dBrain, trying default path then the lightning fallback.
    Raises FileNotFoundError if neither exists.
    """
    for attr in ("nii_path", "nii_path_lightning"):
        path = getattr(settings_data, attr, None)
        if path and os.path.exists(path):
            logging.info(f"Using NIfTI path: {path}")
            return path
    raise FileNotFoundError(
        "No accessible NIfTI path found. Tried nii_path and nii_path_lightning. "
        "Check p2s/config.yaml data section."
    )


def _resolve_bvecs_path(settings_data):
    """Resolve bvecs path for dBrain, trying default then lightning fallback."""
    for attr in ("bvecs_path", "bvecs_path_lightning"):
        path = getattr(settings_data, attr, None)
        if path and os.path.exists(path):
            logging.info(f"Using bvecs path: {path}")
            return path
    raise FileNotFoundError(
        "No accessible bvecs path found. Tried bvecs_path and bvecs_path_lightning. "
        "Check p2s/config.yaml data section."
    )


def _output_subdir(settings):
    """
    Build dataset-specific subdirectory suffix consistent with other modules
    (bvalue / noise_sigma).
    """
    return os.path.join(
        f"bvalue_{settings.data.bvalue}",
        f"noise_sigma_{settings.data.noise_sigma}",
    )


def denoise_dwi_patch2self(data_4d, bvals, p2s_cfg):
    """
    Run Patch2Self on the full 4D DWI dataset, denoising only DWI volumes
    (bvals > b0_threshold) and leaving b0 volumes intact (b0_denoising=False).

    Args:
        data_4d:  np.ndarray shape (X, Y, Z, V) — normalized 4D data.
        bvals:    1D array of b-values, length V, aligned to data_4d's last dim.
        p2s_cfg:  Munch object with model, shift_intensity, clip_negative_vals,
                  b0_threshold.

    Returns:
        denoised_4d: np.ndarray same shape as data_4d, float32.
    """
    b0_threshold = int(getattr(p2s_cfg, "b0_threshold", 50))
    n_b0 = int(np.sum(bvals <= b0_threshold))
    n_dwi = int(np.sum(bvals > b0_threshold))

    logging.info(
        f"Gradient split: {len(bvals)} total, "
        f"{n_b0} b0 volumes (bval <= {b0_threshold}), "
        f"{n_dwi} DWI volumes (bval > {b0_threshold})"
    )

    if n_dwi == 0:
        logging.warning(
            "No DWI volumes found (all bvals <= b0_threshold). Returning original data."
        )
        return data_4d.copy().astype(np.float32)

    logging.info(
        f"Running patch2self on {data_4d.shape} — "
        f"model='{p2s_cfg.model}', shift_intensity={p2s_cfg.shift_intensity}, "
        f"clip_negative_vals={p2s_cfg.clip_negative_vals}, "
        f"b0_threshold={b0_threshold}, b0_denoising=False"
    )

    t0 = time.time()
    denoised = patch2self(
        data_4d.astype(np.float64),
        bvals,
        model=p2s_cfg.model,
        b0_threshold=b0_threshold,
        shift_intensity=p2s_cfg.shift_intensity,
        clip_negative_vals=p2s_cfg.clip_negative_vals,
        b0_denoising=False,
        verbose=True,
    )
    elapsed = time.time() - t0
    logging.info(f"patch2self completed in {elapsed:.1f}s")
    logging.info(
        f"Denoised stats — min: {denoised.min():.4f}, "
        f"max: {denoised.max():.4f}, mean: {denoised.mean():.4f}"
    )
    return denoised.astype(np.float32)


def main(
    dataset: str,
    reconstruct: bool = True,
    generate_images: bool = True,
):
    log_file = setup_logging(log_level=logging.INFO)
    logging.info(f"Starting Patch2Self pipeline for dataset: {dataset}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    logging.info(f"Loading config from: {config_path}")
    settings = load_config(config_path)

    if dataset == "dbrain":
        logging.info("Dataset: dBrain")
        settings = settings.dbrain
        nii_path = _resolve_nii_path(settings.data)
        bvecs_path = _resolve_bvecs_path(settings.data)
        data_loader = DBrainDataLoader(
            nii_path=nii_path,
            bvecs_path=bvecs_path,
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
        )
    elif dataset == "stanford":
        logging.info("Dataset: Stanford HARDI")
        settings = settings.stanford
        data_loader = StanfordDataLoader(
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
        )
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Must be 'stanford' or 'dbrain'.")

    # Load data — shape (X, Y, Z, V) normalized to [0, 1] per volume
    logging.info("Loading data...")
    original_data, noisy_data = data_loader.load_data()
    logging.info(f"Noisy data shape: {noisy_data.shape}")

    # Load gradient table to get bvals aligned to all volumes
    logging.info("Loading gradient table...")
    gtab = data_loader.load_gradient_table()
    bvals = gtab.bvals
    logging.info(f"bvals shape: {bvals.shape}, unique: {np.unique(bvals)}")

    # Guard: align bvals length to data volume count in case loader subsets
    n_vols = noisy_data.shape[-1]
    if len(bvals) != n_vols:
        logging.warning(
            f"bvals length ({len(bvals)}) != data volumes ({n_vols}). "
            f"Truncating bvals to first {n_vols} entries."
        )
        bvals = bvals[:n_vols]

    if not reconstruct:
        logging.info("reconstruct=False — skipping denoising and output steps.")
        return

    subdir = _output_subdir(settings)

    # Denoise — b0 volumes are kept intact via b0_denoising=False
    logging.info("Starting Patch2Self denoising...")
    denoised_data = denoise_dwi_patch2self(noisy_data, bvals, settings.patch2self)
    logging.info(f"Denoised output shape: {denoised_data.shape}")

    # Save denoised NIfTI (preserves affine from source file)
    output_dir = os.path.join(settings.reconstruct.output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    output_nii_path = os.path.join(output_dir, "denoised_patch2self.nii.gz")

    if dataset == "stanford":
        hardi_fname, _, _ = get_fnames("stanford_hardi")
        _, affine = load_nifti(hardi_fname)
    else:
        _, affine = load_nifti(nii_path)

    save_nifti(output_nii_path, denoised_data, affine)
    logging.info(f"Saved denoised NIfTI to: {output_nii_path}")

    # Metrics — DWI volumes only (b0s excluded for fairness)
    b0_threshold = int(getattr(settings.patch2self, "b0_threshold", 50))
    dwi_mask = bvals > b0_threshold
    metrics_dir = os.path.join(settings.reconstruct.metrics_dir, subdir)
    os.makedirs(metrics_dir, exist_ok=True)

    if original_data is not None:
        logging.info("Computing metrics against clean reference (dBrain)...")
        ref_dwi = original_data[..., dwi_mask]
        den_dwi = denoised_data[..., dwi_mask]

        metrics = compute_metrics(ref_dwi, den_dwi)
        logging.info(f"Full-image metrics: {metrics}")
        save_metrics(metrics, metrics_dir, filename="metrics.json")

        roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
        if roi_threshold is not None:
            roi_mask_3d = (ref_dwi > roi_threshold).any(axis=-1)
            n_roi = int(np.sum(roi_mask_3d))
            logging.info(
                f"ROI mask: ref_dwi > {roi_threshold}, "
                f"{n_roi} voxels ({100.0 * n_roi / roi_mask_3d.size:.1f}%)"
            )
            metrics_roi = compute_metrics(ref_dwi, den_dwi, mask=roi_mask_3d)
            logging.info(f"ROI metrics: {metrics_roi}")
            save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")
    else:
        logging.info(
            "No clean reference available (Stanford self-supervised). "
            "Skipping GT-based metrics."
        )
        save_metrics({"note": "no_reference_available"}, metrics_dir, filename="metrics.json")

    # Images
    if generate_images and getattr(settings.reconstruct, "generate_images", True):
        images_dir = os.path.join(settings.reconstruct.images_dir, subdir)
        os.makedirs(images_dir, exist_ok=True)
        logging.info(f"Saving comparison images to: {images_dir}")

        noisy_dwi = noisy_data[..., dwi_mask]
        den_dwi = denoised_data[..., dwi_mask]
        n_img_vols = min(
            int(getattr(settings.reconstruct, "num_image_volumes", 10)),
            noisy_dwi.shape[-1],
        )

        if original_data is not None:
            ref_dwi = original_data[..., dwi_mask]
            for i in range(n_img_vols):
                path = os.path.join(images_dir, f"comparison_volume_{i}.png")
                fully_compare_volumes(
                    original_volume=ref_dwi,
                    noisy_volume=noisy_dwi,
                    denoised_volume=den_dwi,
                    file_name=path,
                    volume_idx=i,
                )
        else:
            # Stanford: no ground truth — compare noisy vs denoised only
            for i in range(n_img_vols):
                path = os.path.join(images_dir, f"noisy_vs_denoised_volume_{i}.png")
                fully_compare_volumes(
                    original_volume=noisy_dwi,
                    noisy_volume=noisy_dwi,
                    denoised_volume=den_dwi,
                    file_name=path,
                    volume_idx=i,
                )

        single_path = os.path.join(images_dir, "denoised_single.png")
        visualize_single_volume(den_dwi, file_name=single_path, volume_idx=0)

        noisy_path = os.path.join(images_dir, "noisy_single.png")
        visualize_single_volume(noisy_dwi, file_name=noisy_path, volume_idx=0)

        logging.info("Image generation complete.")

    logging.info("Patch2Self pipeline finished successfully.")
    logging.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main(dataset="dbrain")
