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

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
import wandb
from dipy.data import get_fnames
from dipy.denoise.patch2self import patch2self
from dipy.io.image import load_nifti, save_nifti

from p2s.sklearn_patch2self import patch2self_sklearn
from paper_eval.dti_metrics import save_dti_metrics, try_compute_dti_errors
from utils import setup_logging
from utils.data import DBrainDataLoader, StanfordDataLoader
from utils.eval_protocol import (
    apply_reconstruction_eval_protocol,
    compute_roi_mask,
    metrics_policy_dict,
    save_run_manifest,
    summarize_roi,
)
from utils.metrics import (
    compute_metrics,
    fully_compare_volumes,
    save_metrics,
    visualize_single_volume,
)
from utils.repro_seed import configure_cudnn, set_seed
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
    backend = str(getattr(settings.patch2self, "backend", "dipy")).lower()
    if backend == "sklearn_reference":
        backend_suffix = (
            f"backend_sklearn_reference"
            f"_model_{getattr(settings.patch2self, 'sklearn_model', 'ols')}"
            f"_stride_{int(getattr(settings.patch2self, 'patch_stride', 1))}"
        )
    else:
        backend_suffix = (
            f"backend_dipy"
            f"_model_{getattr(settings.patch2self, 'model', 'ols')}"
        )

    return os.path.join(
        f"bvalue_{settings.data.bvalue}",
        f"noise_sigma_{settings.data.noise_sigma}",
        backend_suffix,
    )


def _log_gradient_split(bvals, b0_threshold):
    n_b0 = int(np.sum(bvals <= b0_threshold))
    n_dwi = int(np.sum(bvals > b0_threshold))
    logging.info(
        f"Gradient split: {len(bvals)} total, "
        f"{n_b0} b0 (bval <= {b0_threshold}), "
        f"{n_dwi} DWI (bval > {b0_threshold})"
    )
    return n_b0, n_dwi


def denoise_dwi_patch2self(data_4d, bvals, p2s_cfg):
    """Run DIPY Patch2Self on the full 4D DWI dataset.

    b0 volumes (bvals <= b0_threshold) are left intact (b0_denoising=False).
    Only diffusion-weighted volumes are denoised.

    Args:
        data_4d:  np.ndarray shape (X, Y, Z, V) — normalized 4D data.
        bvals:    1D array of b-values, length V.
        p2s_cfg:  Munch with model, shift_intensity, clip_negative_vals,
                  b0_threshold.

    Returns:
        denoised_4d: np.ndarray same shape, float32.
    """
    b0_threshold = int(getattr(p2s_cfg, "b0_threshold", 50))
    n_b0, n_dwi = _log_gradient_split(bvals, b0_threshold)

    if n_dwi == 0:
        logging.warning("No DWI volumes; returning input unchanged.")
        return data_4d.copy().astype(np.float32)

    logging.info(
        f"Running DIPY patch2self on {data_4d.shape} — "
        f"model='{p2s_cfg.model}', "
        f"shift_intensity={p2s_cfg.shift_intensity}, "
        f"clip_negative_vals={p2s_cfg.clip_negative_vals}, "
        f"b0_threshold={b0_threshold}"
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
    logging.info(f"DIPY patch2self completed in {time.time() - t0:.1f}s")
    logging.info(
        f"Denoised stats — min: {denoised.min():.4f}, "
        f"max: {denoised.max():.4f}, mean: {denoised.mean():.4f}"
    )
    return denoised.astype(np.float32)


def denoise_dwi_sklearn_reference(data_4d, bvals, p2s_cfg):
    """Run the sklearn reference Patch2Self (MD-S2S volume hold-out) on 4D DWI.

    Dispatches to :func:`p2s.sklearn_patch2self.patch2self_sklearn`.
    Config keys read from *p2s_cfg*:

    * ``b0_threshold``        (default 50)
    * ``sklearn_model``       (default ``"ols"``)
    * ``patch_radius``        (default ``[0, 0, 0]``)
    * ``patch_stride``        (default 1)
    * ``use_b0_as_predictors`` (default ``True``)

    Args:
        data_4d:  np.ndarray shape (X, Y, Z, V) — normalized 4D data.
        bvals:    1D array of b-values, length V.
        p2s_cfg:  Munch with the keys above.

    Returns:
        denoised_4d: np.ndarray same shape, float32.
    """
    b0_threshold = int(getattr(p2s_cfg, "b0_threshold", 50))
    model_name = str(getattr(p2s_cfg, "sklearn_model", "ols"))
    patch_radius = list(getattr(p2s_cfg, "patch_radius", [0, 0, 0]))
    stride = int(getattr(p2s_cfg, "patch_stride", 1))
    use_b0 = bool(getattr(p2s_cfg, "use_b0_as_predictors", True))

    _log_gradient_split(bvals, b0_threshold)
    logging.info("Using sklearn reference backend (MD-S2S style)")

    t0 = time.time()
    denoised = patch2self_sklearn(
        data_4d,
        bvals,
        b0_threshold=b0_threshold,
        model_name=model_name,
        patch_radius=patch_radius,
        stride=stride,
        use_b0_as_predictors=use_b0,
    )
    logging.info(f"sklearn reference completed in {time.time() - t0:.1f}s")
    return denoised


def main(
    dataset: str,
    reconstruct: bool = True,
    generate_images: bool = True,
    use_wandb: Optional[bool] = None,
):
    log_file = setup_logging(log_level=logging.INFO)
    logging.info(f"Starting Patch2Self pipeline for dataset: {dataset}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    logging.info(f"Loading config from: {config_path}")
    full_settings = load_config(config_path)

    wb_cfg = getattr(full_settings, "wandb", None)
    if use_wandb is None:
        use_wandb = bool(getattr(wb_cfg, "enabled", True)) if wb_cfg is not None else True
    wb_project = (
        getattr(wb_cfg, "project", "DWMRI-Denoising") if wb_cfg is not None else "DWMRI-Denoising"
    )

    if dataset == "dbrain":
        logging.info("Dataset: dBrain")
        settings = full_settings.dbrain
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
        settings = full_settings.stanford
        nii_path = None
        data_loader = StanfordDataLoader(
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
        )
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Must be 'stanford' or 'dbrain'.")
    seed = int(getattr(getattr(settings, "train", {}), "seed", 42))
    reproducible = bool(getattr(getattr(settings, "train", {}), "reproducible", False))
    set_seed(seed)
    configure_cudnn(fast=not reproducible)

    logging.info("Setting up wandb...")
    wandb_run = None
    try:
        if use_wandb:
            wandb_run = wandb.init(
                project=wb_project,
                tags=["Patch2Self", "P2S", dataset],
                config={
                    "dataset": dataset,
                    "model_name": "Patch2Self",
                    **settings.toDict(),
                },
            )
        else:
            logging.info("wandb disabled (use_wandb=False or wandb.enabled=false).")

        # Load data — shape (X, Y, Z, V) normalized to [0, 1] per volume
        logging.info("Loading data...")
        original_data, noisy_data = data_loader.load_data()
        logging.info(f"Noisy data shape: {noisy_data.shape}")

        if wandb_run is not None:
            wandb.log(
                {
                    "data/noisy_shape_x": noisy_data.shape[0],
                    "data/noisy_shape_y": noisy_data.shape[1],
                    "data/noisy_shape_z": noisy_data.shape[2],
                    "data/noisy_shape_volumes": noisy_data.shape[3],
                    "data/noisy_min": float(noisy_data.min()),
                    "data/noisy_max": float(noisy_data.max()),
                    "data/noisy_mean": float(noisy_data.mean()),
                }
            )

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

        # Denoise — backend selected by config (dipy | sklearn_reference)
        backend = str(getattr(settings.patch2self, "backend", "dipy")).lower()
        logging.info(f"Starting Patch2Self denoising (backend='{backend}')...")
        t_denoise = time.time()
        if backend == "sklearn_reference":
            denoised_data = denoise_dwi_sklearn_reference(
                noisy_data, bvals, settings.patch2self
            )
        else:
            if backend != "dipy":
                logging.warning(
                    f"Unknown backend '{backend}'; falling back to 'dipy'."
                )
            denoised_data = denoise_dwi_patch2self(
                noisy_data, bvals, settings.patch2self
            )
        denoise_seconds = time.time() - t_denoise
        logging.info(f"Denoised output shape: {denoised_data.shape}")

        if wandb_run is not None:
            wandb.log(
                {
                    "patch2self/denoise_seconds": denoise_seconds,
                    "patch2self/denoised_min": float(denoised_data.min()),
                    "patch2self/denoised_max": float(denoised_data.max()),
                    "patch2self/denoised_mean": float(denoised_data.mean()),
                }
            )

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

        if wandb_run is not None:
            wandb.log({"output/nifti_path": output_nii_path})

        # Metrics — DWI volumes only (b0s excluded for fairness)
        b0_threshold = int(getattr(settings.patch2self, "b0_threshold", 50))
        dwi_mask = bvals > b0_threshold
        den_dwi = denoised_data[..., dwi_mask]
        logging.info(f"Reconstructed DWIs shape: {den_dwi.shape}")
        logging.info(
            f"Reconstructed DWIs min: {den_dwi.min():.4f}, "
            f"max: {den_dwi.max():.4f}, mean: {den_dwi.mean():.4f}"
        )
        logging.info(f"Reconstructed DWIs dtype: {den_dwi.dtype}")
        metrics_dir = os.path.join(settings.reconstruct.metrics_dir, subdir)
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_roi = None
        if original_data is not None:
            logging.info("Computing metrics against clean reference (dBrain)...")
            ref_dwi = original_data[..., dwi_mask]
            den_dwi = apply_reconstruction_eval_protocol(
                den_dwi,
                ref_dwi,
                rescale_to_01=bool(getattr(settings.reconstruct, "rescale_to_01", True)),
                rescale_mode=str(getattr(settings.reconstruct, "rescale_mode", "per_volume")),
                clip_to_range=bool(getattr(settings.reconstruct, "clip_to_range", True)),
            )

            metrics = compute_metrics(ref_dwi, den_dwi)
            logging.info(f"Metrics: {metrics}")
            save_metrics(metrics, metrics_dir, filename="metrics.json")

            roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
            roi_mask_3d = compute_roi_mask(ref_dwi, roi_threshold)
            if roi_mask_3d is not None:
                n_roi, roi_pct = summarize_roi(roi_mask_3d)
                logging.info(
                    f"ROI mask: original > {roi_threshold}, "
                    f"{n_roi} voxels ({roi_pct:.1f}%)"
                )
                metrics_roi = compute_metrics(ref_dwi, den_dwi, mask=roi_mask_3d)
                logging.info(
                    f"Metrics (ROI, brain/tissue only): {metrics_roi}"
                )
                save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")
        else:
            logging.info(
                "No clean reference available (Stanford self-supervised). "
                "Computing proxy metrics vs noisy input."
            )
            noisy_dwi = noisy_data[..., dwi_mask]
            den_dwi = apply_reconstruction_eval_protocol(
                den_dwi,
                noisy_dwi,
                rescale_to_01=bool(getattr(settings.reconstruct, "rescale_to_01", True)),
                rescale_mode=str(getattr(settings.reconstruct, "rescale_mode", "per_volume")),
                clip_to_range=bool(getattr(settings.reconstruct, "clip_to_range", True)),
            )
            metrics = compute_metrics(noisy_dwi, den_dwi)
            logging.info(f"Metrics: {metrics}")
            save_metrics(metrics, metrics_dir, filename="metrics.json")

            roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
            roi_mask_3d = compute_roi_mask(noisy_dwi, roi_threshold)
            if roi_mask_3d is not None:
                n_roi, roi_pct = summarize_roi(roi_mask_3d)
                logging.info(
                    f"ROI mask: original > {roi_threshold}, "
                    f"{n_roi} voxels ({roi_pct:.1f}%)"
                )
                metrics_roi = compute_metrics(noisy_dwi, den_dwi, mask=roi_mask_3d)
                logging.info(
                    f"Metrics (ROI, brain/tissue only): {metrics_roi}"
                )
                save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")

        if wandb_run is not None:
            if original_data is not None:
                wandb.log(
                    {
                        "reconstruct/metrics_mse": metrics["mse"],
                        "reconstruct/metrics_ssim": metrics["ssim"],
                        "reconstruct/metrics_psnr": metrics["psnr"],
                    }
                )
                if metrics_roi is not None:
                    wandb.log(
                        {
                            "reconstruct/metrics_roi_mse": metrics_roi["mse"],
                            "reconstruct/metrics_roi_ssim": metrics_roi["ssim"],
                            "reconstruct/metrics_roi_psnr": metrics_roi["psnr"],
                        }
                    )
            else:
                wandb.log({"reconstruct/no_reference_metrics": True})

        if getattr(settings.reconstruct, "compute_dti", True) and dataset == "dbrain":
            if original_data is not None:
                bvals_full = np.asarray(gtab.bvals)[:n_vols]
                bvecs_full = np.asarray(gtab.bvecs)[:n_vols]
                roi_thr = getattr(settings.reconstruct, "metrics_roi_threshold", 0.02)
                dti = try_compute_dti_errors(
                    denoised_data.astype(np.float64),
                    original_data.astype(np.float64),
                    bvals_full,
                    bvecs_full,
                    roi_threshold=roi_thr,
                )
                save_dti_metrics(dti, metrics_dir)
            else:
                save_dti_metrics(
                    {
                        "fa_mae": None,
                        "md_mae": None,
                        "ad_mae": None,
                        "rd_mae": None,
                        "dti_reference": "self_reference_noisy",
                        "dti_skipped_reason": "no_clean_gt",
                    },
                    metrics_dir,
                )
        else:
            save_dti_metrics(
                {
                    "fa_mae": None,
                    "md_mae": None,
                    "ad_mae": None,
                    "rd_mae": None,
                    "dti_reference": "self_reference_noisy"
                    if original_data is None
                    else "clean_gt",
                    "dti_skipped_reason": "compute_dti_false_or_non_dbrain",
                },
                metrics_dir,
            )
        metrics_policy = metrics_policy_dict(
            reference_name="clean_gt" if original_data is not None else "self_reference_noisy",
            rescale_to_01=bool(getattr(settings.reconstruct, "rescale_to_01", True)),
            rescale_mode=str(getattr(settings.reconstruct, "rescale_mode", "per_volume")),
            clip_to_range=bool(getattr(settings.reconstruct, "clip_to_range", True)),
            roi_threshold=getattr(settings.reconstruct, "metrics_roi_threshold", None),
        )
        save_run_manifest(
            out_dir=metrics_dir,
            seed=seed,
            reproducible=reproducible,
            runtime_device="cpu",
            config={
                "dataset": dataset,
                "architecture": "patch2self",
                "backend": str(getattr(settings.patch2self, "backend", "dipy")).lower(),
                "b0_threshold": int(getattr(settings.patch2self, "b0_threshold", 50)),
            },
            metrics_policy=metrics_policy,
        )

        # Images
        if generate_images and getattr(settings.reconstruct, "generate_images", True):
            images_dir = os.path.join(settings.reconstruct.images_dir, subdir)
            os.makedirs(images_dir, exist_ok=True)
            logging.info(f"Saving comparison images to: {images_dir}")

            noisy_dwi = noisy_data[..., dwi_mask]
            n_img_vols = min(
                int(getattr(settings.reconstruct, "num_image_volumes", 10)),
                noisy_dwi.shape[-1],
            )

            # Transpose (X,Y,Z,V) → (Z,V,X,Y) so that fully_compare_volumes
            # and visualize_single_volume index [slice_z, vol, :, :] correctly,
            # producing a full axial slice rather than a band artifact.
            def _viz(arr):
                return np.transpose(arr, (2, 3, 0, 1))

            wandb_images = []
            if original_data is not None:
                ref_dwi = original_data[..., dwi_mask]
                for i in range(n_img_vols):
                    path = os.path.join(images_dir, f"comparison_volume_{i}.png")
                    fully_compare_volumes(
                        original_volume=_viz(ref_dwi),
                        noisy_volume=_viz(noisy_dwi),
                        denoised_volume=_viz(den_dwi),
                        file_name=path,
                        volume_idx=i,
                    )
                    if wandb_run is not None:
                        wandb_images.append(
                            wandb.Image(path, caption=f"Volume index {i}")
                        )
            else:
                # Stanford: no ground truth — compare noisy vs denoised only
                for i in range(n_img_vols):
                    path = os.path.join(
                        images_dir, f"noisy_vs_denoised_volume_{i}.png"
                    )
                    fully_compare_volumes(
                        original_volume=_viz(noisy_dwi),
                        noisy_volume=_viz(noisy_dwi),
                        denoised_volume=_viz(den_dwi),
                        file_name=path,
                        volume_idx=i,
                    )
                    if wandb_run is not None:
                        wandb_images.append(
                            wandb.Image(
                                path, caption=f"Noisy vs denoised, volume {i}"
                            )
                        )

            single_path = os.path.join(images_dir, "denoised_single.png")
            visualize_single_volume(
                _viz(den_dwi), file_name=single_path, volume_idx=0
            )

            noisy_path = os.path.join(images_dir, "noisy_single.png")
            visualize_single_volume(
                _viz(noisy_dwi), file_name=noisy_path, volume_idx=0
            )

            if wandb_run is not None and wandb_images:
                wandb.log({"reconstruct/comparison": wandb_images})
                wandb.log(
                    {
                        "reconstruct/denoised_single": wandb.Image(
                            single_path, caption="Denoised (first DWI)"
                        ),
                        "reconstruct/noisy_single": wandb.Image(
                            noisy_path, caption="Noisy (first DWI)"
                        ),
                    }
                )

            logging.info("Image generation complete.")

        logging.info("Patch2Self pipeline finished successfully.")
        logging.info(f"Log file: {log_file}")

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main(dataset="dbrain")
