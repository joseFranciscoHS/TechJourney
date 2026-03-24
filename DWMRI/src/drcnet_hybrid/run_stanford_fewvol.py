"""
Stanford HARDI (DRCNet): train on few DWI volumes; optionally reconstruct all non-b0 DWI.

``input_channels`` must match ``train_num_volumes``. Full-dataset inference runs
:func:`drcnet_hybrid.reconstruction.reconstruct_full_dwi_chunked`, which calls
``reconstruct_dwis`` on contiguous volume blocks and concatenates.

Usage (from ``DWMRI/src`` with PYTHONPATH)::

    python -m drcnet_hybrid.run_stanford_fewvol
    python -m drcnet_hybrid.run_stanford_fewvol --force-train
    python -m drcnet_hybrid.run_stanford_fewvol --skip-train
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys

import numpy as np
import torch
from drcnet_hybrid.data import TrainingDataSet
from drcnet_hybrid.fit import fit_model
from drcnet_hybrid.model import DenoiserNet
from drcnet_hybrid.reconstruction import reconstruct_dwis, reconstruct_full_dwi_chunked
from drcnet_hybrid.run import fit_progressive
from torch.utils.data import DataLoader, Subset
from utils import setup_logging
from utils.checkpoint import load_checkpoint
from utils.data import (
    StanfordDataLoader,
    compute_brain_mask,
    rescale_reconstruction_to_01,
)
from utils.metrics import compute_metrics, save_metrics
from utils.multi_gpu import create_multi_gpu_config_from_dict, setup_multi_gpu
from utils.utils import load_config, noise_path_segment

import wandb


def _build_checkpoint_dir(settings, train_num_volumes: int) -> str:
    noise_segment = noise_path_segment(
        getattr(settings.data, "noise_type", "rician"),
        getattr(settings.data, "noise_sigma", 0.1),
    )
    bvalue_segment = f"b{getattr(settings.data, 'bvalue', 2500)}"
    return os.path.join(
        settings.train.checkpoint_dir,
        bvalue_segment,
        f"num_volumes_{train_num_volumes}",
        noise_segment,
        f"learning_rate_{settings.train.learning_rate}",
    )


def _prepare_stanford_arrays(settings, train_num_volumes: int):
    d = settings.data
    data_loader = StanfordDataLoader(
        bvalue=d.bvalue,
        noise_sigma=d.noise_sigma,
    )
    original_data, noisy_data = data_loader.load_data()
    if original_data is None:
        original_data = noisy_data
        logging.info(
            "StanfordDataLoader returned original_data=None; using noisy_data as reference"
        )

    tx, ty, tz = d.take_x, d.take_y, d.take_z
    cropped_noisy = noisy_data[:tx, :ty, :tz, :].copy()
    cropped_orig = original_data[:tx, :ty, :tz, :].copy()

    n_b0 = int(d.num_b0s)
    dwi_noisy = cropped_noisy[..., n_b0:]
    dwi_orig = cropped_orig[..., n_b0:]
    n_dwi = dwi_noisy.shape[-1]
    logging.info(
        f"After spatial crop and b0 skip: DWI shape (X,Y,Z,V)={dwi_noisy.shape}, "
        f"V={n_dwi} diffusion volumes (excluding {n_b0} b0s)"
    )

    if train_num_volumes > n_dwi:
        raise ValueError(
            f"train_num_volumes={train_num_volumes} exceeds available DWI volumes ({n_dwi})"
        )

    train_noisy = dwi_noisy[..., :train_num_volumes]
    train_orig = dwi_orig[..., :train_num_volumes]

    return train_noisy, train_orig, dwi_noisy, dwi_orig


def _make_denoiser(
    settings,
    inp_channels: int,
    spatial_patch: tuple[int, int, int],
):
    return DenoiserNet(
        input_channels=inp_channels,
        output_channels=settings.model.out_channel,
        groups=settings.model.groups,
        dense_convs=settings.model.dense_convs,
        residual=settings.model.residual,
        base_filters=settings.model.base_filters,
        output_shape=(
            settings.model.out_channel,
            spatial_patch[0],
            spatial_patch[1],
            spatial_patch[2],
        ),
        device=settings.train.device,
        output_activation=getattr(settings.model, "output_activation", "prelu"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="DRCNet: train on few Stanford DWI volumes; reconstruct all non-b0 DWI."
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-reconstruct", action="store_true")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases (no wandb.init / logging to the cloud).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    settings = load_config(config_path).stanford

    train_num_volumes = int(
        getattr(settings.data, "train_num_volumes", settings.data.num_volumes)
    )
    reconstruct_full = bool(getattr(settings.data, "reconstruct_full_dwi", True))

    settings.data.num_volumes = train_num_volumes
    settings.model.in_channel = train_num_volumes

    log_file = setup_logging(log_level=logging.INFO)
    logging.info(
        f"Stanford few-volume (DRCNet): train_num_volumes={train_num_volumes}, "
        f"reconstruct_full_dwi={reconstruct_full}"
    )
    logging.info(
        "Logs also go to the file above; if the terminal looks empty, open that path "
        "or run with: python -u runner_stanford.py"
    )

    wandb_run = None
    if not args.no_wandb:
        noise_segment = noise_path_segment(
            getattr(settings.data, "noise_type", "rician"),
            getattr(settings.data, "noise_sigma", 0.1),
        )
        bvalue_segment = f"b{getattr(settings.data, 'bvalue', 2500)}"
        wandb_config = {
            "dataset": "stanford_fewvol_drcnet",
            "train_num_volumes": train_num_volumes,
            "reconstruct_full_dwi": reconstruct_full,
            "learning_rate": settings.train.learning_rate,
            "bvalue": getattr(settings.data, "bvalue", None),
            "noise_sigma": getattr(settings.data, "noise_sigma", None),
            "noise_type": getattr(settings.data, "noise_type", None),
        }
        wandb_run = wandb.init(
            project="DWMRI-Denoising",
            name=f"drcnet_stanford_fewvol_{bvalue_segment}_nvol{train_num_volumes}_{noise_segment}",
            tags=["stanford", "fewvol", "drcnet", bvalue_segment, noise_segment],
            config=wandb_config,
        )
        logging.info("wandb run started (project DWMRI-Denoising).")

    try:
        _run_stanford_fewvol_body(
            args,
            settings,
            train_num_volumes,
            reconstruct_full,
            log_file,
            wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_stanford_fewvol_body(
    args, settings, train_num_volumes, reconstruct_full, log_file, wandb_run
):
    train_noisy, train_orig, full_noisy, full_orig = _prepare_stanford_arrays(
        settings, train_num_volumes
    )

    patch_filter_method = getattr(settings.data, "patch_filter_method", "none")
    min_signal_threshold = getattr(settings.data, "min_signal_threshold", 0.0)
    otsu_median_radius = getattr(settings.data, "otsu_median_radius", 2)
    otsu_numpass = getattr(settings.data, "otsu_numpass", 1)

    brain_mask = None
    if patch_filter_method == "otsu":
        brain_mask = compute_brain_mask(
            train_orig,
            median_radius=otsu_median_radius,
            numpass=otsu_numpass,
        )

    checkpoint_dir = _build_checkpoint_dir(settings, train_num_volumes)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "best_loss_checkpoint.pth")

    noise_segment = noise_path_segment(
        getattr(settings.data, "noise_type", "rician"),
        getattr(settings.data, "noise_sigma", 0.1),
    )
    bvalue_segment = f"b{getattr(settings.data, 'bvalue', 2500)}"
    loss_dir = os.path.join(
        "drcnet_hybrid/losses",
        "stanford_fewvol",
        bvalue_segment,
        f"num_volumes_{train_num_volumes}",
        noise_segment,
        f"learning_rate_{settings.train.learning_rate}",
    )
    os.makedirs(loss_dir, exist_ok=True)

    progressive_enabled = hasattr(settings.train, "progressive") and getattr(
        settings.train.progressive, "enabled", False
    )

    do_train = not args.skip_train
    if do_train and not args.force_train and os.path.isfile(best_ckpt):
        logging.info(f"Checkpoint exists, skipping training: {best_ckpt}")
        do_train = False

    if do_train:
        if progressive_enabled:
            fs = settings.train.progressive.stages[0]
            spatial = (fs.patch_size, fs.patch_size, fs.patch_size)
        else:
            ps = settings.data.patch_size
            spatial = (ps, ps, ps)

        logging.info(
            f"Initializing DenoiserNet for training (spatial patch {spatial[0]}³)..."
        )
        model = _make_denoiser(settings, train_num_volumes, spatial)

        batch_for_multi_gpu = (
            settings.train.progressive.stages[0].batch_size
            if progressive_enabled
            else settings.train.batch_size
        )
        multi_gpu_config = create_multi_gpu_config_from_dict(
            {
                "multi_gpu": settings.train.multi_gpu,
                "gpu_ids": settings.train.gpu_ids,
                "auto_scale_lr": settings.train.auto_scale_lr,
                "learning_rate": settings.train.learning_rate,
                "batch_size": batch_for_multi_gpu,
                "auto_exclude_imbalanced": settings.train.auto_exclude_imbalanced,
                "memory_threshold": settings.train.memory_threshold,
            }
        )
        model, effective_lr, effective_batch_size = setup_multi_gpu(
            model, multi_gpu_config
        )
        logging.info(
            f"Optimizer prep: effective_lr={effective_lr:.6f}, "
            f"effective_batch_size={effective_batch_size}"
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
        scheduler = None
        if getattr(settings.train, "use_scheduler", False):
            stype = getattr(settings.train, "scheduler_type", "step")
            if stype == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=settings.train.scheduler_step_size,
                    gamma=settings.train.scheduler_gamma,
                )
            elif stype == "reduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=settings.train.scheduler_patience,
                    factor=settings.train.scheduler_factor,
                    min_lr=settings.train.min_lr,
                )
            elif stype == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=settings.train.scheduler_T_0,
                    T_mult=settings.train.scheduler_T_mult,
                    eta_min=settings.train.eta_min_lr,
                )

        use_amp = getattr(settings.train, "use_amp", True)

        if progressive_enabled:
            fit_progressive(
                model=model,
                settings=settings,
                noisy_data=train_noisy,
                original_data=train_orig,
                brain_mask=brain_mask,
                checkpoint_dir=checkpoint_dir,
                loss_dir=loss_dir,
                patch_filter_method=patch_filter_method,
                min_signal_threshold=min_signal_threshold,
            )
        else:
            train_set = TrainingDataSet(
                data=train_noisy,
                patch_size=(
                    train_num_volumes,
                    settings.data.patch_size,
                    settings.data.patch_size,
                    settings.data.patch_size,
                ),
                step=settings.data.step,
                mask_p=settings.train.mask_p,
                clean_data=train_orig,
                brain_mask=brain_mask,
                patch_filter_method=patch_filter_method,
                min_signal_threshold=min_signal_threshold,
            )
            subset_fraction = 0.6
            total_samples = len(train_set)
            num_samples = int(total_samples * subset_fraction)
            np.random.seed(getattr(settings.train, "seed", 42))
            indices = np.random.choice(total_samples, size=num_samples, replace=False)
            train_set = Subset(train_set, indices)
            train_loader = DataLoader(
                train_set,
                batch_size=settings.train.batch_size,
                shuffle=True,
            )
            fit_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                num_epochs=settings.train.num_epochs,
                device=settings.train.device,
                checkpoint_dir=checkpoint_dir,
                loss_dir=loss_dir,
                use_amp=use_amp,
            )

        del model
        gc.collect()
        if settings.train.device.startswith("cuda"):
            torch.cuda.empty_cache()
        logging.info(f"Training finished. Log file: {log_file}")

    if args.skip_reconstruct:
        logging.info("--skip-reconstruct set; done.")
        return

    if not os.path.isfile(best_ckpt):
        logging.error(
            f"No checkpoint at {best_ckpt}; train first or use --force-train."
        )
        sys.exit(1)

    d = settings.data
    spatial_full = (d.take_x, d.take_y, d.take_z)
    logging.info(
        f"Loading checkpoint for full-volume reconstruction (spatial {spatial_full})..."
    )
    rec_dev = settings.reconstruct.device
    rec_model = _make_denoiser(settings, train_num_volumes, spatial_full)
    rec_model, _, _, _, _, _ = load_checkpoint(
        model=rec_model,
        optimizer=None,
        filename=best_ckpt,
        device=rec_dev,
        strict=False,
    )

    if reconstruct_full:
        logging.info(
            f"Reconstructing all {full_noisy.shape[-1]} DWI volumes via "
            f"chunks of {train_num_volumes}..."
        )
        reconstructed = reconstruct_full_dwi_chunked(
            model=rec_model,
            noisy_xyzv=full_noisy,
            train_num_volumes=train_num_volumes,
            device=rec_dev,
            mask_p=settings.reconstruct.mask_p,
            n_preds=settings.reconstruct.n_preds,
        )
        ref_for_metrics = full_orig
    else:
        logging.info("Reconstructing training subset only.")
        x_t = torch.from_numpy(np.transpose(train_noisy, (3, 0, 1, 2))).type(
            torch.float
        )
        rec_vxyz = reconstruct_dwis(
            model=rec_model,
            data=x_t,
            device=rec_dev,
            mask_p=settings.reconstruct.mask_p,
            n_preds=settings.reconstruct.n_preds,
        )
        reconstructed = np.transpose(rec_vxyz, (1, 2, 3, 0))
        ref_for_metrics = train_orig

    if getattr(settings.reconstruct, "rescale_to_01", False):
        mode = getattr(settings.reconstruct, "rescale_mode", "per_volume")
        reference = ref_for_metrics if mode == "match_gt" else None
        reconstructed = rescale_reconstruction_to_01(
            reconstructed, mode=mode, reference=reference
        )

    if getattr(settings.reconstruct, "subtract_background_estimate", False):
        thresh = getattr(settings.reconstruct, "subtract_background_threshold", 0.02)
        bg_mask = (ref_for_metrics <= thresh).all(axis=-1)
        if np.any(bg_mask):
            shift = float(np.median(reconstructed[bg_mask]))
            reconstructed = np.clip(
                reconstructed.astype(np.float64) - shift, 0, 1
            ).astype(np.float32)

    if getattr(settings.reconstruct, "clip_to_range", False):
        reconstructed = np.clip(reconstructed, 0, 1)

    tag = "full_dwi" if reconstruct_full else "train_subset"
    metrics_dir = os.path.join(
        settings.reconstruct.metrics_dir,
        bvalue_segment,
        f"num_volumes_{train_num_volumes}_{tag}",
        noise_segment,
        f"learning_rate_{settings.train.learning_rate}",
    )
    os.makedirs(metrics_dir, exist_ok=True)

    metrics = compute_metrics(ref_for_metrics, reconstructed)
    logging.info(f"Metrics ({tag}): {metrics}")
    save_metrics(metrics, metrics_dir, filename="metrics.json")

    if wandb_run is not None:
        wandb.log(
            {
                f"reconstruct/{tag}/metrics_mse": metrics["mse"],
                f"reconstruct/{tag}/metrics_ssim": metrics["ssim"],
                f"reconstruct/{tag}/metrics_psnr": metrics["psnr"],
            }
        )

    roi_thr = getattr(settings.reconstruct, "metrics_roi_threshold", None)
    if roi_thr is not None:
        roi_mask = (ref_for_metrics > roi_thr).any(axis=-1)
        metrics_roi = compute_metrics(ref_for_metrics, reconstructed, mask=roi_mask)
        logging.info(f"Metrics ROI ({tag}): {metrics_roi}")
        save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")
        if wandb_run is not None:
            wandb.log(
                {
                    f"reconstruct/{tag}/metrics_roi_mse": metrics_roi["mse"],
                    f"reconstruct/{tag}/metrics_roi_ssim": metrics_roi["ssim"],
                    f"reconstruct/{tag}/metrics_roi_psnr": metrics_roi["psnr"],
                }
            )

    logging.info(f"Saved metrics under: {metrics_dir}")
    logging.info(f"Done. Log file: {log_file}")


if __name__ == "__main__":
    main()
