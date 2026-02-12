import logging
import os

import numpy as np
import torch
from drcnet_hybrid_multiple_networks.data import TrainingDataSet
from drcnet_hybrid_multiple_networks.fit import fit_model
from drcnet_hybrid_multiple_networks.model import DenoiserNet
from drcnet_hybrid_multiple_networks.reconstruction import reconstruct_single_volume
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import setup_logging
from utils.checkpoint import load_checkpoint
from utils.data import DBrainDataLoader, StanfordDataLoader
from utils.metrics import (
    compute_metrics,
    fully_compare_volumes,
    save_metrics,
    visualize_single_volume,
)
from utils.multi_gpu import create_multi_gpu_config_from_dict, setup_multi_gpu
from utils.utils import load_config
import wandb


def main(
    dataset: str,
    train: bool = True,
    reconstruct: bool = True,
    generate_images: bool = True,
    transfer_learn: bool = True,
):
    # Setup logging
    log_file = setup_logging(log_level=logging.INFO)
    logging.info(f"Starting training with dataset: {dataset}")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    logging.info(f"Loading config from: {config_path}")

    settings = load_config(config_path)
    logging.info("Configuration loaded successfully")

    if dataset == "dbrain":
        logging.info("Using DBrain dataset configuration")
        settings = settings.dbrain
        data_loader = DBrainDataLoader(
            nii_path=settings.data.nii_path,
            bvecs_path=settings.data.bvecs_path,
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
        )
        logging.info(
            f"DBrainDataLoader initialized with noise_sigma={settings.data.noise_sigma}"
        )
    elif dataset == "stanford":
        logging.info("Using Stanford dataset configuration")
        settings = settings.stanford
        data_loader = StanfordDataLoader(settings.data)
        logging.info("StanfordDataLoader initialized")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    logging.info("Setting up wandb...")
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project="DWMRI-Denoising",
            config={
                "dataset": dataset,
                "model_name": "DRCNet-hybrid-multiple-networks",
                **settings.toDict(),
            },
        )
        logging.info("Loading data...")
        original_data, noisy_data = data_loader.load_data()
        # omitting the b0s from the data
        take_volumes = settings.data.num_b0s + settings.data.num_volumes
        logging.info(f"Taking volumes from {settings.data.num_b0s} to {take_volumes}")
        noisy_data = noisy_data[
            : settings.data.take_x,
            : settings.data.take_y,
            : settings.data.take_z,
            settings.data.num_b0s : take_volumes,
        ]
        original_data = original_data[
            : settings.data.take_x,
            : settings.data.take_y,
            : settings.data.take_z,
            settings.data.num_b0s : take_volumes,
        ]
        logging.info(f"Noisy data shape: {noisy_data.shape}")
        logging.info(
            f"Data type: {noisy_data.dtype}, Min: {noisy_data.min():.4f}, Max: {noisy_data.max():.4f}, Mean: {noisy_data.mean():.4f}"
        )

        # Base checkpoint and loss dirs (one subdir per volume: volume_0, volume_1, ...)
        checkpoint_dir = os.path.join(
            settings.train.checkpoint_dir,
            f"bvalue_{settings.data.bvalue}",
            f"num_volumes_{settings.data.num_volumes}",
            f"noise_sigma_{settings.data.noise_sigma}",
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_dir = os.path.join(
            "drcnet_hybrid_multiple_networks/losses",
            dataset,
            f"bvalue_{settings.data.bvalue}",
            f"num_volumes_{settings.data.num_volumes}",
            f"noise_sigma_{settings.data.noise_sigma}",
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(loss_dir, exist_ok=True)

        # One dataset instance; set_source_volume_index(i) before each volume's training
        train_set = TrainingDataSet(
            data=noisy_data,
            patch_size=(
                settings.data.num_volumes,
                settings.data.patch_size,
                settings.data.patch_size,
                settings.data.patch_size,
            ),
            step=settings.data.step,
            mask_p=settings.train.mask_p,
            source_volume_index=0,
        )

        multi_gpu_config = create_multi_gpu_config_from_dict(
            {
                "multi_gpu": settings.train.multi_gpu,
                "gpu_ids": settings.train.gpu_ids,
                "auto_scale_lr": settings.train.auto_scale_lr,
                "learning_rate": settings.train.learning_rate,
                "batch_size": settings.train.batch_size,
                "auto_exclude_imbalanced": settings.train.auto_exclude_imbalanced,
                "memory_threshold": settings.train.memory_threshold,
            }
        )

        if train:
            for volume_idx in range(settings.data.num_volumes):
                logging.info(
                    f"Training model for volume {volume_idx}/{settings.data.num_volumes}"
                )
                train_set.set_source_volume_index(volume_idx)
                train_loader = DataLoader(
                    train_set,
                    batch_size=settings.train.batch_size,
                    shuffle=True,
                )
                logging.info(
                    f"DataLoader for volume {volume_idx}: batch_size={settings.train.batch_size}, num_batches={len(train_loader)}"
                )
                model = DenoiserNet(
                    input_channels=settings.model.in_channel,
                    output_channels=settings.model.out_channel,
                    groups=settings.model.groups,
                    dense_convs=settings.model.dense_convs,
                    residual=settings.model.residual,
                    base_filters=settings.model.base_filters,
                    output_shape=(
                        settings.model.out_channel,
                        settings.data.patch_size,
                        settings.data.patch_size,
                        settings.data.patch_size,
                    ),
                    device=settings.train.device,
                )
                model, effective_lr, effective_batch_size = setup_multi_gpu(
                    model, multi_gpu_config
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
                scheduler = None
                if settings.train.use_scheduler:
                    if settings.train.scheduler_type == "step":
                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=settings.train.scheduler_step_size,
                            gamma=settings.train.scheduler_gamma,
                        )
                    elif settings.train.scheduler_type == "cosine":
                        scheduler = (
                            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer,
                                T_0=settings.train.scheduler_T_0,
                                T_mult=settings.train.scheduler_T_mult,
                                eta_min=settings.train.eta_min_lr,
                            )
                        )
                    elif settings.train.scheduler_type == "reduceLROnPlateau":
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            patience=settings.train.scheduler_patience,
                            factor=settings.train.scheduler_factor,
                            min_lr=settings.train.min_lr,
                        )
                volume_checkpoint_dir = os.path.join(
                    checkpoint_dir, f"volume_{volume_idx}"
                )
                volume_loss_dir = os.path.join(loss_dir, f"volume_{volume_idx}")
                os.makedirs(volume_checkpoint_dir, exist_ok=True)
                os.makedirs(volume_loss_dir, exist_ok=True)
                fit_model(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loader=train_loader,
                    num_epochs=settings.train.num_epochs,
                    device=settings.train.device,
                    checkpoint_dir=volume_checkpoint_dir,
                    loss_dir=volume_loss_dir,
                )
                del model
                if settings.train.device != "cpu":
                    torch.cuda.empty_cache()
                logging.info(f"Completed training for volume {volume_idx}")

                # TODO: remove break after testing
                # break
            logging.info(f"Training completed. Log file: {log_file}")

        if reconstruct:
            logging.info("Reconstructing DWIs (one model per volume)...")
            x_reconstruct = torch.from_numpy(
                np.transpose(noisy_data, (3, 0, 1, 2))
            ).type(torch.float)
            num_vols = settings.data.num_volumes
            reconstructed_dwis = np.zeros(
                (
                    num_vols,
                    settings.data.take_x,
                    settings.data.take_y,
                    settings.data.take_z,
                ),
                dtype=np.float32,
            )
            for vol_idx in tqdm(range(num_vols), desc="Reconstructing volumes"):
                volume_checkpoint = os.path.join(
                    checkpoint_dir, f"volume_{vol_idx}", "best_loss_checkpoint.pth"
                )
                if not os.path.isfile(volume_checkpoint):
                    logging.warning(
                        f"Checkpoint not found for volume {vol_idx}: {volume_checkpoint}; skipping."
                    )
                    continue
                model = DenoiserNet(
                    input_channels=settings.model.in_channel,
                    output_channels=settings.model.out_channel,
                    groups=settings.model.groups,
                    dense_convs=settings.model.dense_convs,
                    residual=settings.model.residual,
                    base_filters=settings.model.base_filters,
                    output_shape=(
                        settings.model.out_channel,
                        settings.data.take_x,
                        settings.data.take_y,
                        settings.data.take_z,
                    ),
                    device=settings.train.device,
                )
                load_checkpoint(
                    model=model,
                    optimizer=None,
                    filename=volume_checkpoint,
                    device=settings.reconstruct.device,
                    strict=False,
                )
                reconstructed_dwis[vol_idx] = reconstruct_single_volume(
                    model=model,
                    data=x_reconstruct,
                    vol_idx=vol_idx,
                    device=settings.reconstruct.device,
                    mask_p=settings.reconstruct.mask_p,
                    n_preds=settings.reconstruct.n_preds,
                )
                del model
                if settings.reconstruct.device != "cpu":
                    torch.cuda.empty_cache()
                # TODO: remove after testing
                # break
            # (Vols, X, Y, Z) -> (X, Y, Z, Vols) for metrics and visualization
            reconstructed_dwis = np.transpose(reconstructed_dwis, (1, 2, 3, 0))
            logging.info(f"Reconstructed DWIs shape: {reconstructed_dwis.shape}")
            logging.info(
                f"Reconstructed DWIs min: {reconstructed_dwis.min():.4f}, "
                f"max: {reconstructed_dwis.max():.4f}, "
                f"mean: {reconstructed_dwis.mean():.4f}"
            )
            logging.info(f"Reconstructed DWIs dtype: {reconstructed_dwis.dtype}")

            # Optional: clip to [0, 1] so metrics match original range (avoids background lift)
            if getattr(settings.reconstruct, "clip_to_range", False):
                reconstructed_dwis = np.clip(reconstructed_dwis, 0.0, 1.0)
                logging.info(
                    "Clipped reconstruction to [0, 1]. "
                    "Reconstructed min/max/mean: "
                    f"{reconstructed_dwis.min():.4f}, {reconstructed_dwis.max():.4f}, {reconstructed_dwis.mean():.4f}"
                )

            # Full-image metrics (background voxels can dominate and worsen PSNR/SSIM)
            metrics = compute_metrics(original_data, reconstructed_dwis)
            logging.info(f"Metrics (full image): {metrics}")

            # ROI metrics: only over voxels where original > threshold (excludes air/background)
            roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
            if roi_threshold is not None:
                roi_mask = (original_data > roi_threshold).any(axis=-1)
                n_roi = int(np.sum(roi_mask))
                logging.info(
                    f"ROI mask: original > {roi_threshold}, {n_roi} voxels ({100.0 * n_roi / roi_mask.size:.1f}%)"
                )
                metrics_roi = compute_metrics(
                    original_data, reconstructed_dwis, mask=roi_mask
                )
                logging.info(f"Metrics (ROI, brain/tissue only): {metrics_roi}")
            else:
                metrics_roi = None

            # Log metrics to wandb
            if wandb_run is not None:
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
            # setting metrics dir taking into account run/model parameters
            metrics_dir = os.path.join(
                settings.reconstruct.metrics_dir,
                f"bvalue_{settings.data.bvalue}",
                f"num_volumes_{settings.data.num_volumes}",
                f"noise_sigma_{settings.data.noise_sigma}",
                f"learning_rate_{settings.train.learning_rate}",
            )
            os.makedirs(metrics_dir, exist_ok=True)
            save_metrics(metrics, metrics_dir)
            if metrics_roi is not None:
                save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")

            if generate_images:
                logging.info("Generating images...")
                # setting images dir taking into account run/model parameters
                images_dir = os.path.join(
                    settings.reconstruct.images_dir,
                    f"bvalue_{settings.data.bvalue}",
                    f"num_volumes_{settings.data.num_volumes}",
                    f"noise_sigma_{settings.data.noise_sigma}",
                    f"learning_rate_{settings.train.learning_rate}",
                )
                os.makedirs(images_dir, exist_ok=True)
                logging.info(f"Saving images to: {images_dir}")

                # Generate comparison image
                wandb_images = []
                for i in range(settings.data.num_volumes):
                    comparison_path = os.path.join(
                        images_dir, f"comparison_volume_{i}.png"
                    )
                    fully_compare_volumes(
                        original_volume=np.transpose(original_data, (2, 3, 0, 1)),
                        noisy_volume=np.transpose(noisy_data, (2, 3, 0, 1)),
                        denoised_volume=np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                        file_name=comparison_path,
                        volume_idx=i,
                    )
                    wandb_images.append(
                        wandb.Image(comparison_path, caption=f"Volume index {i}")
                    )
                # Log images to wandb
                if wandb_run is not None:
                    wandb.log(
                        {
                            "reconstruct/comparison": wandb_images,
                        }
                    )

                # Generate single volume images
                single_path = os.path.join(images_dir, "single.png")
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=single_path,
                    volume_idx=0,
                    slice_idx=10,
                )
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=None,
                    volume_idx=0,
                    slice_idx=20,
                )
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=None,
                    volume_idx=0,
                    slice_idx=30,
                )
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=None,
                    volume_idx=0,
                    slice_idx=40,
                )
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=None,
                    volume_idx=0,
                    slice_idx=50,
                )
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=None,
                    volume_idx=0,
                    slice_idx=60,
                )
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=None,
                    volume_idx=0,
                    slice_idx=70,
                )

                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (2, 3, 0, 1)),
                    file_name=single_path,
                    volume_idx=0,
                )

                noisy_path = os.path.join(images_dir, "noisy.png")
                visualize_single_volume(
                    np.transpose(noisy_data, (2, 3, 0, 1)),
                    file_name=noisy_path,
                    volume_idx=0,
                )

    finally:
        # Ensure wandb run is always finished, even if an exception occurs
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main(dataset="dbrain")
