import logging
import os

import numpy as np
import torch
from drcnet_s2s.data import TrainingDataSet
from drcnet_s2s.fit import fit_model
from drcnet_s2s.model import DenoiserNet
from drcnet_s2s.reconstruction import reconstruct_dwis
from torch.utils.data import DataLoader
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
):
    """Train and/or reconstruct with DRCNet-S2S: single volume with masked pixels, denoise by volume. Data layout: (Z, Vols, X, Y)."""
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
                "model_name": "DRCNet-s2s",
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

        train_set = TrainingDataSet(
            data=noisy_data,
            patch_size=(
                settings.data.patch_size,
                settings.data.num_volumes,
                settings.data.patch_size,
                settings.data.patch_size,
            ),
            step=settings.data.step,
            mask_p=settings.train.mask_p,
        )
        train_loader = DataLoader(
            train_set, batch_size=settings.train.batch_size, shuffle=True
        )
        logging.info(
            f"DataLoader created with batch_size={settings.train.batch_size}, num_batches={len(train_loader)}"
        )
        logging.info("Initializing DenoiserNet model...")
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
        logging.info(
            f"Model initialized - in_channel: {settings.model.in_channel}, out_channel: {settings.model.out_channel}"
        )
        logging.info(
            f"Total model parameters: {sum(p.numel() for p in model.parameters())}"
        )

        # Setup multi-GPU training
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

        model, effective_lr, effective_batch_size = setup_multi_gpu(
            model, multi_gpu_config
        )

        logging.info("Setting up optimizer and scheduler...")
        optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
        logging.info(f"Optimizer: Adam(lr={effective_lr:.6f})")
        logging.info(
            f"Effective batch size: {effective_batch_size} (per-GPU: {settings.train.batch_size})"
        )

        scheduler = None
        if settings.train.use_scheduler:
            if settings.train.scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=settings.train.scheduler_step_size,
                    gamma=settings.train.scheduler_gamma,
                )
                logging.info(
                    f"Scheduler: StepLR(step_size={settings.train.scheduler_step_size}, "
                    f"gamma={settings.train.scheduler_gamma})"
                )
            elif settings.train.scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=settings.train.scheduler_T_0,
                    T_mult=settings.train.scheduler_T_mult,
                    eta_min=settings.train.eta_min_lr,
                )
                logging.info(
                    f"Scheduler: CosineAnnealingWarmRestarts(T_0={settings.train.scheduler_T_0}, T_mult={settings.train.scheduler_T_mult}, eta_min={settings.train.eta_min_lr})"
                )
            elif settings.train.scheduler_type == "reduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=settings.train.scheduler_patience,
                    factor=settings.train.scheduler_factor,
                    min_lr=settings.train.min_lr,
                )
                logging.info(
                    f"Scheduler: ReduceLROnPlateau(patience={settings.train.scheduler_patience}, factor={settings.train.scheduler_factor}, min_lr={settings.train.min_lr})"
                )

        logging.info(f"Training device: {settings.train.device}")
        logging.info(f"Number of epochs: {settings.train.num_epochs}")
        logging.info(f"Checkpoint directory: {settings.train.checkpoint_dir}")

        # setting checkpoint dir taking into account run/model parameters
        checkpoint_dir = os.path.join(
            settings.train.checkpoint_dir,
            f"bvalue_{settings.data.bvalue}",
            f"num_volumes_{settings.data.num_volumes}",
            f"noise_sigma_{settings.data.noise_sigma}",
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # setting loss dir taking into account run/model parameters
        loss_dir = os.path.join(
            "drcnet_s2s/losses",
            dataset,
            f"bvalue_{settings.data.bvalue}",
            f"num_volumes_{settings.data.num_volumes}",
            f"noise_sigma_{settings.data.noise_sigma}",
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(loss_dir, exist_ok=True)

        # Training
        if train:
            fit_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                num_epochs=settings.train.num_epochs,
                device=settings.train.device,
                checkpoint_dir=checkpoint_dir,
                loss_dir=loss_dir,
            )

            logging.info("Training setup completed successfully")
            logging.info(f"Training completed. Log file: {log_file}")

        if reconstruct:
            logging.info("Reconstructing DWIs...")
            best_loss_checkpoint = os.path.join(
                checkpoint_dir, "best_loss_checkpoint.pth"
            )
            del model
            reconstruct_model = DenoiserNet(
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
            reconstruct_model, _, _, _, _ = load_checkpoint(
                model=reconstruct_model,
                optimizer=None,
                filename=best_loss_checkpoint,
                device=settings.reconstruct.device,
                strict=False,  # Allow partial loading for architecture changes
            )
            # Prepare data for reconstruction
            x_reconstruct = torch.from_numpy(noisy_data).type(torch.float)

            reconstructed_dwis = reconstruct_dwis(
                model=reconstruct_model,
                data=x_reconstruct,
                device=settings.reconstruct.device,
                mask_p=settings.reconstruct.mask_p,
                n_preds=settings.reconstruct.n_preds,
            )
            # reconstructed_dwis is (Vols, X, Y, Z); for compute_metrics we transpose to (X, Y, Z, Vols)
            logging.info(f"Reconstructed DWIs shape: {reconstructed_dwis.shape}")
            logging.info(
                f"Reconstructed DWIs min: {reconstructed_dwis.min():.4f}, "
                f"max: {reconstructed_dwis.max():.4f}, "
                f"mean: {reconstructed_dwis.mean():.4f}"
            )
            logging.info(f"Reconstructed DWIs dtype: {reconstructed_dwis.dtype}")

            metrics = compute_metrics(
                original_data,
                np.transpose(
                    reconstructed_dwis, (1, 2, 3, 0)
                ),  # (Vols, X, Y, Z) -> (X, Y, Z, Vols)
            )
            logging.info(f"Metrics: {metrics}")
            # Log metrics to wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        "reconstruct/metrics_mse": metrics["mse"],
                        "reconstruct/metrics_ssim": metrics["ssim"],
                        "reconstruct/metrics_psnr": metrics["psnr"],
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
                        denoised_volume=np.transpose(
                            reconstructed_dwis, (3, 0, 1, 2)
                        ),  # (Vols, X, Y, Z) -> (Z, Vols, X, Y)
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

                # Generate single volume images (fully_compare_volumes/visualize expect (Z, Vols, X, Y))
                single_path = os.path.join(images_dir, "single.png")
                visualize_single_volume(
                    np.transpose(reconstructed_dwis, (3, 0, 1, 2)),
                    file_name=single_path,
                    volume_idx=0,
                )

                noisy_path = os.path.join(images_dir, "noisy.png")
                visualize_single_volume(
                    np.transpose(noisy_data, (3, 0, 1, 2)),
                    file_name=noisy_path,
                    volume_idx=0,
                )

    finally:
        # Ensure wandb run is always finished, even if an exception occurs
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main(dataset="dbrain")
