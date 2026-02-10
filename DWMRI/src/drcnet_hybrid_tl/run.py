import logging
import os

import numpy as np
import torch
from drcnet_hybrid_tl.data import TrainingDataSet
from drcnet_hybrid_tl.fit import fit_model, fit_transfer_wrapper
from drcnet_hybrid_tl.model import DenoiserNet, DenoiserNetTransferWrapper
from drcnet_hybrid_tl.reconstruction import reconstruct_dwis, reconstruct_single_volume
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
                "model_name": "DRCNet-hybrid-TL",
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
                settings.data.num_volumes,
                settings.data.patch_size,
                settings.data.patch_size,
                settings.data.patch_size,
            ),
            step=settings.data.step,
            mask_p=settings.train.mask_p,
            source_volume_index=0,
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
            "drcnet_hybrid_tl/losses",
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

        if transfer_learn and hasattr(settings, "transfer"):
            logging.info("Starting transfer learning phase...")
            target_indices = settings.transfer.target_volume_indices
            if target_indices == "all":
                target_indices = list(range(1, settings.data.num_volumes))
            elif not isinstance(target_indices, list):
                target_indices = list(range(1, settings.data.num_volumes))

            best_loss_checkpoint = os.path.join(
                checkpoint_dir, "best_loss_checkpoint.pth"
            )
            if not os.path.isfile(best_loss_checkpoint):
                logging.warning(
                    "Phase 1 best checkpoint not found; skipping transfer. Run Phase 1 training first."
                )
            else:
                for target_idx in target_indices:
                    if target_idx >= settings.data.num_volumes:
                        logging.warning(
                            f"Skipping transfer target_volume_index={target_idx} (>= num_volumes)"
                        )
                        continue
                    logging.info(
                        f"Transfer learning for target volume index {target_idx}"
                    )
                    # New base model and wrapper per target volume
                    base_model = DenoiserNet(
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
                    wrapper = DenoiserNetTransferWrapper(
                        base_model=base_model,
                        adapter_in_ch=settings.model.in_channel,
                        adapter_out_ch=1,
                    )
                    wrapper.load_base_checkpoint(
                        best_loss_checkpoint, device=settings.train.device
                    )
                    if getattr(settings.transfer, "freeze_base", True):
                        for p in wrapper.base_model.parameters():
                            p.requires_grad = False

                    transfer_train_set = TrainingDataSet(
                        data=noisy_data,
                        patch_size=(
                            settings.data.num_volumes,
                            settings.data.patch_size,
                            settings.data.patch_size,
                            settings.data.patch_size,
                        ),
                        step=settings.data.step,
                        mask_p=settings.train.mask_p,
                        source_volume_index=target_idx,
                    )
                    transfer_loader = DataLoader(
                        transfer_train_set,
                        batch_size=settings.train.batch_size,
                        shuffle=True,
                    )
                    adapter_params = list(wrapper.input_adapter.parameters()) + list(
                        wrapper.output_adapter.parameters()
                    )
                    transfer_optimizer = torch.optim.Adam(
                        adapter_params, lr=settings.transfer.learning_rate
                    )
                    transfer_scheduler = None
                    if getattr(settings.train, "use_scheduler", False):
                        if settings.train.scheduler_type == "cosine":
                            transfer_scheduler = (
                                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                    transfer_optimizer,
                                    T_0=getattr(
                                        settings.train,
                                        "scheduler_T_0",
                                        20,
                                    ),
                                    T_mult=getattr(
                                        settings.train,
                                        "scheduler_T_mult",
                                        2,
                                    ),
                                    eta_min=getattr(
                                        settings.train,
                                        "eta_min_lr",
                                        0.0001,
                                    ),
                                )
                            )

                    transfer_checkpoint_dir = os.path.join(
                        checkpoint_dir, f"transfer_vol_{target_idx}"
                    )
                    transfer_loss_dir = os.path.join(
                        loss_dir, f"transfer_vol_{target_idx}"
                    )
                    os.makedirs(transfer_checkpoint_dir, exist_ok=True)
                    os.makedirs(transfer_loss_dir, exist_ok=True)

                    fit_transfer_wrapper(
                        model=wrapper,
                        optimizer=transfer_optimizer,
                        scheduler=transfer_scheduler,
                        train_loader=transfer_loader,
                        num_epochs=settings.transfer.num_epochs,
                        device=settings.train.device,
                        checkpoint_dir=transfer_checkpoint_dir,
                        loss_dir=transfer_loss_dir,
                    )
                    logging.info(
                        f"Transfer completed for target volume index {target_idx}"
                    )
        elif transfer_learn and not hasattr(settings, "transfer"):
            logging.warning(
                "transfer_learn=True but no 'transfer' section in config; skipping."
            )

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
            # Prepare data for reconstruction: transpose from (X, Y, Z, Vols) to (Vols, X, Y, Z)
            x_reconstruct = torch.from_numpy(
                np.transpose(noisy_data, (3, 0, 1, 2))
            ).type(torch.float)

            num_vols = settings.data.num_volumes
            has_transfer = any(
                os.path.isfile(
                    os.path.join(
                        checkpoint_dir,
                        f"transfer_vol_{v}",
                        "best_loss_checkpoint.pth",
                    )
                )
                for v in range(1, num_vols)
            )
            if has_transfer:
                # Option A: base for vol 0, wrapper_v for vol v (1..N-1)
                logging.info(
                    "Using transfer reconstruction (base for vol 0, wrapper per volume for 1..N-1)"
                )
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
                    if vol_idx == 0:
                        use_model = reconstruct_model
                    else:
                        transfer_path = os.path.join(
                            checkpoint_dir,
                            f"transfer_vol_{vol_idx}",
                            "best_loss_checkpoint.pth",
                        )
                        if os.path.isfile(transfer_path):
                            base_copy = DenoiserNet(
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
                            wrapper = DenoiserNetTransferWrapper(
                                base_model=base_copy,
                                adapter_in_ch=settings.model.in_channel,
                                adapter_out_ch=1,
                            )
                            wrapper.load_base_checkpoint(
                                best_loss_checkpoint,
                                device=settings.reconstruct.device,
                            )
                            load_checkpoint(
                                model=wrapper,
                                optimizer=None,
                                filename=transfer_path,
                                device=settings.reconstruct.device,
                                strict=False,
                            )
                            use_model = wrapper
                        else:
                            use_model = reconstruct_model
                    reconstructed_dwis[vol_idx] = reconstruct_single_volume(
                        model=use_model,
                        data=x_reconstruct,
                        vol_idx=vol_idx,
                        device=settings.reconstruct.device,
                        mask_p=settings.reconstruct.mask_p,
                        n_preds=settings.reconstruct.n_preds,
                    )
                # (Vols, X, Y, Z) -> (X, Y, Z, Vols) for metrics
                reconstructed_dwis = np.transpose(reconstructed_dwis, (1, 2, 3, 0))
            else:
                reconstructed_dwis = reconstruct_dwis(
                    model=reconstruct_model,
                    data=x_reconstruct,
                    device=settings.reconstruct.device,
                    mask_p=settings.reconstruct.mask_p,
                    n_preds=settings.reconstruct.n_preds,
                )
                # Transpose back to (X, Y, Z, Vols) for metrics and visualization
                reconstructed_dwis = np.transpose(reconstructed_dwis, (1, 2, 3, 0))
            logging.info(f"Reconstructed DWIs shape: {reconstructed_dwis.shape}")
            logging.info(
                f"Reconstructed DWIs min: {reconstructed_dwis.min():.4f}, "
                f"max: {reconstructed_dwis.max():.4f}, "
                f"mean: {reconstructed_dwis.mean():.4f}"
            )
            logging.info(f"Reconstructed DWIs dtype: {reconstructed_dwis.dtype}")

            metrics = compute_metrics(
                original_data,
                reconstructed_dwis,
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
