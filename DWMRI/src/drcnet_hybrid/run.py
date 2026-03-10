import logging
import os
import shutil

import numpy as np
import torch
from drcnet_hybrid.data import TrainingDataSet
from drcnet_hybrid.fit import fit_model
from drcnet_hybrid.model import DenoiserNet
from drcnet_hybrid.reconstruction import reconstruct_dwis
from torch.utils.data import DataLoader, Subset
from utils import setup_logging
from utils.checkpoint import load_checkpoint
from utils.data import DBrainDataLoader, StanfordDataLoader, compute_brain_mask
from utils.metrics import (
    compute_metrics,
    fully_compare_volumes,
    save_metrics,
    visualize_single_volume,
)
from utils.multi_gpu import create_multi_gpu_config_from_dict, setup_multi_gpu
from utils.utils import load_config, noise_path_segment
import wandb


def fit_progressive(
    model,
    settings,
    noisy_data,
    original_data,
    brain_mask,
    checkpoint_dir,
    loss_dir,
    patch_filter_method,
    min_signal_threshold,
):
    """
    Train model progressively with increasing patch sizes.

    Progressive learning stages use per-stage patch_size, step, batch_size, and
    epochs; each stage gets a new dataset, optimizer, and scheduler.
    """
    stages = settings.train.progressive.stages
    total_stages = len(stages)
    subset_seed = getattr(settings.train, "seed", 42)

    logging.info(f"Progressive Learning: {total_stages} stages")
    for i, stage in enumerate(stages):
        logging.info(
            f"  Stage {i+1}: patch={stage.patch_size}³, batch={stage.batch_size}, "
            f"epochs={stage.epochs}, step={stage.step}"
        )

    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        logging.info("=" * 60)
        logging.info(f"PROGRESSIVE STAGE {stage_num}/{total_stages}")
        logging.info(f"  Patch size: {stage.patch_size}³")
        logging.info(f"  Batch size: {stage.batch_size}")
        logging.info(f"  Epochs: {stage.epochs}")
        logging.info(f"  Step: {stage.step}")
        logging.info("=" * 60)

        if settings.train.device == "cuda":
            torch.cuda.empty_cache()
            logging.info("Cleared GPU cache before stage")

        train_set = TrainingDataSet(
            data=noisy_data,
            patch_size=(
                settings.data.num_volumes,
                stage.patch_size,
                stage.patch_size,
                stage.patch_size,
            ),
            step=stage.step,
            mask_p=settings.train.mask_p,
            clean_data=original_data,
            brain_mask=brain_mask,
            patch_filter_method=patch_filter_method,
            min_signal_threshold=min_signal_threshold,
        )

        subset_fraction = 0.1
        total_samples = len(train_set)
        num_samples = int(total_samples * subset_fraction)
        np.random.seed(subset_seed)
        indices = np.random.choice(total_samples, size=num_samples, replace=False)
        train_set = Subset(train_set, indices)

        train_loader = DataLoader(
            train_set, batch_size=stage.batch_size, shuffle=True
        )
        logging.info(
            f"Stage {stage_num} DataLoader: batch_size={stage.batch_size}, "
            f"num_batches={len(train_loader)}, samples={len(train_set)}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=settings.train.learning_rate
        )
        logging.info(
            f"Stage {stage_num} Optimizer: Adam(lr={settings.train.learning_rate})"
        )

        scheduler = None
        if getattr(settings.train, "use_scheduler", False):
            if settings.train.scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=settings.train.scheduler_step_size,
                    gamma=settings.train.scheduler_gamma,
                )
            elif settings.train.scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=min(
                        getattr(settings.train, "scheduler_T_0", 20), stage.epochs
                    ),
                    T_mult=getattr(settings.train, "scheduler_T_mult", 2),
                    eta_min=getattr(settings.train, "eta_min_lr", 0.0001),
                )
            elif settings.train.scheduler_type == "reduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=settings.train.scheduler_patience,
                    factor=settings.train.scheduler_factor,
                    min_lr=settings.train.min_lr,
                )

        stage_checkpoint_dir = os.path.join(
            checkpoint_dir, f"stage_{stage_num}_patch{stage.patch_size}"
        )
        os.makedirs(stage_checkpoint_dir, exist_ok=True)
        stage_loss_dir = os.path.join(
            loss_dir, f"stage_{stage_num}_patch{stage.patch_size}"
        )
        os.makedirs(stage_loss_dir, exist_ok=True)

        fit_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            num_epochs=stage.epochs,
            device=settings.train.device,
            checkpoint_dir=stage_checkpoint_dir,
            loss_dir=stage_loss_dir,
        )

        logging.info(f"Stage {stage_num}/{total_stages} completed")

        if wandb.run is not None:
            wandb.log(
                {
                    "progressive/stage": stage_num,
                    "progressive/patch_size": stage.patch_size,
                    "progressive/batch_size": stage.batch_size,
                }
            )

        if stage_num == total_stages:
            stage_best = os.path.join(
                stage_checkpoint_dir, "best_loss_checkpoint.pth"
            )
            final_best = os.path.join(checkpoint_dir, "best_loss_checkpoint.pth")
            if os.path.exists(stage_best):
                shutil.copy(stage_best, final_best)
                logging.info(f"Copied final stage best checkpoint to: {final_best}")

    logging.info("Progressive learning completed successfully")


def main(
    dataset: str,
    train: bool = True,
    reconstruct: bool = True,
    generate_images: bool = True,
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
            noise_type=getattr(settings.data, "noise_type", "rician"),
            n_coils=getattr(settings.data, "noise_n_coils", 1),
        )
        logging.info(
            f"DBrainDataLoader initialized with noise_sigma={settings.data.noise_sigma}, noise_type={getattr(settings.data, 'noise_type', 'rician')}"
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
                "model_name": "DRCNet-hybrid",
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

        # Patch filtering configuration
        patch_filter_method = getattr(settings.data, "patch_filter_method", "none")
        min_signal_threshold = getattr(settings.data, "min_signal_threshold", 0.0)
        otsu_median_radius = getattr(settings.data, "otsu_median_radius", 2)
        otsu_numpass = getattr(settings.data, "otsu_numpass", 1)

        logging.info(
            f"Patch filtering: method={patch_filter_method}, "
            f"threshold={min_signal_threshold}, otsu_radius={otsu_median_radius}, otsu_numpass={otsu_numpass}"
        )

        # Compute brain mask if using otsu method
        brain_mask = None
        if patch_filter_method == "otsu":
            logging.info("Computing brain mask using median_otsu...")
            brain_mask = compute_brain_mask(
                original_data,
                median_radius=otsu_median_radius,
                numpass=otsu_numpass,
            )

        progressive_enabled = hasattr(settings.train, "progressive") and getattr(
            settings.train.progressive, "enabled", False
        )
        noise_segment = noise_path_segment(
            getattr(settings.data, "noise_type", "rician"),
            getattr(settings.data, "noise_sigma", 0.1),
        )
        checkpoint_dir = os.path.join(
            settings.train.checkpoint_dir,
            f"bvalue_{settings.data.bvalue}",
            f"num_volumes_{settings.data.num_volumes}",
            noise_segment,
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_dir = os.path.join(
            "drcnet_hybrid/losses",
            dataset,
            f"bvalue_{settings.data.bvalue}",
            f"num_volumes_{settings.data.num_volumes}",
            noise_segment,
            f"learning_rate_{settings.train.learning_rate}",
        )
        os.makedirs(loss_dir, exist_ok=True)

        if train:
            if progressive_enabled:
                logging.info("Using progressive learning training strategy")
                first_stage = settings.train.progressive.stages[0]
                logging.info("Initializing DenoiserNet model (first stage patch size)...")
                model = DenoiserNet(
                    input_channels=settings.model.in_channel,
                    output_channels=settings.model.out_channel,
                    groups=settings.model.groups,
                    dense_convs=settings.model.dense_convs,
                    residual=settings.model.residual,
                    base_filters=settings.model.base_filters,
                    output_shape=(
                        settings.model.out_channel,
                        first_stage.patch_size,
                        first_stage.patch_size,
                        first_stage.patch_size,
                    ),
                    device=settings.train.device,
                    output_activation=getattr(
                        settings.model, "output_activation", "prelu"
                    ),
                )
                logging.info(
                    f"Model initialized - in_channel: {settings.model.in_channel}, "
                    f"out_channel: {settings.model.out_channel}"
                )
                logging.info(
                    f"Total model parameters: {sum(p.numel() for p in model.parameters())}"
                )
                multi_gpu_config = create_multi_gpu_config_from_dict(
                    {
                        "multi_gpu": settings.train.multi_gpu,
                        "gpu_ids": settings.train.gpu_ids,
                        "auto_scale_lr": settings.train.auto_scale_lr,
                        "learning_rate": settings.train.learning_rate,
                        "batch_size": first_stage.batch_size,
                        "auto_exclude_imbalanced": settings.train.auto_exclude_imbalanced,
                        "memory_threshold": settings.train.memory_threshold,
                    }
                )
                model, _, _ = setup_multi_gpu(model, multi_gpu_config)
                fit_progressive(
                    model=model,
                    settings=settings,
                    noisy_data=noisy_data,
                    original_data=original_data,
                    brain_mask=brain_mask,
                    checkpoint_dir=checkpoint_dir,
                    loss_dir=loss_dir,
                    patch_filter_method=patch_filter_method,
                    min_signal_threshold=min_signal_threshold,
                )
            else:
                logging.info("Using standard training (progressive learning disabled)")
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
                    clean_data=original_data,
                    brain_mask=brain_mask,
                    patch_filter_method=patch_filter_method,
                    min_signal_threshold=min_signal_threshold,
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=settings.train.batch_size,
                    shuffle=True,
                )
                logging.info(
                    f"DataLoader created with batch_size={settings.train.batch_size}, "
                    f"num_batches={len(train_loader)}"
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
                    output_activation=getattr(
                        settings.model, "output_activation", "prelu"
                    ),
                )
                logging.info(
                    f"Model initialized - in_channel: {settings.model.in_channel}, "
                    f"out_channel: {settings.model.out_channel}"
                )
                logging.info(
                    f"Total model parameters: {sum(p.numel() for p in model.parameters())}"
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
                model, effective_lr, effective_batch_size = setup_multi_gpu(
                    model, multi_gpu_config
                )
                logging.info("Setting up optimizer and scheduler...")
                optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
                logging.info(f"Optimizer: Adam(lr={effective_lr:.6f})")
                logging.info(
                    f"Effective batch size: {effective_batch_size} "
                    f"(per-GPU: {settings.train.batch_size})"
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
                            f"Scheduler: CosineAnnealingWarmRestarts(T_0={settings.train.scheduler_T_0}, "
                            f"T_mult={settings.train.scheduler_T_mult}, eta_min={settings.train.eta_min_lr})"
                        )
                    elif settings.train.scheduler_type == "reduceLROnPlateau":
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            patience=settings.train.scheduler_patience,
                            factor=settings.train.scheduler_factor,
                            min_lr=settings.train.min_lr,
                        )
                        logging.info(
                            f"Scheduler: ReduceLROnPlateau(patience={settings.train.scheduler_patience}, "
                            f"factor={settings.train.scheduler_factor}, min_lr={settings.train.min_lr})"
                        )
                logging.info(f"Training device: {settings.train.device}")
                logging.info(f"Number of epochs: {settings.train.num_epochs}")
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
                output_activation=getattr(settings.model, "output_activation", "prelu"),
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

            # Optional: subtract estimated background level then clip
            if getattr(settings.reconstruct, "subtract_background_estimate", False):
                thresh = getattr(
                    settings.reconstruct, "subtract_background_threshold", 0.02
                )
                bg_mask = (original_data <= thresh).all(axis=-1)
                if np.any(bg_mask):
                    bg_vals = reconstructed_dwis[bg_mask]
                    shift = float(np.median(bg_vals))
                    logging.info(
                        f"Background subtraction: shift={shift:.6f} from {np.sum(bg_mask):,} voxels"
                    )
                    reconstructed_dwis = reconstructed_dwis.astype(np.float64) - shift
                    reconstructed_dwis = np.clip(reconstructed_dwis, 0, 1)

            # Optional: clip to [0, 1] range
            if getattr(settings.reconstruct, "clip_to_range", False):
                reconstructed_dwis = np.clip(reconstructed_dwis, 0, 1)
                logging.info(
                    f"Clipped to [0, 1]: min={reconstructed_dwis.min():.4f}, "
                    f"max={reconstructed_dwis.max():.4f}, mean={reconstructed_dwis.mean():.4f}"
                )

            # setting metrics dir taking into account run/model parameters
            metrics_dir = os.path.join(
                settings.reconstruct.metrics_dir,
                f"bvalue_{settings.data.bvalue}",
                f"num_volumes_{settings.data.num_volumes}",
                noise_segment,
                f"learning_rate_{settings.train.learning_rate}",
            )
            os.makedirs(metrics_dir, exist_ok=True)

            # Full-image metrics
            metrics = compute_metrics(
                original_data,
                reconstructed_dwis,
            )
            logging.info(f"Metrics (full image): {metrics}")
            save_metrics(metrics, metrics_dir)

            # Log metrics to wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        "reconstruct/metrics_mse": metrics["mse"],
                        "reconstruct/metrics_ssim": metrics["ssim"],
                        "reconstruct/metrics_psnr": metrics["psnr"],
                    }
                )

            # ROI-based metrics (brain/tissue only)
            roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
            if roi_threshold is not None:
                roi_mask = (original_data > roi_threshold).any(axis=-1)
                n_roi = int(np.sum(roi_mask))
                logging.info(
                    f"ROI mask: original > {roi_threshold}, {n_roi:,} voxels ({100.0 * n_roi / roi_mask.size:.1f}%)"
                )
                metrics_roi = compute_metrics(
                    original_data, reconstructed_dwis, mask=roi_mask
                )
                logging.info(f"Metrics (ROI, brain/tissue only): {metrics_roi}")
                save_metrics(metrics_roi, metrics_dir, filename="metrics_roi.json")

                if wandb_run is not None:
                    wandb.log(
                        {
                            "reconstruct/metrics_roi_mse": metrics_roi["mse"],
                            "reconstruct/metrics_roi_ssim": metrics_roi["ssim"],
                            "reconstruct/metrics_roi_psnr": metrics_roi["psnr"],
                        }
                    )

            if generate_images:
                logging.info("Generating images...")
                # setting images dir taking into account run/model parameters
                images_dir = os.path.join(
                    settings.reconstruct.images_dir,
                    f"bvalue_{settings.data.bvalue}",
                    f"num_volumes_{settings.data.num_volumes}",
                    noise_segment,
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
