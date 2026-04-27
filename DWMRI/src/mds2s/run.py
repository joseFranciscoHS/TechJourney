import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mds2s.fit import fit_model
from mds2s.model import Self2self
from mds2s.reconstruction import reconstruct_dwis
from paper_eval.dti_metrics import save_dti_metrics, try_compute_dti_errors
from utils import setup_logging
from utils.checkpoint import load_checkpoint
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

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency in batch/smoke runs
    wandb = None


def main(
    dataset: str,
    train: bool = True,
    reconstruct: bool = True,
    generate_images: bool = True,
    use_wandb: bool = True,
    seed_override: int | None = None,
    reproducible_override: bool | None = None,
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
        data_loader = StanfordDataLoader(
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
        )
        logging.info("StanfordDataLoader initialized")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    seed = int(
        seed_override
        if seed_override is not None
        else getattr(settings.train, "seed", 42)
    )
    reproducible = bool(
        reproducible_override
        if reproducible_override is not None
        else getattr(settings.train, "reproducible", False)
    )
    set_seed(seed)
    configure_cudnn(fast=not reproducible)

    if wandb is None:
        use_wandb = False
    logging.info("Setting up wandb...")
    wandb_run = None
    try:
        if use_wandb:
            wandb_run = wandb.init(
                project="DWMRI-Denoising",
                config={
                    "dataset": dataset,
                    "model_name": "MDS2S",
                    **settings.toDict(),
                },
            )
        else:
            logging.info("wandb disabled (use_wandb=False).")
        logging.info("Loading data...")
        original_from_loader, noisy_data = data_loader.load_data()
        clean_reference = original_from_loader is not None
        if original_from_loader is None:
            logging.info(
                "original_data is None (Stanford loader has no separate GT); "
                "using normalized volume as reference for metrics/visuals"
            )
            original_data = noisy_data
        else:
            original_data = original_from_loader
        logging.info(f"Noisy data shape: {noisy_data.shape}")

        # Permute from (X, Y, Z, Bvalues) to (Z, Bvalues, X, Y)
        # taking Z as different data points for training
        # taking B values as channels
        # taking X and Y as spatial dimensions to predict
        logging.info(f"Transposing data with num_volumes={settings.data.num_volumes}")
        # omitting the b0s from the data
        take_volumes = settings.data.num_b0s + settings.data.num_volumes
        gt_xyzv_for_dti = (
            original_from_loader[..., :take_volumes].astype(np.float64).copy()
            if clean_reference
            else None
        )
        noisy_data = np.transpose(
            noisy_data[..., settings.data.num_b0s : take_volumes],
            (2, 3, 0, 1),
        )
        original_data = np.transpose(
            original_data[..., settings.data.num_b0s : take_volumes],
            (2, 3, 0, 1),
        )
        logging.info(f"Transposed data shape: {noisy_data.shape}")
        logging.info(
            f"Data type: {noisy_data.dtype}, Min: {noisy_data.min():.4f}, Max: {noisy_data.max():.4f}, Mean: {noisy_data.mean():.4f}"
        )

        x_train = torch.from_numpy(noisy_data).type(torch.float)
        logging.info(
            f"Converted to torch tensor: {x_train.shape}, dtype: {x_train.dtype}"
        )

        train_set = TensorDataset(x_train)
        train_loader = DataLoader(
            train_set, batch_size=settings.train.batch_size, shuffle=True
        )
        logging.info(
            f"DataLoader created with batch_size={settings.train.batch_size}, num_batches={len(train_loader)}"
        )

        logging.info("Initializing Self2self model...")
        model = Self2self(
            in_channel=settings.model.in_channel,
            out_channel=settings.model.out_channel,
            p=settings.train.dropout_p,
        )
        logging.info(
            f"Model initialized - in_channel: {settings.model.in_channel}, out_channel: {settings.model.out_channel}, dropout_p: {settings.train.dropout_p}"
        )
        logging.info(
            f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        logging.info("Setting up optimizer and scheduler...")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=settings.train.learning_rate
        )
        logging.info(f"Optimizer: Adam(lr={settings.train.learning_rate})")

        scheduler = None
        if settings.train.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=settings.train.scheduler_step_size,
                gamma=settings.train.scheduler_gamma,
            )
            logging.info(
                f"Scheduler: StepLR(step_size={settings.train.scheduler_step_size}, "
                f"gamma={settings.train.scheduler_gamma})"
            )

        logging.info(f"Training device: {settings.train.device}")
        logging.info(f"Number of epochs: {settings.train.num_epochs}")
        logging.info(f"Mask probability: {settings.train.mask_p}")
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
            "mds2s/losses",
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
                mask_p=settings.train.mask_p,
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
            reconstruct_model = Self2self(
                in_channel=settings.model.in_channel,
                out_channel=settings.model.out_channel,
                p=settings.train.dropout_p,
            )
            reconstruct_model, _, _, _, _, _ = load_checkpoint(
                model=reconstruct_model,
                optimizer=optimizer,
                filename=best_loss_checkpoint,
                device=settings.reconstruct.device,
            )
            reconstruct_loader = DataLoader(train_set, batch_size=1, shuffle=False)
            reconstructed_dwis = reconstruct_dwis(
                model=reconstruct_model,
                data_loader=reconstruct_loader,
                device=settings.reconstruct.device,
                data_shape=x_train.shape,
                mask_p=settings.reconstruct.mask_p,
                n_preds=settings.reconstruct.n_preds,
            )
            logging.info(f"Reconstructed DWIs shape: {reconstructed_dwis.shape}")
            logging.info(
                f"Reconstructed DWIs min: {reconstructed_dwis.min():.4f}, "
                f"max: {reconstructed_dwis.max():.4f}, "
                f"mean: {reconstructed_dwis.mean():.4f}"
            )
            logging.info(f"Reconstructed DWIs dtype: {reconstructed_dwis.dtype}")
            reconstructed_dwis = apply_reconstruction_eval_protocol(
                reconstructed_dwis,
                original_data,
                rescale_to_01=bool(getattr(settings.reconstruct, "rescale_to_01", True)),
                rescale_mode=str(getattr(settings.reconstruct, "rescale_mode", "per_volume")),
                clip_to_range=bool(getattr(settings.reconstruct, "clip_to_range", True)),
            )

            # Full-image metrics (background voxels can dominate and worsen PSNR/SSIM)
            metrics = compute_metrics(original_data, reconstructed_dwis)
            logging.info(f"Metrics: {metrics}")
            # ROI metrics: only over voxels where original > threshold (excludes air/background)
            roi_threshold = getattr(settings.reconstruct, "metrics_roi_threshold", None)
            roi_mask = compute_roi_mask(original_data, roi_threshold)
            if roi_mask is not None:
                n_roi, roi_pct = summarize_roi(roi_mask)
                logging.info(
                    f"ROI mask: original > {roi_threshold}, {n_roi} voxels ({roi_pct:.1f}%)"
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

            if (
                gt_xyzv_for_dti is not None
                and getattr(settings.reconstruct, "compute_dti", True)
            ):
                try:
                    nb0 = int(settings.data.num_b0s)
                    den_dwi_xyzv = np.transpose(
                        reconstructed_dwis.astype(np.float64), (2, 3, 0, 1)
                    )
                    den_xyzv = np.concatenate(
                        [gt_xyzv_for_dti[..., :nb0], den_dwi_xyzv], axis=-1
                    )
                    gtab = data_loader.load_gradient_table()
                    bvals = np.asarray(gtab.bvals)[: int(take_volumes)]
                    bvecs = np.asarray(gtab.bvecs)[: int(take_volumes)]
                    roi_thr = getattr(
                        settings.reconstruct, "metrics_roi_threshold", 0.02
                    )
                    dti = try_compute_dti_errors(
                        den_xyzv,
                        gt_xyzv_for_dti,
                        bvals,
                        bvecs,
                        roi_threshold=roi_thr,
                    )
                    save_dti_metrics(dti, metrics_dir)
                except Exception as dti_exc:
                    logging.warning("DTI metrics skipped: %s", dti_exc)
                    save_dti_metrics(
                        {
                            "fa_mae": None,
                            "md_mae": None,
                            "ad_mae": None,
                            "rd_mae": None,
                            "dti_reference": "clean_gt",
                            "dti_skipped_reason": str(dti_exc),
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
                        if not clean_reference
                        else "clean_gt",
                        "dti_skipped_reason": "no_clean_gt_or_compute_dti_false",
                    },
                    metrics_dir,
                )
            metrics_policy = metrics_policy_dict(
                reference_name="clean_gt" if clean_reference else "self_reference_noisy",
                rescale_to_01=bool(getattr(settings.reconstruct, "rescale_to_01", True)),
                rescale_mode=str(getattr(settings.reconstruct, "rescale_mode", "per_volume")),
                clip_to_range=bool(getattr(settings.reconstruct, "clip_to_range", True)),
                roi_threshold=roi_threshold,
            )
            save_run_manifest(
                out_dir=metrics_dir,
                seed=seed,
                reproducible=reproducible,
                runtime_device=str(settings.reconstruct.device),
                config={
                    "dataset": dataset,
                    "architecture": "mds2s",
                    "num_volumes": int(settings.data.num_volumes),
                    "num_b0s": int(settings.data.num_b0s),
                    "n_preds": int(settings.reconstruct.n_preds),
                },
                metrics_policy=metrics_policy,
            )

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
                        original_volume=original_data,
                        noisy_volume=noisy_data,
                        denoised_volume=reconstructed_dwis,
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
                    reconstructed_dwis,
                    file_name=single_path,
                    volume_idx=0,
                )

                noisy_path = os.path.join(images_dir, "noisy.png")
                visualize_single_volume(
                    noisy_data,
                    file_name=noisy_path,
                    volume_idx=0,
                )

    finally:
        # Ensure wandb run is always finished, even if an exception occurs
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MDS2S baseline")
    parser.add_argument("--dataset", default="dbrain", choices=["dbrain", "stanford"])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-reconstruct", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reproducible", choices=["true", "false"], default=None)
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        train=not args.skip_train,
        reconstruct=not args.skip_reconstruct,
        generate_images=not args.no_images,
        use_wandb=not args.no_wandb,
        seed_override=args.seed,
        reproducible_override=(
            None if args.reproducible is None else args.reproducible == "true"
        ),
    )
