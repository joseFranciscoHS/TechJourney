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
from utils.data import DBrainDataLoader, StanfordDataLoader, invert_normalization
from utils.eval_protocol import (
    apply_reconstruction_eval_protocol,
    compute_roi_mask,
    metrics_policy_dict,
    save_run_manifest,
    summarize_roi,
)
from utils.experiment_runtime import losses_dir_from_train_checkpoint_dir
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


def _resolve_device(device_value: str) -> str:
    if device_value == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if device_value == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            logging.warning("MPS requested but unavailable; falling back to CPU.")
            return "cpu"
    return device_value


def main(
    dataset: str,
    train: bool = True,
    reconstruct: bool = True,
    generate_images: bool = True,
    use_wandb: bool = True,
    seed_override: int | None = None,
    reproducible_override: bool | None = None,
    nii_path_override: str | None = None,
    bvecs_path_override: str | None = None,
    device_override: str | None = None,
    num_epochs_override: int | None = None,
    noise_sigma_override: float | None = None,
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
        if noise_sigma_override is not None:
            settings.data.noise_sigma = float(noise_sigma_override)
        if nii_path_override is not None:
            settings.data.nii_path = nii_path_override
        if bvecs_path_override is not None:
            settings.data.bvecs_path = bvecs_path_override
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
        if noise_sigma_override is not None:
            settings.data.noise_sigma = float(noise_sigma_override)
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
    if device_override is not None:
        resolved_device = _resolve_device(str(device_override))
        settings.train.device = resolved_device
        settings.reconstruct.device = resolved_device
    if num_epochs_override is not None:
        settings.train.num_epochs = int(num_epochs_override)

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
        _loss_train_root = losses_dir_from_train_checkpoint_dir(
            settings.train.checkpoint_dir
        )
        loss_dir = os.path.join(
            _loss_train_root,
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
                rescale_to_01=bool(
                    getattr(settings.reconstruct, "rescale_to_01", True)
                ),
                rescale_mode=str(
                    getattr(settings.reconstruct, "rescale_mode", "per_volume")
                ),
                clip_to_range=bool(
                    getattr(settings.reconstruct, "clip_to_range", True)
                ),
            )

            # Persist full 4D volume for paper qualitative panels / FA-MD maps.
            # MDS2S reconstructs DWI-only in (Z, V, X, Y) normalized space; assemble
            # physical-intensity [b0s + DWIs] to match hybrid/baseline export layout.
            _vol_dir = os.environ.get("MDS2S_SAVE_VOLUME_DIR", "").strip()
            if _vol_dir:
                os.makedirs(_vol_dir, exist_ok=True)
                _raw = np.asarray(reconstructed_dwis)
                _raw_path = os.path.join(_vol_dir, "denoised_mds2s_raw.npy")
                np.save(_raw_path, _raw.astype(np.float32))
                logging.info("Saved MDS2S raw volume %s shape=%s", _raw_path, _raw.shape)
                try:
                    from paper_eval.export_denoised import save_arm
                    from utils.data import invert_normalization

                    nb0 = int(settings.data.num_b0s)
                    n_dwi = int(settings.data.num_volumes)
                    take_volumes = nb0 + n_dwi
                    if _raw.ndim == 4 and _raw.shape[1] == n_dwi:
                        dwi_01 = np.transpose(_raw, (2, 3, 0, 1)).astype(np.float64)
                    elif _raw.ndim == 4 and _raw.shape[-1] == n_dwi:
                        dwi_01 = _raw.astype(np.float64)
                    else:
                        raise ValueError(
                            f"unexpected MDS2S reconstruct shape {_raw.shape}"
                        )
                    # Prefer live loader GT (normalized) for b0s + norm params.
                    src_full = (
                        original_from_loader
                        if original_from_loader is not None
                        else None
                    )
                    if src_full is None or src_full.shape[-1] < take_volumes:
                        raise RuntimeError("missing full normalized volume for b0 assemble")
                    # Crop Z to match reconstruct if needed (shared take_z vs raw Z).
                    if dwi_01.shape[2] != src_full.shape[2]:
                        z = min(dwi_01.shape[2], src_full.shape[2])
                        dwi_01 = dwi_01[:, :, :z, :]
                        src_full = src_full[:, :, :z, :]
                    norm_params = getattr(data_loader, "norm_params_", None)
                    if norm_params is None:
                        raise RuntimeError("data_loader.norm_params_ missing")
                    b0_phys = invert_normalization(
                        src_full[..., :nb0].astype(np.float64), norm_params[:nb0]
                    )
                    dwi_phys = invert_normalization(
                        dwi_01, norm_params[nb0:take_volumes]
                    )
                    vol_4d = np.concatenate([b0_phys, dwi_phys], axis=-1).astype(
                        np.float32
                    )
                    gtab = data_loader.load_gradient_table()
                    bvals = np.asarray(gtab.bvals)[:take_volumes]
                    bvecs = np.asarray(gtab.bvecs)[:take_volumes]
                    affine = getattr(data_loader, "affine_", None)
                    if affine is None:
                        affine = np.eye(4, dtype=np.float64)
                        affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.4
                    save_arm(_vol_dir, "mds2s", vol_4d, affine, bvals, bvecs, nb0)
                    logging.info(
                        "Saved MDS2S paper volume via save_arm shape=%s", vol_4d.shape
                    )
                except Exception as exc:
                    logging.exception("MDS2S paper volume assemble failed: %s", exc)

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

            if gt_xyzv_for_dti is not None and getattr(
                settings.reconstruct, "compute_dti", True
            ):
                try:
                    nb0 = int(settings.data.num_b0s)
                    den_dwi_xyzv = np.transpose(
                        reconstructed_dwis.astype(np.float64), (2, 3, 0, 1)
                    )
                    den_xyzv = np.concatenate(
                        [gt_xyzv_for_dti[..., :nb0], den_dwi_xyzv], axis=-1
                    )
                    norm_params = getattr(data_loader, "norm_params_", None)
                    if norm_params is not None:
                        gt_xyzv_for_dti = invert_normalization(
                            gt_xyzv_for_dti, norm_params[: int(take_volumes)]
                        )
                        den_dwis = invert_normalization(
                            den_dwi_xyzv, norm_params[nb0 : int(take_volumes)]
                        )
                        den_xyzv = np.concatenate(
                            [gt_xyzv_for_dti[..., :nb0], den_dwis.astype(np.float64)],
                            axis=-1,
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
                reference_name="clean_gt"
                if clean_reference
                else "self_reference_noisy",
                rescale_to_01=bool(
                    getattr(settings.reconstruct, "rescale_to_01", True)
                ),
                rescale_mode=str(
                    getattr(settings.reconstruct, "rescale_mode", "per_volume")
                ),
                clip_to_range=bool(
                    getattr(settings.reconstruct, "clip_to_range", True)
                ),
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
    parser.add_argument("--nii-path", default=None)
    parser.add_argument("--bvecs-path", default=None)
    parser.add_argument("--reproducible", choices=["true", "false"], default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--noise-sigma", type=float, default=None)
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        train=not args.skip_train,
        reconstruct=not args.skip_reconstruct,
        generate_images=not args.no_images,
        use_wandb=not args.no_wandb,
        seed_override=args.seed,
        nii_path_override=args.nii_path,
        bvecs_path_override=args.bvecs_path,
        device_override=args.device,
        num_epochs_override=args.num_epochs,
        noise_sigma_override=args.noise_sigma,
        reproducible_override=(
            None if args.reproducible is None else args.reproducible == "true"
        ),
    )
