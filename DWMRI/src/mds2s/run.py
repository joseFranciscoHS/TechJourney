import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mds2s.data import DBrainDataLoader, StanfordDataLoader
from mds2s.fit import fit_model
from mds2s.model import Self2self
from mds2s.reconstruction import reconstruct_dwis
from utils import setup_logging
from utils.checkpoint import load_checkpoint
from utils.metrics import (
    compare_volumes,
    compute_metrics,
    save_metrics,
    visualize_single_volume,
)
from utils.utils import load_config


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
    logging.info(f"Configuration loaded successfully")

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

    logging.info("Loading data...")
    _, noisy_data = data_loader.load_data()
    logging.info(f"Noisy data shape: {noisy_data.shape}")

    # Permute from (X, Y, Z, Bvalues) to (Z, Bvalues, X, Y)
    # taking Z as different data points for training
    # taking B values as channels
    # taking X and Y as spatial dimensions to predict
    logging.info(
        f"Transposing data with num_volumes={settings.data.num_volumes}"
    )
    noisy_data = np.transpose(
        noisy_data[..., : settings.data.num_volumes], (2, 3, 0, 1)
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
        )

        logging.info("Training setup completed successfully")
        logging.info(f"Training completed. Log file: {log_file}")

    if reconstruct:
        logging.info("Reconstructing DWIs...")
        best_loss_checkpoint = os.path.join(
            checkpoint_dir, "best_loss_checkpoint.pth"
        )
        model, _, _, _, _ = load_checkpoint(
            model=model,
            optimizer=optimizer,
            filename=best_loss_checkpoint,
            device=settings.reconstruct.device,
        )
        reconstruct_loader = DataLoader(train_set, batch_size=1, shuffle=False)
        reconstructed_dwis = reconstruct_dwis(
            model=model,
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

        metrics = compute_metrics(noisy_data, reconstructed_dwis)
        logging.info(f"Metrics: {metrics}")
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
            compare_volumes(
                noisy_data,
                reconstructed_dwis,
                file_name=os.path.join(images_dir, "comparison.png"),
            )
            logging.info(
                f"Saving single volume image to: {images_dir}/single.png"
            )
            visualize_single_volume(
                reconstructed_dwis,
                file_name=os.path.join(images_dir, "single.png"),
            )


if __name__ == "__main__":
    main(dataset="dbrain")
