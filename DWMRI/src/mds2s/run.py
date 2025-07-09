import os
import sys

import torch

# Add the parent directory to Python path to find utils module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DBrainDataLoader, StanfordDataLoader
from fit import fit_model
from model import Self2self

from utils.utils import load_config


def main(dataset: str):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    settings = load_config(config_path)
    print(f"Loaded configuration: {settings}")

    if dataset == "dbrain":
        settings = settings.dbrain
        data_loader = DBrainDataLoader(
            nii_path=settings.data.nii_path,
            bvecs_path=settings.data.bvecs_path,
            bvalue=settings.data.bvalue,
            noise_sigma=settings.data.noise_sigma,
        )
    elif dataset == "stanford":
        data_loader = StanfordDataLoader(settings.data)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    _, noisy_data = data_loader.load_data()
    noisy_data = noisy_data[..., : settings.data.num_volumes]

    model = Self2self(
        in_channel=settings.model.in_channel,
        out_channel=settings.dbrain.model.out_channel,
        p=settings.dbrain.train.dropout_p,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=settings.train.scheduler_step_size,
        gamma=settings.train.scheduler_gamma,
    )

    # checkpoint_dir =

    fit_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=noisy_data,
        num_epochs=settings.train.num_epochs,
        device=settings.train.device,
        mask_p=settings.train.mask_p,
        checkpoint_dir=settings.train.checkpoint_dir,
    )


if __name__ == "__main__":
    main(dataset="dbrain")
