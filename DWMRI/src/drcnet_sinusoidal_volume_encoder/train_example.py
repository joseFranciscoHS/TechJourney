#!/usr/bin/env python3
"""
Example script demonstrating how to train the DRCNet with sinusoidal volume encoding
and EdgeAwareLoss for DWMRI reconstruction.
"""

import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import DenoiserNet
from fit import fit_model, create_training_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main training function with EdgeAwareLoss integration."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create training configuration
    config = create_training_config(
        use_edge_aware_loss=True,
        edge_loss_alpha=0.5,  # Balance between MSE and edge preservation
        num_epochs=50,
        device=device,
        checkpoint_dir="./checkpoints_edge_aware",
    )

    # Model parameters
    model_params = {
        "input_channels": 10,  # Number of input volumes
        "output_channels": 1,  # Single output volume
        "groups": 16,
        "dense_convs": 2,
        "residual": True,
        "base_filters": 32,
        "device": device,
        "num_volumes": 10,
        "use_sinusoidal_encoding": True,
        "embedding_dim": 64,
        "encoding_scale": 0.1,
    }

    # Create model
    model = DenoiserNet(**model_params)
    logging.info(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Create scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart period
        T_mult=2,  # Period multiplier
        eta_min=1e-6,  # Minimum learning rate
    )

    # Note: You would load your actual data loader here
    # train_loader = create_dataloader(...)

    # Example training call (uncomment when you have data)
    # fit_model(
    #     model=model,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     train_loader=train_loader,
    #     num_epochs=config["num_epochs"],
    #     device=config["device"],
    #     checkpoint_dir=config["checkpoint_dir"],
    #     use_edge_aware_loss=config["use_edge_aware_loss"],
    #     edge_loss_alpha=config["edge_loss_alpha"],
    # )

    logging.info("Training setup completed!")
    logging.info(
        "To start training, uncomment the fit_model call and provide your data loader."
    )


if __name__ == "__main__":
    main()
