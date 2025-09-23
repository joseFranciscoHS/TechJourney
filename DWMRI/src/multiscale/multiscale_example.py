#!/usr/bin/env python3
"""
Example usage of MultiScaleDetailNet with EdgeAwareLoss for DWMRI reconstruction.
This script demonstrates how to use the new multi-scale architecture for detail preservation.
"""

import logging
import torch
import torch.nn as nn
from multiscale.model import MultiScaleDetailNet, EdgeAwareLoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model_example():
    """Create and demonstrate the MultiScaleDetailNet model"""

    # Model parameters
    input_channels = 9  # Number of input volumes
    output_channels = 1  # Single output volume
    num_volumes = 10  # Total number of volumes in dataset
    base_filters = 32  # Base number of filters

    # Create the multi-scale model
    model = MultiScaleDetailNet(
        input_channels=input_channels,
        output_channels=output_channels,
        groups=1,
        dense_convs=2,
        residual=True,
        base_filters=base_filters,
        num_volumes=num_volumes,
        use_sinusoidal_encoding=True,
        embedding_dim=64,
        encoding_scale=0.1,
    )

    # Create edge-aware loss function
    criterion = EdgeAwareLoss(alpha=0.5, beta=0.1)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger.info("Model and loss function created successfully!")

    return model, criterion, optimizer


def test_forward_pass():
    """Test the forward pass with sample data"""

    model, criterion, optimizer = create_model_example()

    # Create sample data
    batch_size = 2
    num_input_volumes = 9
    spatial_dims = (64, 64, 64)  # Smaller for testing

    # Sample input volumes
    inputs = torch.randn(batch_size, num_input_volumes, *spatial_dims)

    # Sample volume indices (which volumes are being used)
    volume_indices = torch.randint(0, 10, (batch_size, num_input_volumes))

    # Sample target volume
    target = torch.randn(batch_size, 1, *spatial_dims)

    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Volume indices shape: {volume_indices.shape}")
    logger.info(f"Target shape: {target.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(inputs, volume_indices)
        logger.info(f"Output shape: {output.shape}")

        # Test loss computation
        loss = criterion(output, target)
        logger.info(f"Edge-aware loss: {loss.item():.6f}")

    # Test training step
    model.train()
    optimizer.zero_grad()

    output = model(inputs, volume_indices)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    logger.info(f"Training step completed. Loss: {loss.item():.6f}")


def test_model_parameters():
    """Test MultiScaleDetailNet parameters and functionality"""

    # Create model
    model = MultiScaleDetailNet(
        input_channels=9,
        output_channels=1,
        base_filters=32,
        num_volumes=10,
        use_sinusoidal_encoding=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"MultiScaleDetailNet parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Test with sample input
    inputs = torch.randn(1, 9, 32, 32, 32)
    volume_indices = torch.randint(0, 10, (1, 9))

    with torch.no_grad():
        output = model(inputs, volume_indices)
        logger.info(f"Model output shape: {output.shape}")
        logger.info(f"Model test completed successfully")


def training_example():
    """Example of training with mixed precision and gradient accumulation"""

    model, criterion, optimizer = create_model_example()

    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Training parameters
    accumulation_steps = 4
    num_epochs = 10

    logger.info("Starting training example...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Simulate training loop
        for step in range(10):  # 10 steps per epoch
            # Create sample batch
            inputs = torch.randn(2, 9, 32, 32, 32)
            volume_indices = torch.randint(0, 10, (2, 9))
            targets = torch.randn(2, 1, 32, 32, 32)

            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, volume_indices)
                    loss = criterion(outputs, targets) / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs, volume_indices)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / 10
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    logger.info("=== MultiScaleDetailNet Example ===")

    # Test basic functionality
    test_forward_pass()

    logger.info("\n=== Model Parameters ===")
    test_model_parameters()

    logger.info("\n=== Training Example ===")
    training_example()

    logger.info("All examples completed successfully!")
