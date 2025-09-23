#!/usr/bin/env python3
"""
Test script for the cleaned MultiScaleDetailNet module.
This script verifies that all components work correctly after removing the old DenoiserNet.
"""

import logging
import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import MultiScaleDetailNet, EdgeAwareLoss


def test_model_creation():
    """Test MultiScaleDetailNet model creation"""
    logging.info("Testing MultiScaleDetailNet model creation...")

    try:
        model = MultiScaleDetailNet(
            input_channels=9,
            output_channels=1,
            base_filters=32,
            num_volumes=10,
            use_sinusoidal_encoding=True,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"✅ Model created successfully with {total_params:,} parameters")
        return model

    except Exception as e:
        logging.error(f"❌ Model creation failed: {e}")
        return None


def test_loss_function():
    """Test EdgeAwareLoss function"""
    logging.info("Testing EdgeAwareLoss function...")

    try:
        criterion = EdgeAwareLoss(alpha=0.5, beta=0.1)

        # Create sample data
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)

        loss = criterion(pred, target)
        logging.info(f"✅ Loss function works correctly. Loss: {loss.item():.6f}")
        return criterion

    except Exception as e:
        logging.error(f"❌ Loss function failed: {e}")
        return None


def test_forward_pass(model):
    """Test model forward pass"""
    logging.info("Testing model forward pass...")

    try:
        # Create sample input
        inputs = torch.randn(2, 9, 32, 32, 32)
        volume_indices = torch.randint(0, 10, (2, 9))

        # Forward pass
        with torch.no_grad():
            output = model(inputs, volume_indices)

        logging.info(f"✅ Forward pass successful. Output shape: {output.shape}")
        return True

    except Exception as e:
        logging.error(f"❌ Forward pass failed: {e}")
        return False


def test_training_step(model, criterion):
    """Test a single training step"""
    logging.info("Testing training step...")

    try:
        # Create sample data
        inputs = torch.randn(2, 9, 32, 32, 32)
        volume_indices = torch.randint(0, 10, (2, 9))
        targets = torch.randn(2, 1, 32, 32, 32)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(inputs, volume_indices)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        logging.info(f"✅ Training step successful. Loss: {loss.item():.6f}")
        return True

    except Exception as e:
        logging.error(f"❌ Training step failed: {e}")
        return False


def test_mixed_precision(model, criterion):
    """Test mixed precision training"""
    logging.info("Testing mixed precision training...")

    if not torch.cuda.is_available():
        logging.info("⚠️ CUDA not available, skipping mixed precision test")
        return True

    try:
        from torch.cuda.amp import autocast, GradScaler

        # Create sample data
        inputs = torch.randn(2, 9, 32, 32, 32).cuda()
        volume_indices = torch.randint(0, 10, (2, 9)).cuda()
        targets = torch.randn(2, 1, 32, 32, 32).cuda()

        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()

        # Mixed precision training step
        model.train()
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs, volume_indices)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logging.info(f"✅ Mixed precision training successful. Loss: {loss.item():.6f}")
        return True

    except Exception as e:
        logging.error(f"❌ Mixed precision training failed: {e}")
        return False


def main():
    """Run all tests"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("🧪 Starting MultiScaleDetailNet Module Tests")
    logger.info("=" * 50)

    # Test 1: Model creation
    model = test_model_creation()
    if model is None:
        logger.error("❌ Module test failed at model creation")
        return False

    # Test 2: Loss function
    criterion = test_loss_function()
    if criterion is None:
        logger.error("❌ Module test failed at loss function")
        return False

    # Test 3: Forward pass
    if not test_forward_pass(model):
        logger.error("❌ Module test failed at forward pass")
        return False

    # Test 4: Training step
    if not test_training_step(model, criterion):
        logger.error("❌ Module test failed at training step")
        return False

    # Test 5: Mixed precision
    if not test_mixed_precision(model, criterion):
        logger.error("❌ Module test failed at mixed precision")
        return False

    logger.info("=" * 50)
    logger.info("🎉 All tests passed! Module is working correctly.")
    logger.info("✅ MultiScaleDetailNet module is ready for use.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
