#!/usr/bin/env python3
"""
Test script to verify EdgeAwareLoss functionality.
"""

import logging
import torch
import numpy as np

from model import EdgeAwareLoss

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_edge_aware_loss():
    """Test EdgeAwareLoss with synthetic data."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Testing EdgeAwareLoss on device: {device}")

    # Create loss function
    edge_loss = EdgeAwareLoss(alpha=0.5)
    edge_loss.to(device)

    # Create synthetic test data
    batch_size = 2
    channels = 1
    x, y, z = 32, 32, 32

    # Create target with clear edges (step function)
    target = torch.zeros(batch_size, channels, x, y, z, device=device)
    target[:, :, 16:, :, :] = 1.0  # Clear edge at x=16

    # Create prediction with slightly blurred edges
    pred = target.clone()
    # Add some blur to the edges
    pred[:, :, 14:18, :, :] = 0.5

    # Add some noise
    noise = torch.randn_like(pred) * 0.1
    pred = pred + noise
    pred = torch.clamp(pred, 0, 1)

    logging.info(f"Target shape: {target.shape}")
    logging.info(f"Prediction shape: {pred.shape}")

    # Test edge detection
    target_edges = edge_loss.detect_edges(target)
    pred_edges = edge_loss.detect_edges(pred)

    logging.info(f"Target edges shape: {target_edges.shape}")
    logging.info(f"Prediction edges shape: {pred_edges.shape}")
    logging.info(f"Target edges max: {target_edges.max().item():.4f}")
    logging.info(f"Prediction edges max: {pred_edges.max().item():.4f}")

    # Test loss computation
    loss = edge_loss(pred, target)

    logging.info(f"Total loss: {loss.item():.6f}")

    # Test individual components
    mse_loss = torch.nn.functional.mse_loss(pred, target)
    edge_loss_component = torch.nn.functional.mse_loss(pred_edges, target_edges)

    logging.info(f"MSE loss: {mse_loss.item():.6f}")
    logging.info(f"Edge loss: {edge_loss_component.item():.6f}")
    logging.info(
        f"Expected total: {mse_loss.item() + 0.5 * edge_loss_component.item():.6f}"
    )

    # Test with different alpha values
    logging.info("\nTesting different alpha values:")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        test_loss = EdgeAwareLoss(alpha=alpha)
        test_loss.to(device)
        loss_val = test_loss(pred, target)
        logging.info(f"Alpha {alpha}: Loss = {loss_val.item():.6f}")

    logging.info("EdgeAwareLoss test completed successfully!")


def test_gradient_flow():
    """Test that gradients flow properly through EdgeAwareLoss."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Testing gradient flow on device: {device}")

    # Create loss function
    edge_loss = EdgeAwareLoss(alpha=0.5)
    edge_loss.to(device)

    # Create test data
    batch_size = 1
    channels = 1
    x, y, z = 16, 16, 16

    target = torch.randn(batch_size, channels, x, y, z, device=device)
    pred = torch.randn(batch_size, channels, x, y, z, device=device, requires_grad=True)

    # Compute loss
    loss = edge_loss(pred, target)

    # Backward pass
    loss.backward()

    # Check gradients
    if pred.grad is not None:
        logging.info(f"Gradient shape: {pred.grad.shape}")
        logging.info(f"Gradient mean: {pred.grad.mean().item():.6f}")
        logging.info(f"Gradient std: {pred.grad.std().item():.6f}")
        logging.info(f"Gradient max: {pred.grad.max().item():.6f}")
        logging.info("Gradients computed successfully!")
    else:
        logging.error("No gradients computed!")

    logging.info("Gradient flow test completed!")


if __name__ == "__main__":
    test_edge_aware_loss()
    print("\n" + "=" * 50 + "\n")
    test_gradient_flow()
