"""
Reconstruction for DRCNet-S2S with variance reduction via mask averaging.

Implements the S2S dropout-based ensemble: multiple forward passes with different
Bernoulli masks are averaged to reduce variance of the final prediction.
See technical report Section 2 (Variance reduction).
"""

import logging

import numpy as np
import torch


def reconstruct_dwis(model, data, device, mask_p=0.3, n_preds=10):
    """
    Reconstruct full-size DWI data using DRCNet with S2S framework.

    All volumes at a time, masked with Bernoulli mask (same as training).
    Multiple predictions with different masks are averaged for robustness
    (variance reduction per J-invariance / Self2Self).

    Args:
        model: Trained DRCNet-S2S model
        data: Input data of shape (Vols, X, Y, Z) - full-size data
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)

    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
    """
    logging.info(f"Starting DWI reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(f"Using S2S framework with mask_p={mask_p}, n_preds={n_preds}")

    model.to(device)
    model.eval()

    num_vols, x_size, y_size, z_size = data.shape

    # Initialize output array (Vols, X, Y, Z)
    sum_preds = np.zeros((num_vols, x_size, y_size, z_size), dtype=np.float32)

    with torch.inference_mode():
        data_device = data.to(device)

        # Process each target volume separately
        for pred_idx in range(n_preds):
            # Create masked input (same as training approach)
            # Generate random mask for the target volumes (Vols, X, Y, Z)
            p_mtx = np.random.uniform(size=(num_vols, x_size, y_size, z_size))
            mask = (p_mtx > mask_p).astype(np.float32)
            mask_tensor = torch.tensor(mask).to(device, dtype=torch.float32)
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, Vols, X, Y, Z)

            # data_masked: (Vols, X, Y, Z); add channel and batch -> (1, Vols, X, Y, Z)
            data_masked = data_device.clone()
            data_masked = data_masked.unsqueeze(0) * mask_tensor  # (1, Vols, X, Y, Z)

            # Forward pass: model expects (B, Vols, X, Y, Z)
            reconstructed = model((data_device, data_masked))

            # reconstructed shape: (1, 1, Vols, X, Y, Z)
            pred_volume = reconstructed.squeeze(0).squeeze(0).detach().cpu().numpy()

            # Accumulate predictions
            sum_preds += pred_volume

        # Average predictions
        reconstructed = sum_preds / n_preds

        logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
        logging.info(
            f"Output stats - Min: {reconstructed.min():.4f}, "
            f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
        )

    return reconstructed
