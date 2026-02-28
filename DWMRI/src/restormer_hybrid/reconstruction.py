import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis(model, data, device, mask_p=0.3, n_preds=10):
    """
    Reconstruct full-size DWI data using hybrid MD-S2S approach.

    Uses the same masking strategy as training: all volumes are included,
    but the target volume is masked with Bernoulli mask. Multiple predictions
    with different masks are averaged for robustness.

    Memory optimization: Keeps the prediction accumulator on GPU to eliminate
    unnecessary GPU-CPU transfers during the reconstruction loop.

    Args:
        model: Trained Restormer3D-hybrid model
        data: Input data of shape (Vols, X, Y, Z) - full-size data
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)

    Returns:
        Reconstructed data of shape (Vols, X, Y, Z) as numpy array
    """
    logging.info(f"Starting DWI reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(
        f"Using hybrid MD-S2S approach with mask_p={mask_p}, n_preds={n_preds}"
    )

    model.to(device)
    model.eval()

    num_vols, x_size, y_size, z_size = data.shape
    spatial_dims = (x_size, y_size, z_size)

    # Keep accumulator on GPU to eliminate GPU-CPU transfers in the loop
    sum_preds = torch.zeros(
        (num_vols, *spatial_dims), dtype=torch.float32, device=device
    )

    with torch.inference_mode():
        data_device = data.to(device)

        # Process each target volume separately
        for vol_idx in tqdm(range(num_vols), desc="Processing volumes"):
            logging.info(f"Processing volume {vol_idx + 1}/{num_vols}")

            # Multiple predictions per volume for robustness (different random masks)
            for pred_idx in range(n_preds):
                # Generate random mask directly on GPU
                mask_tensor = (
                    torch.rand(spatial_dims, device=device, dtype=torch.float32) > mask_p
                ).float().unsqueeze(0)

                # Apply mask to target volume only (keep all volumes in input)
                data_masked = data_device.clone()
                data_masked[vol_idx] = data_device[vol_idx] * mask_tensor

                # Add batch dimension: (1, num_vols, X, Y, Z)
                data_masked = data_masked.unsqueeze(0)

                # Forward pass: model expects (B, C, X, Y, Z)
                reconstructed = model(data_masked)

                # Accumulate predictions on GPU (no CPU transfer)
                sum_preds[vol_idx] += reconstructed.squeeze(0).squeeze(0)

        # Average predictions and move to CPU only at the end
        reconstructed = (sum_preds / n_preds).cpu().numpy()

        logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
        logging.info(
            f"Output stats - Min: {reconstructed.min():.4f}, "
            f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
        )

    return reconstructed
