import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis_fullsize(model, data, device, mask_p=0.3, n_preds=10):
    """
    Reconstruct full-size DWI data directly (like DRCNet).
    Model processes the entire volume at once.
    
    Args:
        model: Trained Unet3D model
        data: Input data of shape (Vols, X, Y, Z)
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)
    
    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
    """
    logging.info(f"Starting full-size reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(f"Mode: Direct full-size inference (no sliding windows)")
    
    model.to(device)
    model.eval()
    
    num_vols, x_size, y_size, z_size = data.shape
    spatial_dims = (x_size, y_size, z_size)
    
    # Initialize output array
    sum_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)
    
    with torch.inference_mode():
        data_device = data.to(device)
        
        # Process each target volume separately
        for vol_idx in tqdm(range(num_vols), desc="Processing volumes"):
            logging.info(f"Processing volume {vol_idx + 1}/{num_vols}")
            
            # Multiple predictions per volume for robustness (different random masks)
            for pred_idx in range(n_preds):
                # Create masked input (same as training approach)
                # Generate random mask for the target volume
                p_mtx = np.random.uniform(size=spatial_dims)
                mask = (p_mtx > mask_p).astype(np.float32)
                mask_tensor = (
                    torch.tensor(mask)
                    .to(device, dtype=torch.float32)
                    .unsqueeze(0)
                )
                
                # Apply mask to target volume only
                data_masked = data_device.clone()
                data_masked[vol_idx] = data_device[vol_idx] * mask_tensor
                
                # Add batch dimension: (1, num_vols, X, Y, Z)
                data_masked = data_masked.unsqueeze(0)
                
                # Forward pass: model expects (B, C, X, Y, Z)
                reconstructed = model(data_masked)
                
                # Extract the target volume prediction
                # reconstructed shape: (1, 1, X, Y, Z)
                pred_volume = (
                    reconstructed.squeeze(0).squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                
                # Accumulate predictions
                sum_preds[vol_idx] += pred_volume
        
        # Average predictions
        reconstructed = sum_preds / n_preds
        
        logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
        logging.info(
            f"Output stats - Min: {reconstructed.min():.4f}, "
            f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
        )
    
    return reconstructed


def reconstruct_dwis(
    model,
    data,
    device,
    mask_p=0.3,
    n_preds=10,
    patch_size=32,
    step=16,
    use_sliding_window=True,
):
    """
    Reconstruct full-size DWI data.
    
    Supports two modes:
    1. Direct full-size: Processes entire volume at once (faster, requires more memory)
    2. Sliding window: Processes volume in patches and aggregates (slower, more memory efficient)
    
    Args:
        model: Trained Unet3D model
        data: Input data of shape (Vols, X, Y, Z)
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging)
        patch_size: Size of patches used during training (only for sliding window mode)
        step: Step size for sliding windows (only for sliding window mode)
        use_sliding_window: If True, use sliding window mode; if False, use direct full-size mode
    
    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
    """
    if use_sliding_window:
        return reconstruct_dwis_sliding_window(
            model, data, device, mask_p, n_preds, patch_size, step
        )
    else:
        return reconstruct_dwis_fullsize(model, data, device, mask_p, n_preds)


def reconstruct_dwis_sliding_window(
    model, data, device, mask_p=0.3, n_preds=10, patch_size=32, step=16
):
    """
    Reconstruct full-size DWI data using sliding window inference.
    Breaks the volume into patches and aggregates results.
    
    Args:
        model: Trained Unet3D model
        data: Input data of shape (Vols, X, Y, Z)
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging)
        patch_size: Size of patches used during training
        step: Step size for sliding windows
    
    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
    """
    logging.info(f"Starting sliding window reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(f"Using patch_size={patch_size}, step={step}")
    
    model.to(device)
    model.eval()
    
    num_vols, x_size, y_size, z_size = data.shape
    spatial_dims = (x_size, y_size, z_size)
    
    # Initialize output arrays for aggregation
    # We'll accumulate predictions and counts for averaging
    sum_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)
    count_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)
    
    # Generate sliding window positions
    x_starts = list(range(0, x_size - patch_size + 1, step))
    y_starts = list(range(0, y_size - patch_size + 1, step))
    z_starts = list(range(0, z_size - patch_size + 1, step))
    
    # Handle edge cases - ensure we cover the entire volume
    if x_starts[-1] + patch_size < x_size:
        x_starts.append(x_size - patch_size)
    if y_starts[-1] + patch_size < y_size:
        y_starts.append(y_size - patch_size)
    if z_starts[-1] + patch_size < z_size:
        z_starts.append(z_size - patch_size)
    
    total_patches = len(x_starts) * len(y_starts) * len(z_starts)
    logging.info(f"Total patches to process: {total_patches}")
    
    with torch.inference_mode():
        data_device = data.to(device)
        
        # Process each target volume separately
        for vol_idx in tqdm(range(num_vols), desc="Processing volumes"):
            logging.info(f"Processing volume {vol_idx + 1}/{num_vols}")
            
            # Multiple predictions per volume for robustness (different random masks)
            for pred_idx in range(n_preds):
                # Process each patch location
                patch_count = 0
                for x_start in x_starts:
                    for y_start in y_starts:
                        for z_start in z_starts:
                            patch_count += 1
                            if patch_count % 100 == 0:
                                logging.debug(
                                    f"Processed {patch_count}/{total_patches} patches "
                                    f"for volume {vol_idx+1}, prediction {pred_idx+1}"
                                )
                            
                            # Extract patch: shape (num_vols, patch_size, patch_size, patch_size)
                            patch = data_device[
                                :,
                                x_start : x_start + patch_size,
                                y_start : y_start + patch_size,
                                z_start : z_start + patch_size,
                            ]
                            
                            # Create masked input (same as training approach)
                            # Generate random mask for the target volume
                            p_mtx = np.random.uniform(
                                size=(patch_size, patch_size, patch_size)
                            )
                            mask = (p_mtx > mask_p).astype(np.float32)
                            mask_tensor = (
                                torch.tensor(mask)
                                .to(device, dtype=torch.float32)
                                .unsqueeze(0)
                            )
                            
                            # Apply mask to target volume only
                            patch_masked = patch.clone()
                            patch_masked[vol_idx] = patch[vol_idx] * mask_tensor
                            
                            # Add batch dimension: (1, num_vols, patch_size, patch_size, patch_size)
                            patch_masked = patch_masked.unsqueeze(0)
                            
                            # Forward pass: model expects (B, C, X, Y, Z)
                            reconstructed_patch = model(patch_masked)
                            
                            # Extract the target volume prediction
                            # reconstructed_patch shape: (1, 1, patch_size, patch_size, patch_size)
                            pred_volume = (
                                reconstructed_patch.squeeze(0).squeeze(0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            
                            # Accumulate predictions and counts for averaging
                            sum_preds[
                                vol_idx,
                                x_start : x_start + patch_size,
                                y_start : y_start + patch_size,
                                z_start : z_start + patch_size,
                            ] += pred_volume
                            
                            count_preds[
                                vol_idx,
                                x_start : x_start + patch_size,
                                y_start : y_start + patch_size,
                                z_start : z_start + patch_size,
                            ] += 1.0
        
        # Average predictions (handle overlapping regions)
        # Avoid division by zero
        count_preds = np.maximum(count_preds, 1.0)
        reconstructed = sum_preds / count_preds
        
        logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
        logging.info(
            f"Output stats - Min: {reconstructed.min():.4f}, "
            f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
        )
    
    return reconstructed


def reconstruct_dwis(
    model,
    data,
    device,
    mask_p=0.3,
    n_preds=10,
    patch_size=32,
    step=16,
    use_sliding_window=True,
):
    """
    Reconstruct full-size DWI data.
    
    Supports two modes:
    1. Direct full-size: Processes entire volume at once (faster, requires more memory)
    2. Sliding window: Processes volume in patches and aggregates (slower, more memory efficient)
    
    Args:
        model: Trained Unet3D model
        data: Input data of shape (Vols, X, Y, Z)
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging)
        patch_size: Size of patches used during training (only for sliding window mode)
        step: Step size for sliding windows (only for sliding window mode)
        use_sliding_window: If True, use sliding window mode; if False, use direct full-size mode
    
    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
    """
    if use_sliding_window:
        return reconstruct_dwis_sliding_window(
            model, data, device, mask_p, n_preds, patch_size, step
        )
    else:
        return reconstruct_dwis_fullsize(model, data, device, mask_p, n_preds)
