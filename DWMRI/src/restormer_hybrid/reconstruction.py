import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis(
    model, data, device, mask_p=0.3, n_preds=10, patch_size=32, overlap=8, use_amp=True
):
    """
    Reconstruct full-size DWI data using sliding window approach.

    Uses patch-based inference to handle memory constraints - the same strategy
    that makes training possible on limited GPU memory. Patches are processed
    with overlap and blended using weighted averaging for smooth transitions.

    Args:
        model: Trained Restormer3D-hybrid model
        data: Input data of shape (Vols, X, Y, Z) - full-size data
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)
        patch_size: Size of cubic patches for inference (default 32)
        overlap: Overlap between adjacent patches for blending (default 8)
        use_amp: Use automatic mixed precision for reduced memory (default True)

    Returns:
        Reconstructed data of shape (Vols, X, Y, Z) as numpy array
    """
    logging.info(f"Starting DWI reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(f"Using sliding window: patch_size={patch_size}, overlap={overlap}")
    logging.info(f"Mask probability: {mask_p}, n_preds: {n_preds}")
    if use_amp and device == "cuda":
        logging.info("Using AMP (float16) for reduced memory during inference")

    model.to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    num_vols, x_size, y_size, z_size = data.shape
    stride = patch_size - overlap

    # Pad data if needed to fit patch grid
    pad_x = (stride - (x_size - patch_size) % stride) % stride
    pad_y = (stride - (y_size - patch_size) % stride) % stride
    pad_z = (stride - (z_size - patch_size) % stride) % stride

    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        data = torch.nn.functional.pad(
            data, (0, pad_z, 0, pad_y, 0, pad_x), mode="reflect"
        )
        logging.info(f"Padded data to shape: {data.shape}")

    padded_x, padded_y, padded_z = data.shape[1], data.shape[2], data.shape[3]

    # Output accumulator and weight map (on CPU to save GPU memory)
    sum_preds = np.zeros((num_vols, padded_x, padded_y, padded_z), dtype=np.float32)
    weight_map = np.zeros((padded_x, padded_y, padded_z), dtype=np.float32)

    # Create blending weights (cosine taper at edges)
    blend_weights = _create_blend_weights(patch_size, overlap)

    # Calculate patch positions
    x_positions = list(range(0, padded_x - patch_size + 1, stride))
    y_positions = list(range(0, padded_y - patch_size + 1, stride))
    z_positions = list(range(0, padded_z - patch_size + 1, stride))

    total_patches = len(x_positions) * len(y_positions) * len(z_positions)
    logging.info(f"Processing {total_patches} patches per volume per prediction")

    with torch.inference_mode():
        for vol_idx in tqdm(range(num_vols), desc="Processing volumes"):
            logging.info(f"Processing volume {vol_idx + 1}/{num_vols}")

            for pred_idx in range(n_preds):
                # Process patches for this volume/prediction
                for x_start in x_positions:
                    for y_start in y_positions:
                        for z_start in z_positions:
                            x_end = x_start + patch_size
                            y_end = y_start + patch_size
                            z_end = z_start + patch_size

                            # Extract patch (all volumes)
                            patch = data[
                                :, x_start:x_end, y_start:y_end, z_start:z_end
                            ].clone()

                            # Apply mask to target volume
                            mask = (
                                torch.rand(
                                    (patch_size, patch_size, patch_size),
                                    dtype=torch.float32,
                                )
                                > mask_p
                            ).float()
                            patch[vol_idx] = patch[vol_idx] * mask

                            # Move to device, add batch dim
                            patch = patch.unsqueeze(0).to(device)

                            # Forward pass with AMP
                            if use_amp and device == "cuda":
                                with torch.amp.autocast(
                                    device_type="cuda", dtype=torch.float16
                                ):
                                    pred = model(patch)
                            else:
                                pred = model(patch)

                            # Move prediction to CPU and accumulate
                            pred_np = pred.float().squeeze(0).squeeze(0).cpu().numpy()
                            sum_preds[
                                vol_idx, x_start:x_end, y_start:y_end, z_start:z_end
                            ] += (pred_np * blend_weights)

                            del patch, pred

                # Accumulate weights (same for all predictions of this volume)
                if pred_idx == 0:
                    for x_start in x_positions:
                        for y_start in y_positions:
                            for z_start in z_positions:
                                weight_map[
                                    x_start : x_start + patch_size,
                                    y_start : y_start + patch_size,
                                    z_start : z_start + patch_size,
                                ] += blend_weights

            if device == "cuda":
                torch.cuda.empty_cache()

    # Normalize by weights and number of predictions
    weight_map = np.maximum(weight_map, 1e-8)  # Avoid division by zero
    reconstructed = sum_preds / (weight_map[np.newaxis, ...] * n_preds)

    # Remove padding
    reconstructed = reconstructed[:, :x_size, :y_size, :z_size]

    logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
    logging.info(
        f"Output stats - Min: {reconstructed.min():.4f}, "
        f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
    )

    return reconstructed


def reconstruct_full_dwi_chunked(
    model,
    noisy_xyzv,
    train_num_volumes,
    device,
    mask_p=0.3,
    n_preds=10,
    patch_size=32,
    overlap=8,
    use_amp=True,
):
    """
    Run reconstruct_dwis on contiguous blocks along the volume axis.

    The model was trained with ``train_num_volumes`` input channels; the full
    DWI stack (all non-b0 volumes) is split into chunks of that size, each
    chunk is reconstructed independently, and results are concatenated along
    the volume dimension.

    Args:
        noisy_xyzv: (X, Y, Z, V) all diffusion-weighted volumes after b0s.
        train_num_volumes: Chunk size (must match training ``inp_channels``).

    Returns:
        Array (X, Y, Z, V) with same V as ``noisy_xyzv``.
    """
    if train_num_volumes < 1:
        raise ValueError("train_num_volumes must be >= 1")
    v = noisy_xyzv.shape[-1]
    pad = (train_num_volumes - (v % train_num_volumes)) % train_num_volumes
    if pad > 0:
        noisy_xyzv = np.pad(
            noisy_xyzv,
            ((0, 0), (0, 0), (0, 0), (0, pad)),
            mode="edge",
        )
        logging.info(
            f"Chunked reconstruction: padded last dim by {pad} (edge) so V is "
            f"multiple of {train_num_volumes}"
        )

    v_padded = noisy_xyzv.shape[-1]
    chunks_out = []
    for start in range(0, v_padded, train_num_volumes):
        block = noisy_xyzv[..., start : start + train_num_volumes]
        x_t = torch.from_numpy(np.transpose(block, (3, 0, 1, 2))).type(torch.float)
        rec_vxyz = reconstruct_dwis(
            model=model,
            data=x_t,
            device=device,
            mask_p=mask_p,
            n_preds=n_preds,
            patch_size=patch_size,
            overlap=overlap,
            use_amp=use_amp,
        )
        chunks_out.append(np.transpose(rec_vxyz, (1, 2, 3, 0)))

    full = np.concatenate(chunks_out, axis=-1)
    return full[..., :v]


def _create_blend_weights(patch_size, overlap):
    """
    Create 3D Gaussian blending weights for smooth patch stitching.
    
    Uses a separable Gaussian window that provides smooth transitions
    at patch boundaries without the corner/edge underweighting issue
    of multiplicative cosine tapers.
    
    Args:
        patch_size: Size of cubic patch
        overlap: Overlap between patches (used to determine sigma)
    
    Returns:
        3D Gaussian weight array of shape (patch_size, patch_size, patch_size)
    """
    # Sigma based on overlap - larger overlap = narrower Gaussian (more blending)
    # For 50% overlap (overlap = patch_size/2), sigma_ratio ~0.3 works well
    sigma_ratio = 0.3 if overlap > 0 else 0.5
    
    center = (patch_size - 1) / 2.0
    sigma = patch_size * sigma_ratio
    
    # Create 1D Gaussian
    x = np.arange(patch_size) - center
    gaussian_1d = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Create 3D Gaussian via outer product (separable)
    weights = np.einsum('i,j,k->ijk', gaussian_1d, gaussian_1d, gaussian_1d)
    
    return weights.astype(np.float32)
