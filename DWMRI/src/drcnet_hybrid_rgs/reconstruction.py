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

    Args:
        model: Trained DRCNet-hybrid model
        data: Input data of shape (Vols, X, Y, Z) - full-size data
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)

    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
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
                    torch.tensor(mask).to(device, dtype=torch.float32).unsqueeze(0)
                )

                # Apply mask to target volume only (keep all volumes in input)
                data_masked = data_device.clone()
                data_masked[vol_idx] = data_device[vol_idx] * mask_tensor

                # Add batch dimension: (1, num_vols, X, Y, Z)
                data_masked = data_masked.unsqueeze(0)

                # Forward pass: model expects (B, C, X, Y, Z)
                reconstructed = model(data_masked)

                # Extract the target volume prediction
                # reconstructed shape: (1, 1, X, Y, Z)
                pred_volume = reconstructed.squeeze(0).squeeze(0).detach().cpu().numpy()

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


def reconstruct_dwis_rgs(
    model,
    data,
    device,
    mask_p=0.3,
    n_preds=10,
    n_context=10,
    target_channel=9,
    num_input=10,
    seed=None,
):
    """
    RGS–Hybrid inference: for each gradient index ``k``, Monte-Carlo average over
    random (K-1)-tuples of context indices with volume ``k`` fixed at ``target_channel``.

    Matches training when ``target_channel`` is the masked slot (e.g. 9 of 10) and
    training used random K-subsets from the full shell.

    Args:
        data: (Vols, X, Y, Z) full noisy shell on CPU (float tensor or numpy).
        n_context: outer MC passes (random 9-tuples + k).
        n_preds: inner spatial Bernoulli mask passes per context.
        target_channel: 0-based index where volume ``k`` is placed (must be ``num_input - 1`` for parity with "K-th draw" training).
        num_input: K (must match model input channels).

    Returns:
        Reconstructed array (Vols, X, Y, Z).
    """
    logging.info(
        f"RGS reconstruction: mask_p={mask_p}, n_context={n_context}, n_preds={n_preds}, "
        f"target_channel={target_channel}, K={num_input}"
    )
    rng = np.random.default_rng(seed)

    if isinstance(data, np.ndarray):
        data_t = torch.from_numpy(data.astype(np.float32))
    else:
        data_t = data.float()

    num_vols, x_size, y_size, z_size = data_t.shape
    spatial_dims = (x_size, y_size, z_size)
    if num_input != target_channel + 1:
        logging.warning(
            f"target_channel={target_channel} with K={num_input}: typical parity uses "
            f"target_channel=K-1 (last slot = masked volume)."
        )

    sum_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)
    denom = float(n_context * n_preds)

    model.to(device)
    model.eval()
    data_dev = data_t.to(device)

    with torch.inference_mode():
        for vol_k in tqdm(range(num_vols), desc="RGS volumes"):
            acc = torch.zeros(
                (x_size, y_size, z_size), device=device, dtype=torch.float32
            )
            others = [i for i in range(num_vols) if i != vol_k]
            for _ in range(n_context):
                ctx = rng.choice(others, size=num_input - 1, replace=False)
                order = np.concatenate([ctx, np.array([vol_k], dtype=np.int64)])
                stack = data_dev[order]
                for _pred in range(n_preds):
                    p_mtx = np.random.uniform(size=spatial_dims)
                    mask_np = (p_mtx > mask_p).astype(np.float32)
                    mask_tensor = torch.tensor(
                        mask_np, device=device, dtype=torch.float32
                    )

                    inp = stack.clone()
                    inp[target_channel] = inp[target_channel] * mask_tensor
                    inp_b = inp.unsqueeze(0)
                    out = model(inp_b)
                    pred = out.squeeze(0).squeeze(0)
                    acc = acc + pred
            sum_preds[vol_k] = (acc / denom).detach().cpu().numpy()

    logging.info(
        f"RGS reconstruction done. Min={sum_preds.min():.4f}, Max={sum_preds.max():.4f}, "
        f"Mean={sum_preds.mean():.4f}"
    )
    return sum_preds


def reconstruct_dwis_index_volume(model, data, index, device, mask_p=0.3, n_preds=10):
    """
    Reconstruct full-size DWI data using hybrid MD-S2S approach.

    Uses the same masking strategy as training: all volumes are included,
    but the target volume is masked with Bernoulli mask. Multiple predictions
    with different masks are averaged for robustness.

    Args:
        model: Trained DRCNet-hybrid model
        data: Input data of shape (Vols, X, Y, Z) - full-size data
        device: Device to run inference on
        mask_p: Mask probability (same as training)
        n_preds: Number of predictions per volume (for averaging with different masks)

    Returns:
        Reconstructed data of shape (Vols, X, Y, Z)
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

    # Initialize output array
    sum_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)

    with torch.inference_mode():
        data_device = data.to(device)

        # Process each target volume separately
        for vol_idx in tqdm(range(index, index + 1), desc="Processing volumes"):
            logging.info(f"Processing volume {vol_idx + 1}/{num_vols}")

            # Multiple predictions per volume for robustness (different random masks)
            for pred_idx in range(n_preds):
                # Create masked input (same as training approach)
                # Generate random mask for the target volume
                p_mtx = np.random.uniform(size=spatial_dims)
                mask = (p_mtx > mask_p).astype(np.float32)
                mask_tensor = (
                    torch.tensor(mask).to(device, dtype=torch.float32).unsqueeze(0)
                )

                # Apply mask to target volume only (keep all volumes in input)
                data_masked = data_device.clone()
                data_masked[vol_idx] = data_device[vol_idx] * mask_tensor

                # Add batch dimension: (1, num_vols, X, Y, Z)
                data_masked = data_masked.unsqueeze(0)

                # Forward pass: model expects (B, C, X, Y, Z)
                reconstructed = model(data_masked)

                # Extract the target volume prediction
                # reconstructed shape: (1, 1, X, Y, Z)
                pred_volume = reconstructed.squeeze(0).squeeze(0).detach().cpu().numpy()

                # Accumulate predictions
                sum_preds[vol_idx] += pred_volume

        # Average predictions
        reconstructed = sum_preds / n_preds

        logging.info(f"Reconstruction completed. Output shape: {reconstructed.shape}")
        logging.info(
            f"Output stats - Min: {reconstructed.min():.4f}, "
            f"Max: {reconstructed.max():.4f}, Mean: {reconstructed.mean():.4f}"
        )

    return reconstructed[index : index + 1]


def reconstruct_full_dwi_static_base(
    model,
    noisy_xyzv,
    train_num_volumes,
    device,
    mask_p=0.3,
    n_preds=10,
):
    """
    Reconstruct all DWI volumes by keeping
    ``train_num_volumes`` (must match training ``input_channels``) as the static base
    just sliding over the next volume, calling
    :func:`reconstruct_dwis` and concatenating along the volume axis.

    Args:
        noisy_xyzv: (X, Y, Z, V) all diffusion-weighted volumes after b0s.
        train_num_volumes: Chunk size (model input channels).

    Returns:
        (X, Y, Z, V) with the same V as ``noisy_xyzv``.
    """
    if train_num_volumes < 1:
        raise ValueError("train_num_volumes must be >= 1")
    v = noisy_xyzv.shape[-1]
    chunks_out = []
    # reconstruct the initial volumes taken for training
    block = noisy_xyzv[..., :train_num_volumes]
    x_t = torch.from_numpy(np.transpose(block, (3, 0, 1, 2))).type(torch.float)
    rec_vxyz = reconstruct_dwis(
        model=model,
        data=x_t,
        device=device,
        mask_p=mask_p,
        n_preds=n_preds,
    )
    chunks_out.append(np.transpose(rec_vxyz, (1, 2, 3, 0)))
    for start in range(train_num_volumes, v):
        indices_to_select = np.r_[
            0 : train_num_volumes - 1, start
        ]  # 0:9 gives 0, 1, ..., 8
        block = noisy_xyzv[..., indices_to_select]
        x_t = torch.from_numpy(np.transpose(block, (3, 0, 1, 2))).type(torch.float)
        rec_vxyz = reconstruct_dwis_index_volume(
            model=model,
            data=x_t,
            index=train_num_volumes - 1,
            device=device,
            mask_p=mask_p,
            n_preds=n_preds,
        )
        chunks_out.append(np.transpose(rec_vxyz, (1, 2, 3, 0)))

    full = np.concatenate(chunks_out, axis=-1)
    return full[..., :v]


def reconstruct_full_dwi_chunked(
    model,
    noisy_xyzv,
    train_num_volumes,
    device,
    mask_p=0.3,
    n_preds=10,
):
    """
    Reconstruct all DWI volumes by splitting into contiguous chunks of
    ``train_num_volumes`` (must match training ``input_channels``), calling
    :func:`reconstruct_dwis` per chunk, and concatenating along the volume axis.

    Args:
        noisy_xyzv: (X, Y, Z, V) all diffusion-weighted volumes after b0s.
        train_num_volumes: Chunk size (model input channels).

    Returns:
        (X, Y, Z, V) with the same V as ``noisy_xyzv``.
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
        )
        chunks_out.append(np.transpose(rec_vxyz, (1, 2, 3, 0)))

    full = np.concatenate(chunks_out, axis=-1)
    return full[..., :v]
