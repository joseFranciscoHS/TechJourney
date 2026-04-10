import logging

import numpy as np
import torch
from tqdm import tqdm


def _cuda_device(device) -> bool:
    return isinstance(device, str) and device[:4] == "cuda"


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
    amp_ok = bool(use_amp) and _cuda_device(device)
    if amp_ok:
        logging.info("Using AMP (float16) for reduced memory during inference")

    model.to(device)
    model.eval()

    if _cuda_device(device):
        torch.cuda.empty_cache()

    num_vols, x_size, y_size, z_size = data.shape
    stride = patch_size - overlap

    pad_x = (stride - (x_size - patch_size) % stride) % stride
    pad_y = (stride - (y_size - patch_size) % stride) % stride
    pad_z = (stride - (z_size - patch_size) % stride) % stride

    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        data = torch.nn.functional.pad(
            data, (0, pad_z, 0, pad_y, 0, pad_x), mode="reflect"
        )
        logging.info(f"Padded data to shape: {data.shape}")

    padded_x, padded_y, padded_z = data.shape[1], data.shape[2], data.shape[3]

    sum_preds = np.zeros((num_vols, padded_x, padded_y, padded_z), dtype=np.float32)
    weight_map = np.zeros((padded_x, padded_y, padded_z), dtype=np.float32)

    blend_weights = _create_blend_weights(patch_size, overlap)

    x_positions = list(range(0, padded_x - patch_size + 1, stride))
    y_positions = list(range(0, padded_y - patch_size + 1, stride))
    z_positions = list(range(0, padded_z - patch_size + 1, stride))

    total_patches = len(x_positions) * len(y_positions) * len(z_positions)
    logging.info(f"Processing {total_patches} patches per volume per prediction")

    with torch.inference_mode():
        for vol_idx in tqdm(range(num_vols), desc="Processing volumes"):
            logging.info(f"Processing volume {vol_idx + 1}/{num_vols}")

            for pred_idx in range(n_preds):
                for x_start in x_positions:
                    for y_start in y_positions:
                        for z_start in z_positions:
                            x_end = x_start + patch_size
                            y_end = y_start + patch_size
                            z_end = z_start + patch_size

                            patch = data[
                                :, x_start:x_end, y_start:y_end, z_start:z_end
                            ].clone()

                            mask = (
                                torch.rand(
                                    (patch_size, patch_size, patch_size),
                                    dtype=torch.float32,
                                )
                                > mask_p
                            ).float()
                            patch[vol_idx] = patch[vol_idx] * mask

                            patch = patch.unsqueeze(0).to(device)

                            if amp_ok:
                                with torch.amp.autocast(
                                    device_type="cuda", dtype=torch.float16
                                ):
                                    pred = model(patch)
                            else:
                                pred = model(patch)

                            pred_np = pred.float().squeeze(0).squeeze(0).cpu().numpy()
                            sum_preds[
                                vol_idx, x_start:x_end, y_start:y_end, z_start:z_end
                            ] += (pred_np * blend_weights)

                            del patch, pred

                if pred_idx == 0:
                    for x_start in x_positions:
                        for y_start in y_positions:
                            for z_start in z_positions:
                                weight_map[
                                    x_start : x_start + patch_size,
                                    y_start : y_start + patch_size,
                                    z_start : z_start + patch_size,
                                ] += blend_weights

            if _cuda_device(device):
                torch.cuda.empty_cache()

    weight_map = np.maximum(weight_map, 1e-8)
    reconstructed = sum_preds / (weight_map[np.newaxis, ...] * n_preds)

    reconstructed = reconstructed[:, :x_size, :y_size, :z_size]

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
    patch_size=32,
    overlap=8,
    use_amp=True,
    seed=None,
    pred_chunk_size=None,
):
    """
    RGS–Hybrid inference with patch-based sliding windows (Restormer memory budget).

    For each shell index ``vol_k``, Monte-Carlo average over random (K-1)-tuples of
    context indices with volume ``vol_k`` fixed at ``target_channel`` in the K-stack,
    times spatial Bernoulli masks (``n_preds``) per context, accumulated and normalized
    with the same Gaussian blend weights as :func:`reconstruct_dwis`.

    Args:
        data: (Vols, X, Y, Z) float tensor or numpy on CPU.
        n_context: outer MC passes (random context tuples + vol_k at target_channel).
        n_preds: inner spatial Bernoulli mask passes per context (batched on GPU).
        target_channel: 0-based slot where ``vol_k`` is placed (typically K-1).
        num_input: K (must match model input channels).
        pred_chunk_size: if set, run mask passes in chunks of this batch size to limit
            GPU memory (outputs are summed; same normalization as full batch).

    Returns:
        Reconstructed array (Vols, X, Y, Z) as numpy float32.
    """
    if pred_chunk_size is not None and int(pred_chunk_size) < 1:
        raise ValueError("pred_chunk_size must be >= 1 when set")
    _chunk = int(pred_chunk_size) if pred_chunk_size is not None else int(n_preds)

    logging.info(
        f"RGS patch reconstruction: patch_size={patch_size}, overlap={overlap}, "
        f"mask_p={mask_p}, n_context={n_context}, n_preds={n_preds}, "
        f"pred_chunk_size={pred_chunk_size or 'full'}, "
        f"target_channel={target_channel}, K={num_input}"
    )
    if num_input != target_channel + 1:
        logging.warning(
            f"target_channel={target_channel} with K={num_input}: typical parity uses "
            f"target_channel=K-1 (last slot = masked volume)."
        )

    amp_ok = bool(use_amp) and _cuda_device(device)

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.astype(np.float32))
    else:
        data = data.float()

    num_vols, x_size, y_size, z_size = data.shape
    stride = patch_size - overlap

    pad_x = (stride - (x_size - patch_size) % stride) % stride
    pad_y = (stride - (y_size - patch_size) % stride) % stride
    pad_z = (stride - (z_size - patch_size) % stride) % stride

    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        data = torch.nn.functional.pad(
            data, (0, pad_z, 0, pad_y, 0, pad_x), mode="reflect"
        )
        logging.info(f"Padded data to shape: {data.shape}")

    padded_x, padded_y, padded_z = data.shape[1], data.shape[2], data.shape[3]

    sum_preds = np.zeros((num_vols, padded_x, padded_y, padded_z), dtype=np.float32)
    weight_map = np.zeros((padded_x, padded_y, padded_z), dtype=np.float32)

    blend_weights = _create_blend_weights(patch_size, overlap)

    x_positions = list(range(0, padded_x - patch_size + 1, stride))
    y_positions = list(range(0, padded_y - patch_size + 1, stride))
    z_positions = list(range(0, padded_z - patch_size + 1, stride))

    total_patches = len(x_positions) * len(y_positions) * len(z_positions)
    denom = float(n_context * n_preds)
    logging.info(
        f"RGS: {total_patches} spatial patches per context "
        f"({n_preds} mask draws batched, chunk={_chunk}); "
        f"normalizer weight_map * {denom}"
    )

    rng = np.random.default_rng(seed)
    order_buf = np.empty((num_input,), dtype=np.int64)

    model.to(device)
    model.eval()

    if _cuda_device(device):
        torch.cuda.empty_cache()

    for x_start in x_positions:
        for y_start in y_positions:
            for z_start in z_positions:
                weight_map[
                    x_start : x_start + patch_size,
                    y_start : y_start + patch_size,
                    z_start : z_start + patch_size,
                ] += blend_weights

    with torch.inference_mode():
        for vol_k in tqdm(range(num_vols), desc="RGS volumes"):
            others = [i for i in range(num_vols) if i != vol_k]

            for _ctx in range(n_context):
                ctx = rng.choice(others, size=num_input - 1, replace=False)
                order_buf[:-1] = ctx
                order_buf[-1] = vol_k
                order_t = torch.from_numpy(order_buf.copy()).long()

                for x_start in x_positions:
                    for y_start in y_positions:
                        for z_start in z_positions:
                            x_end = x_start + patch_size
                            y_end = y_start + patch_size
                            z_end = z_start + patch_size

                            patch = (
                                data[
                                    order_t,
                                    x_start:x_end,
                                    y_start:y_end,
                                    z_start:z_end,
                                ]
                                .clone()
                                .contiguous()
                            )

                            pred_sum_t = None
                            for start in range(0, n_preds, _chunk):
                                bsz = min(_chunk, n_preds - start)
                                masks = (
                                    torch.rand(
                                        bsz,
                                        patch_size,
                                        patch_size,
                                        patch_size,
                                        dtype=torch.float32,
                                        device=patch.device,
                                    )
                                    > mask_p
                                ).float()
                                patch_b = patch.unsqueeze(0).expand(
                                    bsz, -1, -1, -1, -1
                                ).clone()
                                patch_b[:, target_channel] = (
                                    patch[target_channel].unsqueeze(0) * masks
                                )
                                patch_b = patch_b.to(device)

                                if amp_ok:
                                    with torch.amp.autocast(
                                        device_type="cuda", dtype=torch.float16
                                    ):
                                        pred = model(patch_b)
                                else:
                                    pred = model(patch_b)

                                part = pred.float().sum(dim=0)
                                pred_sum_t = (
                                    part if pred_sum_t is None else pred_sum_t + part
                                )
                                del patch_b, pred, part

                            pred_np = (
                                pred_sum_t.squeeze(0).squeeze(0).cpu().numpy()
                            )
                            del pred_sum_t
                            sum_preds[
                                vol_k,
                                x_start:x_end,
                                y_start:y_end,
                                z_start:z_end,
                            ] += pred_np * blend_weights

                            del patch

                if _cuda_device(device):
                    torch.cuda.empty_cache()

    weight_map = np.maximum(weight_map, 1e-8)
    reconstructed = sum_preds / (weight_map[np.newaxis, ...] * denom)
    reconstructed = reconstructed[:, :x_size, :y_size, :z_size]

    logging.info(
        f"RGS reconstruction done. Min={reconstructed.min():.4f}, Max={reconstructed.max():.4f}, "
        f"Mean={reconstructed.mean():.4f}"
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
    sigma_ratio = 0.3 if overlap > 0 else 0.5

    center = (patch_size - 1) / 2.0
    sigma = patch_size * sigma_ratio

    x = np.arange(patch_size) - center
    gaussian_1d = np.exp(-0.5 * (x / sigma) ** 2)

    weights = np.einsum("i,j,k->ijk", gaussian_1d, gaussian_1d, gaussian_1d)

    return weights.astype(np.float32)
