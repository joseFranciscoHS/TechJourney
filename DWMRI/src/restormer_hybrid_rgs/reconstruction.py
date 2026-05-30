import logging

import numpy as np
import torch
from tqdm import tqdm


def _orientation_info_from_order(order, bvecs, bvals, device, batch_size):
    if bvecs is None or bvals is None:
        return None
    if torch.is_tensor(order):
        order = order.detach().cpu().numpy()
    order = np.asarray(order, dtype=np.int64)
    bvals_max = float(np.max(bvals))
    if bvals_max <= 0:
        bvals_max = 1.0
    orientation_np = np.column_stack([bvecs[order], bvals[order] / bvals_max]).astype(
        np.float32
    )
    return (
        torch.from_numpy(orientation_np)
        .to(device=device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )


def _cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).split(":", maxsplit=1)[0] == "cuda"


def reconstruct_dwis(
    model,
    data,
    device,
    mask_p=0.3,
    n_preds=10,
    patch_size=32,
    overlap=8,
    use_amp=True,
    pred_chunk_size=None,
    bvecs=None,
    bvals=None,
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
        pred_chunk_size: if set, run mask passes in chunks of this batch size to limit
            GPU memory (outputs are summed; same normalization as full batch).

    Returns:
        Reconstructed data of shape (Vols, X, Y, Z) as numpy array
    """
    if pred_chunk_size is not None and int(pred_chunk_size) < 1:
        raise ValueError("pred_chunk_size must be >= 1 when set")
    _chunk = int(pred_chunk_size) if pred_chunk_size is not None else int(n_preds)

    logging.info(f"Starting DWI reconstruction on device: {device}")
    logging.info(f"Input data shape: {data.shape}")
    logging.info(f"Using sliding window: patch_size={patch_size}, overlap={overlap}")
    logging.info(
        f"Mask probability: {mask_p}, n_preds: {n_preds}, "
        f"pred_chunk_size={pred_chunk_size or 'full'} (chunk={_chunk})"
    )
    amp_ok = bool(use_amp) and _cuda_device(device)
    if amp_ok:
        logging.info("Using AMP (float16) for reduced memory during inference")

    model.to(device)
    model.eval()

    if _cuda_device(device):
        torch.cuda.empty_cache()

    num_vols, x_size, y_size, z_size = data.shape
    full_order = np.arange(num_vols, dtype=np.int64)
    stride = patch_size - overlap

    pad_x = (stride - (x_size - patch_size) % stride) % stride
    pad_y = (stride - (y_size - patch_size) % stride) % stride
    pad_z = (stride - (z_size - patch_size) % stride) % stride

    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        data = torch.nn.functional.pad(data, (0, pad_z, 0, pad_y, 0, pad_x), mode="reflect")
        logging.info(f"Padded data to shape: {data.shape}")

    padded_x, padded_y, padded_z = data.shape[1], data.shape[2], data.shape[3]

    sum_preds = np.zeros((num_vols, padded_x, padded_y, padded_z), dtype=np.float32)
    weight_map = np.zeros((padded_x, padded_y, padded_z), dtype=np.float32)

    blend_weights = _create_blend_weights(patch_size, overlap)

    x_positions = list(range(0, padded_x - patch_size + 1, stride))
    y_positions = list(range(0, padded_y - patch_size + 1, stride))
    z_positions = list(range(0, padded_z - patch_size + 1, stride))

    total_patches = len(x_positions) * len(y_positions) * len(z_positions)
    logging.info(
        f"Processing {total_patches} spatial patches per volume "
        f"({n_preds} mask draws batched, chunk={_chunk})"
    )

    for x_start in x_positions:
        for y_start in y_positions:
            for z_start in z_positions:
                weight_map[
                    x_start : x_start + patch_size,
                    y_start : y_start + patch_size,
                    z_start : z_start + patch_size,
                ] += blend_weights

    with torch.inference_mode():
        with tqdm(total=total_patches, desc="Processing spatial patches") as pbar:
            for x_start in x_positions:
                for y_start in y_positions:
                    for z_start in z_positions:
                        x_end = x_start + patch_size
                        y_end = y_start + patch_size
                        z_end = z_start + patch_size
                        patch = (
                            data[:, x_start:x_end, y_start:y_end, z_start:z_end]
                            .clone()
                            .contiguous()
                        )

                        for vol_idx in range(num_vols):
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
                                patch_b = patch.unsqueeze(0).expand(bsz, -1, -1, -1, -1).clone()
                                patch_b[:, vol_idx] = patch[vol_idx].unsqueeze(0) * masks
                                patch_b = patch_b.to(device)
                                orientation_info = _orientation_info_from_order(
                                    full_order, bvecs, bvals, device, bsz
                                )

                                if amp_ok:
                                    with torch.amp.autocast(
                                        device_type="cuda", dtype=torch.float16
                                    ):
                                        pred = model(
                                            patch_b, orientation_info=orientation_info
                                        )
                                else:
                                    pred = model(
                                        patch_b, orientation_info=orientation_info
                                    )

                                part = pred.float().sum(dim=0)
                                pred_sum_t = part if pred_sum_t is None else pred_sum_t + part
                                del patch_b, pred, part

                            pred_np = pred_sum_t.squeeze(0).squeeze(0).cpu().numpy()
                            del pred_sum_t
                            sum_preds[vol_idx, x_start:x_end, y_start:y_end, z_start:z_end] += (
                                pred_np * blend_weights
                            )

                        del patch
                        pbar.update(1)

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
    bvecs=None,
    bvals=None,
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
    if n_preds < 1:
        raise ValueError(f"n_preds must be >= 1, got {n_preds}")
    if n_context < 1:
        raise ValueError(f"n_context must be >= 1, got {n_context}")
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
        data = torch.nn.functional.pad(data, (0, pad_z, 0, pad_y, 0, pad_x), mode="reflect")
        logging.info(f"Padded data to shape: {data.shape}")

    padded_x, padded_y, padded_z = data.shape[1], data.shape[2], data.shape[3]

    sum_preds_t = None
    weight_map_t = None

    blend_weights = _create_blend_weights(patch_size, overlap)
    blend_weights_t = None

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

    # TODO: tech debt — data_dev and sum_preds_t assume the full padded shell fits device memory; add explicit fallback path if OOM appears on lower-VRAM GPUs.
    data_dev = data.to(device)
    blend_weights_t = torch.from_numpy(blend_weights).to(device=device, dtype=torch.float32)
    weight_map_t = torch.zeros((padded_x, padded_y, padded_z), device=device, dtype=torch.float32)
    sum_preds_t = torch.zeros(
        (num_vols, padded_x, padded_y, padded_z), device=device, dtype=torch.float32
    )

    mask_generator = None
    if seed is not None:
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(int(seed))

    for x_start in x_positions:
        for y_start in y_positions:
            for z_start in z_positions:
                weight_map_t[
                    x_start : x_start + patch_size,
                    y_start : y_start + patch_size,
                    z_start : z_start + patch_size,
                ] += blend_weights_t

    with torch.inference_mode():
        others_by_vol = [[i for i in range(num_vols) if i != vol_k] for vol_k in range(num_vols)]
        context_orders_by_vol = []
        for vol_k in range(num_vols):
            others = others_by_vol[vol_k]
            vol_orders = []
            for _ctx in range(n_context):
                ctx = rng.choice(others, size=num_input - 1, replace=False)
                order_buf[:-1] = ctx
                order_buf[-1] = vol_k
                vol_orders.append(
                    torch.from_numpy(order_buf.copy()).to(device=device, dtype=torch.long)
                )
            context_orders_by_vol.append(vol_orders)

        with tqdm(total=total_patches, desc="RGS spatial patches") as pbar:
            for x_start in x_positions:
                for y_start in y_positions:
                    for z_start in z_positions:
                        x_end = x_start + patch_size
                        y_end = y_start + patch_size
                        z_end = z_start + patch_size
                        patch_full = data_dev[:, x_start:x_end, y_start:y_end, z_start:z_end]
                        patch_acc_t = torch.zeros(
                            (num_vols, patch_size, patch_size, patch_size),
                            device=device,
                            dtype=torch.float32,
                        )

                        for vol_k in range(num_vols):
                            for order_t in context_orders_by_vol[vol_k]:
                                patch = patch_full.index_select(0, order_t).contiguous()

                                for start in range(0, n_preds, _chunk):
                                    bsz = min(_chunk, n_preds - start)
                                    masks = (
                                        torch.rand(
                                            bsz,
                                            patch_size,
                                            patch_size,
                                            patch_size,
                                            dtype=torch.float32,
                                            device=device,
                                            generator=mask_generator,
                                        )
                                        > mask_p
                                    ).to(torch.float32)
                                    patch_b = (
                                        patch.unsqueeze(0)
                                        .expand(bsz, -1, -1, -1, -1)
                                        .clone()
                                    )
                                    patch_b[:, target_channel] = (
                                        patch[target_channel].unsqueeze(0) * masks
                                    )
                                    orientation_info = _orientation_info_from_order(
                                        order_t, bvecs, bvals, device, bsz
                                    )

                                    if amp_ok:
                                        with torch.amp.autocast(
                                            device_type="cuda", dtype=torch.float16
                                        ):
                                            pred = model(
                                                patch_b,
                                                orientation_info=orientation_info,
                                            )
                                    else:
                                        pred = model(
                                            patch_b, orientation_info=orientation_info
                                        )

                                    patch_acc_t[vol_k] += pred.float().sum(dim=0).squeeze(0)

                        sum_preds_t[
                            :, x_start:x_end, y_start:y_end, z_start:z_end
                        ] += patch_acc_t * blend_weights_t.unsqueeze(0)
                        pbar.update(1)

    weight_map_t = torch.clamp_min(weight_map_t, 1e-8)
    reconstructed_t = sum_preds_t / (weight_map_t.unsqueeze(0) * denom)
    reconstructed = reconstructed_t[:, :x_size, :y_size, :z_size].detach().cpu().numpy()

    logging.info(
        f"RGS reconstruction done. Min={reconstructed.min():.4f}, Max={reconstructed.max():.4f}, "
        f"Mean={reconstructed.mean():.4f}"
    )
    return reconstructed


def reconstruct_dwis_sequential_sliding_k(
    model,
    data,
    device,
    mask_p=0.3,
    n_preds=10,
    num_input=10,
    target_channel=9,
    patch_size=32,
    overlap=8,
    use_amp=True,
    pred_chunk_size=None,
    bvecs=None,
    bvals=None,
):
    """
    Sequential-K inference over full shell G using contiguous sliding windows.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.astype(np.float32))
    else:
        data = data.float()

    num_vols = data.shape[0]
    if num_input > num_vols:
        raise ValueError(f"num_input K={num_input} exceeds shell size G={num_vols}")
    if not (0 <= target_channel < num_input):
        raise ValueError(f"target_channel={target_channel} must be in [0, {num_input - 1}]")

    num_windows = num_vols - num_input + 1
    data_cpu_np = data.detach().cpu().numpy()
    counts_t = torch.zeros((num_vols,), dtype=torch.float32)
    sums_t = torch.zeros_like(data, dtype=torch.float32)

    window_orders = [
        torch.arange(win_start, win_start + num_input, dtype=torch.long)
        for win_start in range(num_windows)
    ]

    for order_t in window_orders:
        x_win = data.index_select(0, order_t)
        rec_win = reconstruct_dwis(
            model=model,
            data=x_win,
            device=device,
            mask_p=mask_p,
            n_preds=n_preds,
            patch_size=patch_size,
            overlap=overlap,
            use_amp=use_amp,
            pred_chunk_size=pred_chunk_size,
            bvecs=bvecs[order_t.detach().cpu().numpy()] if bvecs is not None else None,
            bvals=bvals[order_t.detach().cpu().numpy()] if bvals is not None else None,
        )
        global_target = int(order_t[target_channel].item())
        sums_t[global_target] += torch.from_numpy(rec_win[target_channel]).to(
            dtype=torch.float32
        )
        counts_t[global_target] += 1.0

    out_full = data_cpu_np.copy()
    counts_np = counts_t.detach().cpu().numpy()
    sums_np = sums_t.detach().cpu().numpy()
    valid = counts_np > 0
    out_full[valid] = sums_np[valid] / counts_np[valid, None, None, None]
    logging.info(
        "Sequential-K reconstruction done. covered=%d/%d volumes",
        int(np.sum(valid)),
        num_vols,
    )
    return out_full


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
