"""2D slice-wise reconstruction (Monte Carlo) for RGS and sequential-K shells."""

from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm


def reconstruct_dwis_rgs_2d(
    model,
    data_vxyz,
    device,
    mask_p=0.3,
    n_preds=10,
    n_context=10,
    target_channel=9,
    num_input=10,
    seed=None,
    pred_chunk_size=None,
    slice_chunk_size=None,
):
    """
    RGS inference with a 2D model: for each volume ``k`` and each z-slice, average
    MC predictions matching ``reconstruct_dwis_rgs`` semantics per slice.

    Args:
        data_vols: (V, X, Y, Z) on CPU.
    """
    rng = np.random.default_rng(seed)
    if isinstance(data_vxyz, np.ndarray):
        data_t = torch.from_numpy(data_vxyz.astype(np.float32))
    else:
        data_t = data_vxyz.float()

    v, x_size, y_size, z_size = data_t.shape
    if n_preds < 1:
        raise ValueError(f"n_preds must be >= 1, got {n_preds}")
    if n_context < 1:
        raise ValueError(f"n_context must be >= 1, got {n_context}")
    if pred_chunk_size is None:
        pred_chunk_size = n_preds
    pred_chunk_size = int(pred_chunk_size)
    if pred_chunk_size < 1:
        raise ValueError(f"pred_chunk_size must be >= 1, got {pred_chunk_size}")
    if slice_chunk_size is None:
        slice_chunk_size = z_size
    slice_chunk_size = int(slice_chunk_size)
    if slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
    sum_preds = np.zeros((v, x_size, y_size, z_size), dtype=np.float32)
    denom = float(n_context * n_preds)

    model.to(device)
    model.eval()
    data_dev = data_t.to(device)
    mask_generator = None
    if seed is not None:
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(int(seed))

    with torch.inference_mode():
        for vol_k in tqdm(range(v), desc="RGS2D volumes"):
            others = [i for i in range(v) if i != vol_k]
            acc = torch.zeros((x_size, y_size, z_size), device=device, dtype=torch.float32)
            for _ in range(n_context):
                ctx = rng.choice(others, size=num_input - 1, replace=False)
                order = np.concatenate([ctx, np.array([vol_k], dtype=np.int64)])
                stack = data_dev[order]
                for z_start in range(0, z_size, slice_chunk_size):
                    z_end = min(z_start + slice_chunk_size, z_size)
                    patch_chunk = stack[:, :, :, z_start:z_end].permute(
                        3, 0, 1, 2
                    )  # (slice_chunk, K, X, Y)
                    done = 0
                    while done < n_preds:
                        pred_chunk = min(pred_chunk_size, n_preds - done)
                        slice_chunk = patch_chunk.shape[0]
                        masks = (
                            torch.rand(
                                (pred_chunk, slice_chunk, x_size, y_size),
                                device=device,
                                generator=mask_generator,
                            )
                            > mask_p
                        ).to(torch.float32)
                        inp = (
                            patch_chunk.unsqueeze(0)
                            .expand(pred_chunk, -1, -1, -1, -1)
                            .clone()
                        )
                        inp[:, :, target_channel] = inp[:, :, target_channel] * masks
                        inp_b = inp.reshape(pred_chunk * slice_chunk, num_input, x_size, y_size)
                        out = model(inp_b).squeeze(1)
                        out = out.reshape(pred_chunk, slice_chunk, x_size, y_size)
                        acc[:, :, z_start:z_end] += out.sum(dim=0).permute(1, 2, 0)
                        done += pred_chunk
            sum_preds[vol_k] = (acc / denom).detach().cpu().numpy()

    logging.info(
        "RGS2D reconstruction done. min=%.4f max=%.4f mean=%.4f",
        float(sum_preds.min()),
        float(sum_preds.max()),
        float(sum_preds.mean()),
    )
    return sum_preds


def reconstruct_dwis_sequential_sliding_k_2d(
    model,
    data_vxyz,
    device,
    mask_p=0.3,
    n_preds=10,
    num_input=10,
    target_channel=9,
    seed=None,
    pred_chunk_size=None,
    slice_chunk_size=None,
):
    """Sequential-K 2D slice-wise reconstruction (same shell semantics as 3D)."""
    if isinstance(data_vxyz, np.ndarray):
        data_t = torch.from_numpy(data_vxyz.astype(np.float32))
    else:
        data_t = data_vxyz.float()

    v, x_size, y_size, z_size = data_t.shape
    if num_input > v:
        raise ValueError(f"K={num_input} exceeds V={v}")
    if n_preds < 1:
        raise ValueError(f"n_preds must be >= 1, got {n_preds}")
    if pred_chunk_size is None:
        pred_chunk_size = n_preds
    pred_chunk_size = int(pred_chunk_size)
    if pred_chunk_size < 1:
        raise ValueError(f"pred_chunk_size must be >= 1, got {pred_chunk_size}")
    if slice_chunk_size is None:
        slice_chunk_size = z_size
    slice_chunk_size = int(slice_chunk_size)
    if slice_chunk_size < 1:
        raise ValueError(f"slice_chunk_size must be >= 1, got {slice_chunk_size}")
    num_windows = v - num_input + 1
    sum_preds = np.zeros((v, x_size, y_size, z_size), dtype=np.float32)
    pred_counts = np.zeros((v,), dtype=np.float32)
    data_dev = data_t.to(device)
    model.to(device)
    model.eval()
    mask_generator = None
    if seed is not None:
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(int(seed))

    with torch.inference_mode():
        for win_start in tqdm(range(num_windows), desc="Sequential2D windows"):
            order = np.arange(win_start, win_start + num_input, dtype=np.int64)
            stack = data_dev[order]
            global_target = int(order[target_channel])
            for z_start in range(0, z_size, slice_chunk_size):
                z_end = min(z_start + slice_chunk_size, z_size)
                patch_chunk = stack[:, :, :, z_start:z_end].permute(
                    3, 0, 1, 2
                )  # (slice_chunk, K, X, Y)
                done = 0
                while done < n_preds:
                    pred_chunk = min(pred_chunk_size, n_preds - done)
                    slice_chunk = patch_chunk.shape[0]
                    masks = (
                        torch.rand(
                            (pred_chunk, slice_chunk, x_size, y_size),
                            device=device,
                            generator=mask_generator,
                        )
                        > mask_p
                    ).to(torch.float32)
                    inp = (
                        patch_chunk.unsqueeze(0)
                        .expand(pred_chunk, -1, -1, -1, -1)
                        .clone()
                    )
                    inp[:, :, target_channel] = inp[:, :, target_channel] * masks
                    inp_b = inp.reshape(pred_chunk * slice_chunk, num_input, x_size, y_size)
                    out = model(inp_b).squeeze(1)
                    out = out.reshape(pred_chunk, slice_chunk, x_size, y_size)
                    chunk_sum = out.sum(dim=0).permute(1, 2, 0).detach().cpu().numpy()
                    sum_preds[global_target, :, :, z_start:z_end] += chunk_sum
                    pred_counts[global_target] += float(pred_chunk * slice_chunk)
                    done += pred_chunk

    out_full = data_t.detach().cpu().numpy().copy()
    valid = pred_counts > 0
    for i in np.where(valid)[0]:
        c = pred_counts[i]
        out_full[i] = sum_preds[i] / c
    logging.info(
        "Sequential2D reconstruction done. covered=%d/%d",
        int(np.sum(valid)),
        v,
    )
    return out_full
