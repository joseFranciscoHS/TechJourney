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
    sum_preds = np.zeros((v, x_size, y_size, z_size), dtype=np.float32)
    denom = float(n_context * n_preds)

    model.to(device)
    model.eval()
    data_dev = data_t.to(device)

    with torch.inference_mode():
        for vol_k in tqdm(range(v), desc="RGS2D volumes"):
            others = [i for i in range(v) if i != vol_k]
            acc = torch.zeros((x_size, y_size, z_size), device=device, dtype=torch.float32)
            for _ in range(n_context):
                ctx = rng.choice(others, size=num_input - 1, replace=False)
                order = np.concatenate([ctx, np.array([vol_k], dtype=np.int64)])
                stack = data_dev[order]
                for zi in range(z_size):
                    patch = stack[:, :, :, zi]
                    for _pred in range(n_preds):
                        p_mtx = rng.random(size=(x_size, y_size))
                        mask_np = (p_mtx > mask_p).astype(np.float32)
                        mask_tensor = torch.tensor(
                            mask_np, device=device, dtype=torch.float32
                        )
                        inp = patch.clone()
                        inp[target_channel] = inp[target_channel] * mask_tensor
                        out = model(inp.unsqueeze(0))
                        pred = out.squeeze(0).squeeze(0)
                        acc[:, :, zi] = acc[:, :, zi] + pred
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
):
    """Sequential-K 2D slice-wise reconstruction (same shell semantics as 3D)."""
    if isinstance(data_vxyz, np.ndarray):
        data_t = torch.from_numpy(data_vxyz.astype(np.float32))
    else:
        data_t = data_vxyz.float()

    v, x_size, y_size, z_size = data_t.shape
    if num_input > v:
        raise ValueError(f"K={num_input} exceeds V={v}")
    num_windows = v - num_input + 1
    sum_preds = np.zeros((v, x_size, y_size, z_size), dtype=np.float32)
    pred_counts = np.zeros((v,), dtype=np.float32)
    data_dev = data_t.to(device)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        for win_start in tqdm(range(num_windows), desc="Sequential2D windows"):
            order = np.arange(win_start, win_start + num_input, dtype=np.int64)
            stack = data_dev[order]
            global_target = int(order[target_channel])
            for zi in range(z_size):
                patch = stack[:, :, :, zi]
                for _pred in range(n_preds):
                    p_mtx = np.random.uniform(size=(x_size, y_size))
                    mask_np = (p_mtx > mask_p).astype(np.float32)
                    mask_tensor = torch.tensor(
                        mask_np, device=device, dtype=torch.float32
                    )
                    inp = patch.clone()
                    inp[target_channel] = inp[target_channel] * mask_tensor
                    out = model(inp.unsqueeze(0))
                    pred = out.squeeze(0).squeeze(0).detach().cpu().numpy()
                    sum_preds[global_target, :, :, zi] += pred
                    pred_counts[global_target] += 1.0

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
