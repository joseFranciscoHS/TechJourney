import logging

import numpy as np
import torch
from tqdm import tqdm


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
    pred_chunk_size=None,
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

        pred_chunk_size: Optional micro-batch size for parallel spatial-mask
            predictions inside each context. If None, uses n_preds.

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

    if n_preds < 1:
        raise ValueError(f"n_preds must be >= 1, got {n_preds}")
    if n_context < 1:
        raise ValueError(f"n_context must be >= 1, got {n_context}")
    if pred_chunk_size is None:
        pred_chunk_size = n_preds
    pred_chunk_size = int(pred_chunk_size)
    if pred_chunk_size < 1:
        raise ValueError(f"pred_chunk_size must be >= 1, got {pred_chunk_size}")

    sum_preds = np.zeros((num_vols, *spatial_dims), dtype=np.float32)
    denom = float(n_context * n_preds)

    model.to(device)
    model.eval()
    data_dev = data_t.to(device)

    mask_generator = None
    if seed is not None:
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(int(seed))

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
                done = 0
                while done < n_preds:
                    chunk = min(pred_chunk_size, n_preds - done)
                    masks = (
                        torch.rand(
                            (chunk, x_size, y_size, z_size),
                            device=device,
                            generator=mask_generator,
                        )
                        > mask_p
                    ).to(torch.float32)
                    # NOTE: clone() materializes an independent batch so each item can
                    # receive a different spatial mask on target_channel.
                    inp_b = stack.unsqueeze(0).expand(chunk, -1, -1, -1, -1).clone()
                    inp_b[:, target_channel] = inp_b[:, target_channel] * masks
                    out = model(inp_b).squeeze(1)  # (chunk, X, Y, Z)
                    acc = acc + out.sum(dim=0)
                    done += chunk
            sum_preds[vol_k] = (acc / denom).detach().cpu().numpy()

    logging.info(
        f"RGS reconstruction done. Min={sum_preds.min():.4f}, Max={sum_preds.max():.4f}, "
        f"Mean={sum_preds.mean():.4f}"
    )
    return sum_preds


def reconstruct_dwis_sequential_sliding_k(
    model,
    data,
    device,
    mask_p=0.3,
    n_preds=10,
    num_input=10,
    target_channel=9,
    seed=None,
    pred_chunk_size=None,
):
    """
    Sequential-K inference over full shell G using sliding windows.

    For each contiguous gradient window of size K, only the slot ``target_channel``
    (typically K-1) is reconstructed. Outputs are written to the global shell
    index corresponding to that slot.
    """
    if isinstance(data, np.ndarray):
        data_t = torch.from_numpy(data.astype(np.float32))
    else:
        data_t = data.float()

    num_vols, x_size, y_size, z_size = data_t.shape
    if num_input > num_vols:
        raise ValueError(f"num_input K={num_input} exceeds shell size G={num_vols}")
    if not (0 <= target_channel < num_input):
        raise ValueError(
            f"target_channel={target_channel} must be in [0, {num_input - 1}]"
        )
    if n_preds < 1:
        raise ValueError(f"n_preds must be >= 1, got {n_preds}")
    if pred_chunk_size is None:
        pred_chunk_size = n_preds
    pred_chunk_size = int(pred_chunk_size)
    if pred_chunk_size < 1:
        raise ValueError(f"pred_chunk_size must be >= 1, got {pred_chunk_size}")

    num_windows = num_vols - num_input + 1
    sum_preds = np.zeros((num_vols, x_size, y_size, z_size), dtype=np.float32)
    pred_counts = np.zeros((num_vols,), dtype=np.float32)
    data_dev = data_t.to(device)
    model.to(device)
    model.eval()
    mask_generator = None
    if seed is not None:
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(int(seed))

    with torch.inference_mode():
        for win_start in tqdm(range(num_windows), desc="Sequential-K windows"):
            order = np.arange(win_start, win_start + num_input, dtype=np.int64)
            stack = data_dev[order]
            global_target = int(order[target_channel])
            done = 0
            while done < n_preds:
                chunk = min(pred_chunk_size, n_preds - done)
                masks = (
                    torch.rand(
                        (chunk, x_size, y_size, z_size),
                        device=device,
                        generator=mask_generator,
                    )
                    > mask_p
                ).to(torch.float32)
                # NOTE: clone() is required because each batch item gets a different mask.
                inp_b = stack.unsqueeze(0).expand(chunk, -1, -1, -1, -1).clone()
                inp_b[:, target_channel] = inp_b[:, target_channel] * masks
                out = model(inp_b).squeeze(1)  # (chunk, X, Y, Z)
                sum_preds[global_target] += out.sum(dim=0).detach().cpu().numpy()
                pred_counts[global_target] += float(chunk)
                done += chunk

    # Preserve edge channels not covered by target_channel windows.
    out_full = data_t.detach().cpu().numpy().copy()
    valid = pred_counts > 0
    out_full[valid] = sum_preds[valid] / pred_counts[valid, None, None, None]
    logging.info(
        "Sequential-K reconstruction done. covered=%d/%d volumes",
        int(np.sum(valid)),
        num_vols,
    )
    return out_full
