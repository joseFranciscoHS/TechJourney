"""
Reference sklearn-based Patch2Self for 4D DWI data.

Adapted from Kang et al. MD-S2S (Multidimensional Self2Self):
  https://github.com/B9Kang/MD-S2S-Multidimensional-Self2Self/
  blob/main/model_patch2self.py

Core idea (volume hold-out / J-invariance):
  For each target DWI volume f, all *other* volumes supply predictor
  features; an sklearn regressor maps their stacked patch descriptors
  to the central voxel of f.  b0 volumes (bval <= b0_threshold) are
  never used as targets — they are copied unchanged and optionally
  included as extra fixed predictor channels.

Public API::

    denoised = patch2self_sklearn(
        data_4d,          # (X, Y, Z, V)  float32/64, values in [0,1]
        bvals,            # 1-D array length V
        b0_threshold=50,
        model_name="ols",
        patch_radius=(0, 0, 0),
        stride=1,
        use_b0_as_predictors=True,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Sequence, Tuple

import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def _extract_3d_patches(
    arr: np.ndarray,
    patch_radius: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Return shape ``(V, n_patches, n_features)`` patch matrix.

    Parameters
    ----------
    arr:
        Padded array (X+2rx, Y+2ry, Z+2rz, V).
    patch_radius:
        3-element int array ``[rx, ry, rz]``.
    stride:
        Step between sampled patch centres (1 = dense).
    """
    rx, ry, rz = int(patch_radius[0]), int(patch_radius[1]), int(patch_radius[2])
    patch_size = 2 * patch_radius + 1
    n_feat = int(np.prod(patch_size))
    n_vols = arr.shape[3]

    all_patches = []
    for i in range(rx, arr.shape[0] - rx, stride):
        for j in range(ry, arr.shape[1] - ry, stride):
            for k in range(rz, arr.shape[2] - rz, stride):
                block = arr[
                    i - rx : i + rx + 1,
                    j - ry : j + ry + 1,
                    k - rz : k + rz + 1,
                    :,
                ]  # (px, py, pz, V)
                all_patches.append(block.reshape(n_feat, n_vols))

    # (n_patches, n_feat, V) → (V, n_patches, n_feat)
    stacked = np.array(all_patches)  # (n_patches, n_feat, V)
    return stacked.transpose(2, 0, 1)  # (V, n_patches, n_feat)


def _sampled_grid_shape(
    padded_shape: Tuple[int, int, int],
    patch_radius: np.ndarray,
    stride: int,
) -> Tuple[int, int, int]:
    """Return (x_out, y_out, z_out) after strided patch-centre sampling."""
    rx, ry, rz = int(patch_radius[0]), int(patch_radius[1]), int(patch_radius[2])
    x_out = len(range(rx, padded_shape[0] - rx, stride))
    y_out = len(range(ry, padded_shape[1] - ry, stride))
    z_out = len(range(rz, padded_shape[2] - rz, stride))
    return x_out, y_out, z_out


# ---------------------------------------------------------------------------
# Sklearn model factory
# ---------------------------------------------------------------------------


def _build_model(model_name: str):
    """Instantiate an sklearn regressor by name."""
    name = model_name.lower()
    if name == "ols":
        return linear_model.LinearRegression(
            copy_X=False, fit_intercept=True, n_jobs=-1
        )
    if name == "ridge":
        return linear_model.Ridge()
    if name == "lasso":
        return linear_model.Lasso(max_iter=50)
    if name == "mlp":
        return MLPRegressor(
            activation="relu",
            hidden_layer_sizes=(64, 64, 64, 64),
            learning_rate_init=1e-4,
            early_stopping=True,
            random_state=1,
            max_iter=500,
        )
    raise ValueError(
        f"Unknown sklearn_model '{model_name}'. Choose from: ols, ridge, lasso, mlp"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch2self_sklearn(
    data_4d: np.ndarray,
    bvals: np.ndarray,
    b0_threshold: int = 50,
    model_name: str = "ols",
    patch_radius: Sequence[int] = (0, 0, 0),
    stride: int = 1,
    use_b0_as_predictors: bool = True,
) -> np.ndarray:
    """Denoise a 4D DWI array via volume hold-out sklearn regression.

    b0 volumes (``bval <= b0_threshold``) are copied unchanged.  Each DWI
    volume is denoised by fitting a fresh sklearn regressor on all other
    volumes as predictors, then predicting the held-out volume's patches.

    This follows the exact ``fit(X.T, Y.T); predict(X.T)`` convention from
    the MD-S2S reference, where:

    * ``X`` shape ``(n_feat, n_other * n_patches)`` — patch features of the
      predictor volumes stacked side by side.
    * ``Y`` shape ``(n_patches,)`` — central feature of the target volume.

    Parameters
    ----------
    data_4d:
        Shape ``(X, Y, Z, V)``, values in ``[0, 1]``, float32 or float64.
    bvals:
        1-D array length V, aligned to the last axis of ``data_4d``.
    b0_threshold:
        Volumes with ``bval <= b0_threshold`` are treated as b0.
    model_name:
        ``"ols"`` | ``"ridge"`` | ``"lasso"`` | ``"mlp"``
    patch_radius:
        Half-width along ``(X, Y, Z)``.  ``(0, 0, 0)`` = single-voxel
        patches, i.e. pure inter-volume regression.
    stride:
        Spatial grid step.  ``1`` = every voxel (dense).  Values > 1
        subsample the grid to trade quality for speed and memory.
    use_b0_as_predictors:
        Include b0 volumes as fixed predictor channels for every DWI target.

    Returns
    -------
    denoised_4d:
        Shape ``(X, Y, Z, V)``, float32.  b0 volumes are unmodified.
    """
    data_4d = data_4d.astype(np.float64)
    X_dim, Y_dim, Z_dim, V = data_4d.shape
    patch_radius = np.asarray(patch_radius, dtype=int)

    dwi_mask = bvals > b0_threshold
    b0_mask = ~dwi_mask
    n_b0 = int(np.sum(b0_mask))
    n_dwi = int(np.sum(dwi_mask))

    logging.info(
        "patch2self_sklearn: %d volumes — %d b0 (unchanged), %d DWI",
        V,
        n_b0,
        n_dwi,
    )
    logging.info(
        "  model=%s  patch_radius=%s  stride=%d  b0_as_predictors=%s",
        model_name,
        patch_radius.tolist(),
        stride,
        use_b0_as_predictors,
    )

    if n_dwi == 0:
        logging.warning("No DWI volumes found; returning input unchanged.")
        return data_4d.astype(np.float32)

    # Pad so every interior voxel can be a patch centre
    rx, ry, rz = int(patch_radius[0]), int(patch_radius[1]), int(patch_radius[2])
    padded = np.pad(
        data_4d,
        ((rx, rx), (ry, ry), (rz, rz), (0, 0)),
        mode="constant",
    )
    x_out, y_out, z_out = _sampled_grid_shape(
        (padded.shape[0], padded.shape[1], padded.shape[2]),
        patch_radius,
        stride,
    )
    n_patches = x_out * y_out * z_out
    n_feat = int(np.prod(2 * patch_radius + 1))

    logging.info(
        "  Patch grid: %d centres (%dx%dx%d), n_feat=%d",
        n_patches,
        x_out,
        y_out,
        z_out,
        n_feat,
    )

    # Select which volumes go into the predictor pool
    dwi_indices = np.where(dwi_mask)[0]
    b0_indices = np.where(b0_mask)[0]

    if use_b0_as_predictors and n_b0 > 0:
        pred_vol_indices = np.concatenate([b0_indices, dwi_indices])
    else:
        pred_vol_indices = dwi_indices

    n_b0_pred = n_b0 if (use_b0_as_predictors and n_b0 > 0) else 0

    # Extract patches for all predictor volumes at once
    t0 = time.time()
    logging.info("Extracting patches …")
    train = _extract_3d_patches(
        padded[..., pred_vol_indices],
        patch_radius,
        stride,
    )  # (n_pred, n_patches, n_feat)
    logging.info(
        "  Patch extraction done in %.1fs, train shape=%s",
        time.time() - t0,
        train.shape,
    )

    denoised_4d = data_4d.copy()  # b0 volumes keep their original values

    # Strided grid positions (relative to un-padded array)
    xi = np.array(range(rx, padded.shape[0] - rx, stride)) - rx
    yi = np.array(range(ry, padded.shape[1] - ry, stride)) - ry
    zi = np.array(range(rz, padded.shape[2] - rz, stride)) - rz

    for local_f, global_f in enumerate(dwi_indices):
        t_vol = time.time()

        # Position of this volume in the train tensor
        train_f = n_b0_pred + local_f

        # Predictor volumes: b0 block (if any) + all DWI except train_f
        if n_b0_pred > 0:
            other = np.concatenate(
                [train[:n_b0_pred], train[n_b0_pred:train_f], train[train_f + 1 :]],
                axis=0,
            )
        else:
            other = np.concatenate(
                [train[:train_f], train[train_f + 1 :]],
                axis=0,
            )
        # other: (n_other, n_patches, n_feat)

        # Supervision: central feature of the held-out volume
        Y_target = train[train_f, :, n_feat // 2]  # (n_patches,)

        # TODO: tech debt — validate this sample/feature layout against the original MD-S2S reference on real D-Brain to confirm parity in final metrics.
        # Build design matrix per voxel/patch center:
        # X rows are patch centers; columns are features from all predictor volumes.
        # This keeps sample axis aligned with Y_target (n_patches).
        cur_X = np.transpose(other, (1, 0, 2)).reshape(n_patches, -1)

        reg = _build_model(model_name)
        reg.fit(cur_X, Y_target)
        pred_flat = reg.predict(cur_X)

        # Place predictions back into the correct spatial positions
        xs = np.clip(xi, 0, X_dim - 1)
        ys = np.clip(yi, 0, Y_dim - 1)
        zs = np.clip(zi, 0, Z_dim - 1)
        denoised_4d[
            xs[:, None, None],
            ys[None, :, None],
            zs[None, None, :],
            global_f,
        ] = pred_flat.reshape(x_out, y_out, z_out)

        logging.info(
            "  Volume %d/%d (global %d) denoised in %.1fs",
            local_f + 1,
            n_dwi,
            global_f,
            time.time() - t_vol,
        )

    logging.info(
        "patch2self_sklearn done — range [%.4f, %.4f], mean %.4f",
        denoised_4d.min(),
        denoised_4d.max(),
        denoised_4d.mean(),
    )
    return denoised_4d.astype(np.float32)
