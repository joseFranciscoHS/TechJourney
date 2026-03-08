"""
Noise distributions for DWMRI synthetic data.
Used to validate J-invariance denoising across noise types and levels.
All functions expect normalized data in [0, 1] and apply noise per volume (independent across last axis).
"""

import logging

import numpy as np

NOISE_TYPES = ("rician", "gaussian", "noncentral_chi", "ncchi")


def add_noise(data, sigma, noise_type="rician", seed=None, n_coils=1):
    """
    Add synthetic noise to normalized 4D DWI data.

    Noise is applied per volume (independent across the last axis) so that
    J-invariance assumptions (noise independent across dimensions) hold.
    Output is clipped to [0, 1].

    Parameters
    ----------
    data : np.ndarray
        Normalized MRI data, shape (x, y, z, volumes), range [0, 1].
    sigma : float
        Standard deviation of noise (relative to [0, 1] range).
        For Rician/ncchi: sigma of the Gaussian components.
        For Gaussian: sigma of additive noise.
    noise_type : str
        One of "rician", "gaussian", "noncentral_chi", "ncchi".
        "ncchi" is an alias for "noncentral_chi".
    seed : int or None
        If set, numpy RNG is seeded at the start (for reproducibility).
        Note: this affects global RNG state.
    n_coils : int
        Number of coils for "noncentral_chi" / "ncchi". Ignored for other types.
        n_coils=1 matches Rician up to numerical precision.

    Returns
    -------
    noisy_data : np.ndarray
        Data with noise added, same shape and dtype as data, clipped to [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)

    noise_type = noise_type.lower().strip()
    if noise_type == "ncchi":
        noise_type = "noncentral_chi"
    if noise_type not in ("rician", "gaussian", "noncentral_chi"):
        raise ValueError(
            f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}"
        )

    log_msg = (
        f"Adding {noise_type} noise with sigma={sigma} to data of shape {data.shape}"
    )
    if noise_type == "noncentral_chi":
        log_msg += f", n_coils={n_coils}"
    logging.info(log_msg)

    noisy = np.zeros_like(data, dtype=np.float32)
    for vol in range(data.shape[-1]):
        if noise_type == "rician":
            noisy[..., vol] = _add_rician_one_volume(data[..., vol], sigma)
        elif noise_type == "gaussian":
            noisy[..., vol] = _add_gaussian_one_volume(data[..., vol], sigma)
        else:
            noisy[..., vol] = _add_noncentral_chi_one_volume(
                data[..., vol], sigma, n_coils
            )

    noisy = np.clip(noisy, 0, 1).astype(data.dtype, copy=False)
    logging.info(f"Noise added - noisy data shape: {noisy.shape}")
    return noisy


def _add_rician_one_volume(volume, sigma):
    """Rician noise: magnitude of (signal + complex Gaussian). Per-voxel independent."""
    noise_1 = np.random.normal(0, sigma, volume.shape).astype("float32")
    noise_2 = np.random.normal(0, sigma, volume.shape).astype("float32")
    noisy = np.sqrt((volume + noise_1) ** 2 + noise_2**2)
    return noisy


def _add_gaussian_one_volume(volume, sigma):
    """Additive Gaussian on magnitude: signal + N(0, sigma). Per-voxel independent."""
    noise = np.random.normal(0, sigma, volume.shape).astype("float32")
    return volume + noise


def _add_noncentral_chi_one_volume(volume, sigma, n_coils):
    """
    Non-central chi noise for multi-coil magnitude.
    Real per coil = signal/n_coils + N(0, sigma), Imag = N(0, sigma);
    magnitude = sqrt(sum over coils of real^2 + imag^2).
    With n_coils=1 this matches Rician up to numerical precision.
    """
    shape = volume.shape
    # (x, y, z, n_coils): real and imag components per coil
    real = volume[..., np.newaxis] / n_coils + np.random.normal(
        0, sigma, (*shape, n_coils)
    ).astype("float32")
    imag = np.random.normal(0, sigma, (*shape, n_coils)).astype("float32")
    magnitude_sq = (real**2 + imag**2).sum(axis=-1)
    return np.sqrt(np.maximum(magnitude_sq, 0.0))
