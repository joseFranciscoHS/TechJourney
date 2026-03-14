import logging

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu

from .noise import add_noise

np.random.seed(91021)


class DBrainDataLoader:
    def __init__(self, nii_path, bvecs_path, bvalue=2500, noise_sigma=0.01, noise_type="rician", n_coils=1):
        self.nii_path = nii_path
        self.bvecs_path = bvecs_path
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type
        self.n_coils = n_coils
        self.bvalue = bvalue
        logging.info(
            f"DBrainDataLoader initialized - nii_path: {nii_path}, bvecs_path: {bvecs_path}, bvalue: {bvalue}, noise_sigma: {noise_sigma}, noise_type: {noise_type}, n_coils: {n_coils}"
        )

    def load_gradient_table(self):
        logging.info(f"Loading gradient table from {self.bvecs_path}")
        b_matrix = np.loadtxt(self.bvecs_path)
        logging.info(f"B-matrix shape: {b_matrix.shape}")

        bvecs = np.zeros((b_matrix.shape[0], 3))
        bvecs[:, 0] = bvecs[:, 3]  # xy component
        bvecs[:, 1] = bvecs[:, 4]  # xz component
        bvecs[:, 2] = bvecs[:, 5]  # yz component
        # Compute the magnitudes
        magnitudes = np.linalg.norm(bvecs, axis=1)
        # Identify non-zero magnitudes
        non_zero_mask = (
            magnitudes > 1e-8
        )  # Threshold for considering a magnitude as non-zero
        # Normalize only the non-zero vectors
        bvecs[non_zero_mask] = (
            bvecs[non_zero_mask] / magnitudes[non_zero_mask, np.newaxis]
        )
        # Create b-values
        bvals = np.zeros(b_matrix.shape[0])
        bvals[np.sum(b_matrix, axis=1) > 0] = self.bvalue

        # Now create the gradient table
        gtab = gradient_table(bvals, bvecs=bvecs)
        logging.info(
            f"Gradient table created - {len(bvals)} gradients, {np.sum(bvals > 0)} non-zero b-values"
        )
        return gtab

    def load_data(self):
        logging.info(f"Loading data from {self.nii_path}")

        data, _ = load_nifti(self.nii_path)
        logging.info(f"Raw data loaded - shape: {data.shape}, dtype: {data.dtype}")
        logging.info(
            f"Raw data stats - min: {data.min():.4f}, max: {data.max():.4f}, mean: {data.mean():.4f}"
        )

        logging.info("Normalizing spatial dimensions...")
        data_norm_spatial = normalize_spatial_dimensions(data)
        logging.info(
            f"Normalized data stats - min: {data_norm_spatial.min():.4f}, max: {data_norm_spatial.max():.4f}, mean: {data_norm_spatial.mean():.4f}"
        )

        noisy_data_norm_spatial = add_noise(
            data_norm_spatial,
            sigma=self.noise_sigma,
            noise_type=self.noise_type,
            n_coils=self.n_coils,
        )
        logging.info(
            f"Final noisy data stats - min: {noisy_data_norm_spatial.min():.4f}, max: {noisy_data_norm_spatial.max():.4f}, mean: {noisy_data_norm_spatial.mean():.4f}"
        )

        logging.info(f"Data shape: {data_norm_spatial.shape}")
        logging.info(f"Noisy data shape: {noisy_data_norm_spatial.shape}")

        return data_norm_spatial, noisy_data_norm_spatial


class StanfordDataLoader:
    def __init__(self, bvalue=2500, noise_sigma=0.01):
        self.bvalue = bvalue
        self.noise_sigma = noise_sigma
        logging.info(
            f"StanfordDataLoader initialized - bvalue: {bvalue}, noise_sigma: {noise_sigma}"
        )

    def load_gradient_table(self):
        logging.info("Loading Stanford HARDI gradient table...")
        _, hardi_bval_fname, hardi_bvec_fname = get_fnames("stanford_hardi")
        logging.info(
            f"Stanford HARDI files - bval: {hardi_bval_fname}, bvec: {hardi_bvec_fname}"
        )
        bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
        gtab = gradient_table(bvals, bvecs)
        logging.info(
            f"Stanford gradient table created - {len(bvals)} gradients, {np.sum(bvals > 0)} non-zero b-values"
        )
        return gtab

    def load_data(self):
        logging.info("Loading Stanford HARDI data...")
        hardi_fname, _, _ = get_fnames("stanford_hardi")
        logging.info(f"Stanford HARDI data file: {hardi_fname}")
        data, _ = load_nifti(hardi_fname)
        logging.info(f"Stanford data loaded - shape: {data.shape}, dtype: {data.dtype}")
        logging.info(
            f"Stanford data stats - min: {data.min():.4f}, max: {data.max():.4f}, mean: {data.mean():.4f}"
        )

        logging.info("Normalizing Stanford data spatial dimensions...")
        data_norm_spatial = normalize_spatial_dimensions(data)
        logging.info(
            f"Normalized Stanford data stats - min: {data_norm_spatial.min():.4f}, max: {data_norm_spatial.max():.4f}, mean: {data_norm_spatial.mean():.4f}"
        )

        return None, data_norm_spatial


def add_rician_noise_to_normalized(data, sigma):
    """
    Add Rician noise to already normalized data (range [0,1]).
    Backward-compatible wrapper around the noise module.

    Args:
        data: Normalized MRI data of shape (x, y, z, volumes) in range [0,1]
        sigma: Standard deviation of the Gaussian noise components (relative to [0,1] range)

    Returns:
        noisy_data: Data with Rician noise added, clipped to [0, 1]
    """
    return add_noise(data, sigma, noise_type="rician")


def normalize_spatial_dimensions(data):
    logging.info(f"Normalizing spatial dimensions for data of shape {data.shape}")
    # Assuming data shape is (x, y, z, c)
    normalized_data = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[-1]):  # Iterate over each volume
        volume = data[..., i]
        min_val = np.min(volume)
        max_val = np.max(volume)

        # Normalize to [0, 1] range
        normalized_data[..., i] = (volume - min_val) / (max_val - min_val + 1e-6)

    logging.info(
        f"Spatial normalization completed - output shape: {normalized_data.shape}"
    )
    return normalized_data


def rescale_reconstruction_to_01(data, mode="per_volume", reference=None, eps=1e-6):
    """
    Rescale 4D reconstruction (X, Y, Z, V) to [0, 1] per volume.

    Inverse of the per-volume normalization used in preprocessing, so that
    metrics and difference maps (GT vs denoised) are on a comparable scale.

    Args:
        data: 4D array (X, Y, Z, volumes) - reconstructed DWIs (any scale).
        mode: "per_volume" (default) or "match_gt".
            - per_volume: map each volume to [0,1] using its own min/max.
            - match_gt: map each volume to the min/max range of reference[..., i].
        reference: 4D array (X, Y, Z, volumes), required when mode="match_gt"
            (e.g. original_data / ground truth).
        eps: small constant to avoid division by zero.

    Returns:
        Rescaled array, same shape as data, dtype float32, values in [0, 1].
    """
    if data.ndim != 4:
        raise ValueError(f"rescale_reconstruction_to_01 expects 4D data, got ndim={data.ndim}")
    out = np.zeros_like(data, dtype=np.float32)
    n_vols = data.shape[-1]

    if mode == "per_volume":
        for i in range(n_vols):
            vol = data[..., i].astype(np.float64)
            mn, mx = vol.min(), vol.max()
            r = mx - mn + eps
            out[..., i] = np.clip((vol - mn) / r, 0.0, 1.0).astype(np.float32)
        logging.info(
            f"Rescaled reconstruction to [0,1] per_volume: "
            f"global min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}"
        )
    elif mode == "match_gt":
        if reference is None or reference.shape != data.shape:
            raise ValueError(
                "rescale_reconstruction_to_01 with mode='match_gt' requires "
                "reference array of the same shape as data"
            )
        for i in range(n_vols):
            vol = data[..., i].astype(np.float64)
            ref_vol = reference[..., i].astype(np.float64)
            mn_rec, mx_rec = vol.min(), vol.max()
            mn_ref, mx_ref = ref_vol.min(), ref_vol.max()
            r_rec = mx_rec - mn_rec + eps
            r_ref = mx_ref - mn_ref + eps
            # Map rec to [0,1] then to ref range, then clip to [0,1]
            t = (vol - mn_rec) / r_rec
            out[..., i] = np.clip(t * r_ref + mn_ref, 0.0, 1.0).astype(np.float32)
        logging.info(
            f"Rescaled reconstruction to [0,1] match_gt: "
            f"global min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}"
        )
    else:
        raise ValueError(
            f"rescale_reconstruction_to_01 mode must be 'per_volume' or 'match_gt', got {mode!r}"
        )
    return out


def compute_brain_mask(data, median_radius=2, numpass=1):
    """
    Compute brain mask using DIPY's median_otsu on the mean volume.
    
    Args:
        data: 4D array (X, Y, Z, volumes) - clean (non-noisy) normalized data
        median_radius: radius for median filter (default: 2)
        numpass: number of median filter passes (default: 1)
    
    Returns:
        mask: 3D boolean array (X, Y, Z) where True = brain tissue
    """
    logging.info(
        f"Computing brain mask with median_otsu (radius={median_radius}, numpass={numpass})"
    )
    logging.info(f"Input data shape: {data.shape}")
    
    mean_vol = data.mean(axis=-1)
    logging.info(
        f"Mean volume stats - min: {mean_vol.min():.4f}, max: {mean_vol.max():.4f}, mean: {mean_vol.mean():.4f}"
    )
    
    _, mask = median_otsu(mean_vol, median_radius=median_radius, numpass=numpass)
    
    mask_count = mask.sum()
    total_voxels = mask.size
    mask_pct = 100.0 * mask_count / total_voxels
    logging.info(
        f"Brain mask computed: {mask_count:,} voxels in mask ({mask_pct:.1f}% of total)"
    )
    
    return mask

