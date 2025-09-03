import logging

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti

np.random.seed(91021)


class DBrainDataLoader:
    def __init__(self, nii_path, bvecs_path, bvalue=2500, noise_sigma=0.01):
        self.nii_path = nii_path
        self.bvecs_path = bvecs_path
        self.noise_sigma = noise_sigma
        self.bvalue = bvalue
        logging.info(
            f"DBrainDataLoader initialized - nii_path: {nii_path}, bvecs_path: {bvecs_path}, bvalue: {bvalue}, noise_sigma: {noise_sigma}"
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

        logging.info(f"Adding Rician noise with sigma={self.noise_sigma}...")
        noisy_data = add_rician_noise(data, sigma=self.noise_sigma)
        logging.info(
            f"Noisy data stats - min: {noisy_data.min():.4f}, max: {noisy_data.max():.4f}, mean: {noisy_data.mean():.4f}"
        )

        logging.info("Normalizing spatial dimensions...")
        data_norm_spatial = normalize_spatial_dimensions(data)
        logging.info(
            f"Normalized data stats - min: {data_norm_spatial.min():.4f}, max: {data_norm_spatial.max():.4f}, mean: {data_norm_spatial.mean():.4f}"
        )

        logging.info("Normalizing noisy data spatial dimensions...")
        noisy_data_norm_spatial = normalize_spatial_dimensions(noisy_data)
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


def add_rician_noise(data, sigma):
    logging.info(
        f"Adding Rician noise with sigma={sigma} to data of shape {data.shape}"
    )
    noisy = np.zeros_like(data)
    for vol in range(data.shape[-1]):
        noise_1 = np.random.normal(0, sigma, data[..., vol].shape).astype("float32")
        noise_2 = np.random.normal(0, sigma, data[..., vol].shape).astype("float32")
        noisy[..., vol] = (data[..., vol] + noise_1) ** 2 + noise_2**2
        noisy[..., vol] = noisy[..., vol] ** 0.5
    logging.info(f"Rician noise added - noisy data shape: {noisy.shape}")
    return noisy


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
