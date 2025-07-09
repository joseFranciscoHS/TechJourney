import logging

import numpy as np
import torch
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from torch.utils.data import Dataset

np.random.seed(91021)


class DBrainDataLoader:
    def __init__(self, nii_path, bvecs_path, bvalue=2500, noise_sigma=0.01):
        self.nii_path = nii_path
        self.bvecs_path = bvecs_path
        self.noise_sigma = noise_sigma
        self.bvalue = bvalue

    def load_gradient_table(self):
        b_matrix = np.loadtxt(self.bvecs_path)

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

        return gtab

    def load_data(self):
        logging.info(f"Loading data from {self.nii_path}")

        data, _ = load_nifti(self.nii_path)

        data_norm_spatial = normalize_spatial_dimensions(data)
        noisy_data = add_rician_noise(data_norm_spatial, sigma=self.noise_sigma)
        noisy_data_norm_spatial = normalize_spatial_dimensions(noisy_data)

        logging.info(f"Data shape: {data_norm_spatial.shape}")
        logging.info(f"Noisy data shape: {noisy_data_norm_spatial.shape}")

        return data_norm_spatial, noisy_data_norm_spatial


class StanfordDataLoader:
    def __init__(self, bvalue=2500, noise_sigma=0.01):
        self.bvalue = bvalue
        self.noise_sigma = noise_sigma

    def load_gradient_table(self):
        _, hardi_bval_fname, hardi_bvec_fname = get_fnames("stanford_hardi")
        bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
        gtab = gradient_table(bvals, bvecs)
        return gtab

    def load_data(self):
        hardi_fname, _, _ = get_fnames("stanford_hardi")
        data, _ = load_nifti(hardi_fname)

        data_norm_spatial = normalize_spatial_dimensions(data)

        return data_norm_spatial


def add_rician_noise(data, sigma):
    noisy = np.zeros_like(data)
    for vol in range(data.shape[-1]):
        noise_1 = np.random.normal(0, sigma, data[..., vol].shape).astype("float32")
        noise_2 = np.random.normal(0, sigma, data[..., vol].shape).astype("float32")
        noisy[..., vol] = (data[..., vol] + noise_1) ** 2 + noise_2**2
        noisy[..., vol] = noisy[..., vol] ** 0.5
    return noisy


def normalize_spatial_dimensions(data):
    # Assuming data shape is (x, y, z, c)
    normalized_data = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[-1]):  # Iterate over each volume
        volume = data[..., i]
        min_val = np.min(volume)
        max_val = np.max(volume)

        # Normalize to [0, 1] range
        normalized_data[..., i] = (volume - min_val) / (max_val - min_val + 1e-6)

    return normalized_data


class DataSet(Dataset):
    def __init__(self, data: np.ndarray, take_volumes=8):
        self.data = torch.from_numpy(data).type(torch.float)
        self.n_vols = data.shape[-1]
        # Permute to (Z, X, Y, Bvalues)
        self.data = self.data.permute(2, 0, 1, 3)
        self.take_volumes = take_volumes

    def __getitem__(self, index):
        """returns x: volumes other than index, y: volume at index"""
        x_indices = [(index + i) % self.n_vols for i in range(1, self.take_volumes)]
        x = self.data[x_indices]
        y = self.data[index]
        return x, y

    def __len__(self):
        """number of volumes"""
        return self.n_vols
