"""
2D axial-slice training dataset for hybrid RGS / sequential-K (paper ablation 1.4).

Each sample is a patch (K, H, W) at one z-slice; same Bernoulli mask on ``target_channel``
as the 3D ``TrainingDataSet``. Intended for :class:`DenoiserNet2D` + ``fit_model``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch


class TrainingDataSet2D(torch.utils.data.Dataset):
    """
    Args:
        data_xyzv: (X, Y, Z, V) noisy DWI + b0s (same layout as 3D training).
        shell_sampling_mode: ``"rgs"`` | ``"sequential"``.
        num_input_volumes: K (must be <= V).
        target_channel: channel with spatial mask + supervision (typically K-1).
        patch_hw: (H, W) spatial size; if larger than X/Y, clipped to full FOV.
        step: stride for sliding (x, y) patch origins (full-slice when step >= max(X,Y)).
    """

    def __init__(
        self,
        data_xyzv: np.ndarray,
        *,
        shell_sampling_mode: str = "rgs",
        num_input_volumes: int = 24,
        target_channel: int = 23,
        mask_p: float = 0.3,
        patch_hw: Tuple[int, int] = (32, 32),
        step: int = 16,
        sample_rng_seed: Optional[int] = None,
    ):
        self.shell_sampling_mode = shell_sampling_mode
        self.num_input_volumes = int(num_input_volumes)
        self.target_channel = int(target_channel)
        self.mask_p = float(mask_p)
        self._rng = (
            np.random.default_rng(sample_rng_seed)
            if sample_rng_seed is not None
            else np.random.default_rng()
        )

        x_max, y_max, self.n_z, self.n_vols = data_xyzv.shape[:4]
        self.data_vxyz = np.transpose(data_xyzv, (3, 0, 1, 2)).astype(np.float32)

        ph, pw = int(patch_hw[0]), int(patch_hw[1])
        self.patch_h = min(ph, x_max)
        self.patch_w = min(pw, y_max)
        self._coords: List[Tuple[int, int]] = []
        for x0 in range(0, max(1, x_max - self.patch_h + 1), step):
            for y0 in range(0, max(1, y_max - self.patch_w + 1), step):
                self._coords.append((x0, y0))
        if not self._coords:
            self._coords = [(0, 0)]
            self.patch_h = x_max
            self.patch_w = y_max

        if self.num_input_volumes > self.n_vols:
            raise ValueError(
                f"K={self.num_input_volumes} exceeds V={self.n_vols} in TrainingDataSet2D"
            )
        if not (0 <= self.target_channel < self.num_input_volumes):
            raise ValueError("target_channel out of range for K")

        self.n_windows = max(0, self.n_vols - self.num_input_volumes + 1)
        if shell_sampling_mode == "rgs":
            self.total_samples = len(self._coords) * self.n_z
        else:
            if self.n_windows <= 0:
                raise ValueError("sequential 2D requires K <= V")
            self.total_samples = len(self._coords) * self.n_z * self.n_windows

        logging.info(
            "TrainingDataSet2D: mode=%s K=%s z=%s patches=%s total_samples=%s",
            shell_sampling_mode,
            self.num_input_volumes,
            self.n_z,
            len(self._coords),
            self.total_samples,
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index: int):
        if self.shell_sampling_mode == "rgs":
            coord_i = index // self.n_z
            z_idx = index % self.n_z
            x0, y0 = self._coords[coord_i]
            k = self.num_input_volumes
            indices = self._rng.choice(self.n_vols, size=k, replace=False)
        else:
            z_idx = index % self.n_z
            rest = index // self.n_z
            gw = rest % self.n_windows
            coord_i = rest // self.n_windows
            x0, y0 = self._coords[coord_i]
            indices = np.arange(gw, gw + self.num_input_volumes, dtype=np.int64)

        ph, pw = self.patch_h, self.patch_w
        plane = self.data_vxyz[indices, x0 : x0 + ph, y0 : y0 + pw, z_idx].copy()
        window = torch.from_numpy(plane).float()

        p_mtx = self._rng.random(size=(ph, pw))
        mask = (p_mtx > self.mask_p).astype(np.float64)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        x_masked = window.clone()
        tc = self.target_channel
        x_masked[tc] = x_masked[tc] * mask.squeeze(0)
        noisy_target_volume = window[tc : tc + 1]
        return x_masked, mask, noisy_target_volume
