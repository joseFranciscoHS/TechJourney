"""Tests for DRCNet objective-controlled ablation modes."""

import numpy as np
import torch
from torch import nn

from drcnet_hybrid_rgs.data import TrainingDataSet
from drcnet_hybrid_rgs.fit import _masked_mse
from drcnet_hybrid_rgs.reconstruction import reconstruct_dwis_rgs


def _constant_volume_data(n_vols=5, spatial=4):
    data = np.zeros((spatial, spatial, spatial, n_vols), dtype=np.float32)
    for vol_idx in range(n_vols):
        data[..., vol_idx] = float(vol_idx + 1)
    return data


def test_training_dataset_hybrid_keeps_k_channel_stack():
    data = _constant_volume_data()
    dataset = TrainingDataSet(
        data=data,
        patch_size=(3, 2, 2, 2),
        step=2,
        mask_p=0.5,
        shell_sampling_mode="rgs",
        num_input_volumes=3,
        target_channel=2,
        objective_mode="hybrid",
        sample_rng_seed=123,
    )

    x, mask, target, orientation_info = dataset[0]

    assert x.shape == (3, 2, 2, 2)
    assert mask.shape == (1, 2, 2, 2)
    assert target.shape == (1, 2, 2, 2)
    assert orientation_info.numel() == 0


def test_training_dataset_angular_excludes_noisy_target_from_input():
    data = _constant_volume_data()
    dataset = TrainingDataSet(
        data=data,
        patch_size=(3, 2, 2, 2),
        step=2,
        mask_p=0.5,
        shell_sampling_mode="rgs",
        num_input_volumes=3,
        target_channel=0,
        objective_mode="angular",
        sample_rng_seed=123,
    )

    x, mask, target, _orientation_info = dataset[0]
    target_value = float(target.flatten()[0].item())

    assert x.shape == (3, 2, 2, 2)
    assert target.shape == (1, 2, 2, 2)
    assert torch.all(mask == 1)
    assert target_value not in set(x.unique().tolist())


def test_training_dataset_spatial_masks_single_target_channel():
    data = _constant_volume_data()
    dataset = TrainingDataSet(
        data=data,
        patch_size=(1, 2, 2, 2),
        step=2,
        mask_p=0.5,
        shell_sampling_mode="rgs",
        num_input_volumes=1,
        target_channel=0,
        objective_mode="spatial",
        sample_rng_seed=123,
    )

    x, mask, target, _orientation_info = dataset[0]

    assert x.shape == (1, 2, 2, 2)
    assert target.shape == (1, 2, 2, 2)
    torch.testing.assert_close(x, target * mask)


def test_masked_mse_matches_mds2s_formula():
    pred = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])

    loss = _masked_mse(pred, target, mask)

    assert loss.item() == (2.0**2 + 3.0**2) / 2.0


class _CountingFirstChannelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.examples = 0

    def forward(self, inputs, orientation_info=None):
        del orientation_info
        self.calls += 1
        self.examples += int(inputs.shape[0])
        return inputs[:, :1]


def test_spatial_rgs_reconstruction_averages_npred_masked_passes():
    data = torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2)
    model = _CountingFirstChannelModel()

    recon = reconstruct_dwis_rgs(
        model=model,
        data=data,
        device="cpu",
        mask_p=0.0,
        n_preds=5,
        n_context=7,
        num_input=1,
        target_channel=0,
        pred_chunk_size=2,
        objective_mode="spatial",
    )

    np.testing.assert_allclose(recon, data.numpy())
    assert model.examples == 3 * 5
    assert model.calls == 3 * 3
