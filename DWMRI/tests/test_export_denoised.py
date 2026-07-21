"""Unit tests for the CSD fixel export correctness contract (rules 1, 3, 4, 6)."""

import numpy as np
import pytest

from paper_eval.export_denoised import assemble_denorm_4d, save_arm
from utils.data import (
    invert_normalization,
    normalize_spatial_dimensions_with_params,
)


def test_normalize_invert_roundtrip():
    rng = np.random.default_rng(0)
    raw = rng.uniform(10.0, 500.0, size=(8, 8, 8, 5)).astype(np.float32)
    normed, params = normalize_spatial_dimensions_with_params(raw)
    recovered = invert_normalization(normed, params)
    assert recovered.shape == raw.shape
    np.testing.assert_allclose(recovered, raw, rtol=1e-5, atol=1e-4)


def test_assemble_denorm_4d_shapes_and_order():
    rng = np.random.default_rng(1)
    nb0, n_dwi = 2, 4
    take = nb0 + n_dwi
    original = rng.random((6, 7, 8, take)).astype(np.float32)
    recon_native = rng.random((6, 7, 8, n_dwi)).astype(np.float32)
    # Fake per-volume params: each channel scaled differently.
    params = [(float(i), float(i + 10)) for i in range(take)]
    vol = assemble_denorm_4d(recon_native, original, nb0, take, params)
    assert vol.shape == (6, 7, 8, take)
    # b0 channels come from inverted original; DWI from inverted recon.
    b0_expected = invert_normalization(original[..., :nb0], params[:nb0])
    dwi_expected = invert_normalization(recon_native, params[nb0:])
    np.testing.assert_allclose(vol[..., :nb0], b0_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vol[..., nb0:], dwi_expected, rtol=1e-5, atol=1e-5)


def test_save_arm_asserts_bvals_and_writes(tmp_path):
    rng = np.random.default_rng(2)
    nb0, n_dwi = 2, 3
    take = nb0 + n_dwi
    vol = rng.random((4, 5, 6, take)).astype(np.float32)
    affine = np.eye(4)
    bvals = np.array([0.0, 0.0, 1000.0, 1000.0, 1000.0])
    bvecs = np.zeros((take, 3), dtype=np.float64)
    bvecs[nb0:, 0] = 1.0
    paths = save_arm(str(tmp_path / "arm"), "toy", vol, affine, bvals, bvecs, nb0)
    assert (tmp_path / "arm" / "denoised_toy.npy").is_file()
    assert (tmp_path / "arm" / "denoised_toy.nii.gz").is_file()
    loaded = np.load(paths["npy"])
    assert loaded.shape == vol.shape

    # b0 assert: non-b0 in front must fail.
    with pytest.raises(AssertionError):
        save_arm(
            str(tmp_path / "bad"),
            "bad",
            vol,
            affine,
            np.array([1000.0, 0.0, 1000.0, 1000.0, 1000.0]),
            bvecs,
            nb0,
        )


def test_primary_peak_angular_deviation_identity():
    from paper_eval.csd_fixels import primary_peak_angular_deviation

    dirs = np.zeros((3, 3, 3, 2, 3), dtype=np.float64)
    dirs[..., 0, :] = np.array([1.0, 0.0, 0.0])
    out = primary_peak_angular_deviation(dirs, dirs)
    assert out["n_voxels"] == 27
    assert out["mean_deg"] == pytest.approx(0.0, abs=1e-6)
