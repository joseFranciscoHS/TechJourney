"""Unit tests for utils.training_patch_subset."""

import numpy as np
import pytest
from torch.utils.data import Dataset, Subset

from utils.training_patch_subset import (
    apply_training_patch_subset,
    compute_patch_subset_num_samples,
    patch_subset_indices,
    training_subset_checkpoint_segment,
)


class _TinyDs(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


class _MunchTrain:
    def __init__(self, *, subset_fraction=1.0, subset_max_patches=None, seed=0):
        self.subset_fraction = subset_fraction
        self.subset_max_patches = subset_max_patches
        self.seed = seed


def test_checkpoint_segment_empty_when_full():
    assert training_subset_checkpoint_segment(_MunchTrain()) == ""


def test_checkpoint_segment_fraction_and_max():
    s = training_subset_checkpoint_segment(
        _MunchTrain(subset_fraction=0.1, subset_max_patches=5000)
    )
    assert "_subset_" in s
    assert "f0p1" in s or "f0" in s
    assert "max5000" in s


def test_compute_num_samples_full():
    assert compute_patch_subset_num_samples(100, fraction=1.0, max_patches=None) == 100


def test_compute_num_samples_fraction():
    assert compute_patch_subset_num_samples(100, fraction=0.1, max_patches=None) == 10


def test_compute_num_samples_cap():
    assert compute_patch_subset_num_samples(10000, fraction=1.0, max_patches=50) == 50


def test_compute_num_samples_tiny_n():
    assert compute_patch_subset_num_samples(3, fraction=0.01, max_patches=None) == 1


def test_compute_invalid_fraction():
    with pytest.raises(ValueError):
        compute_patch_subset_num_samples(10, fraction=0.0, max_patches=None)
    with pytest.raises(ValueError):
        compute_patch_subset_num_samples(10, fraction=1.5, max_patches=None)


def test_patch_subset_indices_deterministic():
    a = patch_subset_indices(50, 10, seed=123)
    b = patch_subset_indices(50, 10, seed=123)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 10
    assert len(set(a.tolist())) == 10


def test_apply_training_patch_subset_no_wrap_when_full():
    ds = _TinyDs(20)
    out, n_tot, n_used = apply_training_patch_subset(
        ds, fraction=1.0, max_patches=None, seed=1, log=False
    )
    assert out is ds
    assert n_tot == n_used == 20


def test_apply_training_patch_subset_wraps():
    ds = _TinyDs(100)
    out, n_tot, n_used = apply_training_patch_subset(
        ds, fraction=0.2, max_patches=None, seed=7, log=False
    )
    assert isinstance(out, Subset)
    assert n_tot == 100
    assert n_used == 20
    assert len(out) == 20
