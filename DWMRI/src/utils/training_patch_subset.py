"""Random subsample of training patches (faster epochs) for hybrid RGS runners."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset


def training_subset_checkpoint_segment(settings_train: Any) -> str:
    """Stable path segment when ``subset_fraction`` < 1 or ``subset_max_patches`` is set."""
    fraction = float(getattr(settings_train, "subset_fraction", 1.0))
    raw_max = getattr(settings_train, "subset_max_patches", None)
    max_patches: Optional[int] = None
    if raw_max is not None:
        max_patches = int(raw_max)
    parts: list[str] = []
    if fraction < 1.0 - 1e-12:
        frac_str = f"{fraction:.6f}".rstrip("0").rstrip(".")
        parts.append("f" + frac_str.replace(".", "p"))
    if max_patches is not None and max_patches > 0:
        parts.append(f"max{max_patches}")
    if not parts:
        return ""
    return "_subset_" + "_".join(parts)


def compute_patch_subset_num_samples(
    n_total: int,
    *,
    fraction: float,
    max_patches: Optional[int],
) -> int:
    if n_total < 1:
        raise ValueError(f"Dataset size must be >= 1, got {n_total}")
    if not (0.0 < fraction <= 1.0 + 1e-12):
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}")
    frac = min(float(fraction), 1.0)
    n_target = int(n_total * frac)
    n_target = max(1, min(n_total, n_target))
    if max_patches is not None and int(max_patches) > 0:
        n_target = min(n_target, int(max_patches))
    return max(1, min(n_total, n_target))


def patch_subset_indices(n_total: int, num_samples: int, seed: int) -> np.ndarray:
    if num_samples >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.choice(n_total, size=num_samples, replace=False)


def apply_training_patch_subset(
    dataset: Dataset,
    *,
    fraction: float,
    max_patches: Optional[int],
    seed: int,
    log: bool = True,
) -> Tuple[Dataset, int, int]:
    """Return ``(dataset_out, n_total, n_used)``; ``dataset_out`` is unwrapped if ``n_used == n_total``."""
    n_total = len(dataset)
    num_samples = compute_patch_subset_num_samples(
        n_total, fraction=fraction, max_patches=max_patches
    )
    if num_samples >= n_total:
        if log:
            logging.info(
                "Training patch subset: full dataset (%s patches; fraction=%s max_patches=%s)",
                n_total,
                fraction,
                max_patches,
            )
        return dataset, n_total, n_total
    indices = patch_subset_indices(n_total, num_samples, seed)
    subset: Dataset = Subset(dataset, indices)
    if log:
        logging.info(
            "Training patch subset: %s / %s patches (fraction=%s max_patches=%s)",
            num_samples,
            n_total,
            fraction,
            max_patches,
        )
    return subset, n_total, num_samples


def apply_training_patch_subset_from_train_block(
    dataset: Dataset,
    settings_train: Any,
    *,
    seed: Optional[int] = None,
    log: bool = True,
) -> Tuple[Dataset, int, int]:
    """Read ``subset_fraction``, ``subset_max_patches``, and ``seed`` from a train config node."""
    fraction = float(getattr(settings_train, "subset_fraction", 1.0))
    raw_max = getattr(settings_train, "subset_max_patches", None)
    max_patches = int(raw_max) if raw_max is not None else None
    eff_seed = int(seed if seed is not None else getattr(settings_train, "seed", 42))
    return apply_training_patch_subset(
        dataset,
        fraction=fraction,
        max_patches=max_patches,
        seed=eff_seed,
        log=log,
    )
