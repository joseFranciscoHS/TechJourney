"""Persist denoised Stanford (and D-Brain) arms to disk for the CSD fixel study.

Establishes ONE shared export path for every arm (hybrid 3D/2D, P2S, MP-PCA,
noisy) so CSD always sees the same on-disk convention: a physical-intensity
4D volume `[b0s, denoised DWIs]` with matching bvals/bvecs and an exact affine.

Correctness contract this module enforces (see plan "Correctness contract"):
  1. `data_loader.norm_params_` is the primary source of per-volume (min, max)
     normalization params; `recover_stanford_norm_params` is a FALLBACK ONLY,
     used when exporting arrays outside of a live run (e.g. from a saved
     .npy) where no loader instance is available.
  2. Callers MUST pass the *native* reconstruction (captured before
     `apply_reconstruction_eval_protocol` rescales/clips per volume) — that
     rescale distorts each volume's S(g)/S0 ratio and corrupts CSD peaks.
  3. The training crop `[:tx, :ty, :tz]` starts at voxel (0, 0, 0), so the
     loader's affine is exactly (not approximately) valid for the crop.
  4. bvals/bvecs must be sliced to the same `take_volumes` window and kept in
     `[b0s, DWIs]` order matching the assembled volume.
  5. All arms funnel through `assemble_denorm_4d` + `save_arm` below.
"""

import logging
import os
from typing import Optional, Sequence, Tuple

import numpy as np
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti

from utils.data import invert_normalization, normalize_spatial_dimensions_with_params

NormParams = Sequence[Tuple[float, float]]


def recover_stanford_norm_params(
    take_volumes: int,
) -> Tuple[NormParams, np.ndarray]:
    """FALLBACK ONLY (rule 1) — primary path is `data_loader.norm_params_`.

    Deterministically recomputes the per-volume (min, max) that
    `StanfordDataLoader.load_data()` used, by reloading the raw Stanford
    HARDI volume and recomputing the same min-max normalization. Only
    valid when called with the exact same `take_volumes` used at export
    time, and only needed when no live `StanfordDataLoader` is available.
    """
    hardi_fname, _, _ = get_fnames(name="stanford_hardi")
    raw, affine = load_nifti(hardi_fname)
    _, params = normalize_spatial_dimensions_with_params(raw)
    return params[:take_volumes], affine


def assemble_denorm_4d(
    reconstructed_native: np.ndarray,
    original_xyzv_b0_01: np.ndarray,
    nb0: int,
    take_volumes: int,
    norm_params: NormParams,
) -> np.ndarray:
    """Denormalize + prepend b0s to build a physical-intensity 4D volume.

    Args:
        reconstructed_native: (X, Y, Z, V_dwi) denoised DWIs, normalized
            [0,1], captured BEFORE `apply_reconstruction_eval_protocol`
            (rule 2). This is what all three runners already compute as
            `reconstructed_dwis` / `recon_xyzv` prior to the eval-protocol
            rescale.
        original_xyzv_b0_01: (X, Y, Z, take_volumes) normalized ground-truth
            / noisy crop, i.e. `original_xyzv_b0` in the runners — used only
            for its b0 channels.
        nb0: number of b0 volumes at the front.
        take_volumes: total volumes (b0s + DWIs) the crop was truncated to.
        norm_params: per-volume (min, max) for all `take_volumes` channels,
            in the same order as `original_xyzv_b0_01`.

    Returns:
        (X, Y, Z, take_volumes) float32 array: physical-intensity
        `[b0s, denoised DWIs]`.
    """
    if reconstructed_native.ndim != 4 or original_xyzv_b0_01.ndim != 4:
        raise ValueError(
            "assemble_denorm_4d expects 4D arrays, got "
            f"reconstructed_native.ndim={reconstructed_native.ndim}, "
            f"original_xyzv_b0_01.ndim={original_xyzv_b0_01.ndim}"
        )
    n_dwi = take_volumes - nb0
    if reconstructed_native.shape[-1] != n_dwi:
        raise ValueError(
            f"reconstructed_native has {reconstructed_native.shape[-1]} volumes, "
            f"expected {n_dwi} (take_volumes={take_volumes} - nb0={nb0})"
        )
    if len(norm_params) != take_volumes:
        raise ValueError(
            f"norm_params has {len(norm_params)} entries, expected take_volumes={take_volumes}"
        )

    b0_phys = invert_normalization(
        original_xyzv_b0_01[..., :nb0].astype(np.float64), norm_params[:nb0]
    )
    dwi_phys = invert_normalization(
        reconstructed_native.astype(np.float64), norm_params[nb0:take_volumes]
    )
    vol_4d = np.concatenate([b0_phys, dwi_phys], axis=-1)
    logging.info(
        "assemble_denorm_4d: shape=%s min=%.4f max=%.4f mean=%.4f",
        vol_4d.shape,
        vol_4d.min(),
        vol_4d.max(),
        vol_4d.mean(),
    )
    return vol_4d.astype(np.float32)


def save_arm(
    out_dir: str,
    arm: str,
    vol_4d: np.ndarray,
    affine: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    nb0: int,
) -> dict:
    """Save one arm's denoised 4D volume + gradient table under `out_dir`.

    Writes (rules 3-4 enforced via asserts, not comments):
        denoised_<arm>.npy   - (X,Y,Z,V) float32, physical intensity
        denoised_<arm>.nii.gz - same data with the exact crop-origin affine
        bvals                - FSL-style single row, length V
        bvecs                - FSL-style 3xV, matching volume order

    Returns dict of the written paths (for registry/manifest bookkeeping).
    """
    if vol_4d.ndim != 4:
        raise ValueError(f"save_arm expects a 4D volume, got ndim={vol_4d.ndim}")
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    assert bvals.shape[0] == vol_4d.shape[-1], (
        f"bvals length ({bvals.shape[0]}) must match assembled volume count "
        f"({vol_4d.shape[-1]})"
    )
    assert bvecs.shape == (vol_4d.shape[-1], 3), (
        f"bvecs shape {bvecs.shape} must be (V, 3) = ({vol_4d.shape[-1]}, 3)"
    )
    assert nb0 >= 1 and np.all(bvals[:nb0] < 50), (
        f"first nb0={nb0} volumes must be b0s (bval < 50), got {bvals[:nb0]}"
    )

    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"denoised_{arm}.npy")
    nii_path = os.path.join(out_dir, f"denoised_{arm}.nii.gz")
    bvals_path = os.path.join(out_dir, "bvals")
    bvecs_path = os.path.join(out_dir, "bvecs")

    np.save(npy_path, vol_4d)
    save_nifti(nii_path, vol_4d, affine)
    np.savetxt(bvals_path, bvals[None, :], fmt="%g")
    np.savetxt(bvecs_path, bvecs.T, fmt="%.6f")

    logging.info("Saved denoised arm '%s' to %s (npy + nifti + bvals/bvecs)", arm, out_dir)
    return {
        "npy": npy_path,
        "nifti": nii_path,
        "bvals": bvals_path,
        "bvecs": bvecs_path,
    }


def maybe_export_denoised(
    settings,
    dataset: str,
    arm: str,
    recon_native_xyzv: np.ndarray,
    original_xyzv_b0: np.ndarray,
    data_loader,
    take_volumes: int,
) -> Optional[dict]:
    """No-op unless `reconstruct.save_denoised_{npy,nifti}` is enabled.

    Single call site for all runners (rule 5) — consolidates norm-param
    lookup, denorm+b0 assembly, and disk I/O so each runner only needs to
    capture its native reconstruction and call this function.
    """
    rc = settings.reconstruct
    if not getattr(rc, "save_denoised_npy", False) and not getattr(
        rc, "save_denoised_nifti", False
    ):
        return None

    nb0 = int(settings.data.num_b0s)
    norm_params = getattr(data_loader, "norm_params_", None)  # primary (rule 1)
    affine = getattr(data_loader, "affine_", None)
    if norm_params is None and dataset == "stanford":
        logging.warning(
            "denoised export for arm '%s': data_loader.norm_params_ is None, "
            "falling back to recover_stanford_norm_params (should not happen "
            "after the StanfordDataLoader fix; check loader wiring)",
            arm,
        )
        norm_params, affine = recover_stanford_norm_params(take_volumes)
    if norm_params is None:
        logging.warning(
            "denoised export skipped for arm '%s': no norm params available", arm
        )
        return None
    if affine is None:
        logging.warning(
            "denoised export skipped for arm '%s': no affine available", arm
        )
        return None

    vol_4d = assemble_denorm_4d(
        recon_native_xyzv, original_xyzv_b0, nb0, take_volumes, norm_params
    )
    gtab = data_loader.load_gradient_table()
    bvals = np.asarray(gtab.bvals)[:take_volumes]
    bvecs = np.asarray(gtab.bvecs)[:take_volumes]

    # Plan layout: arrays/<arm>/ (not arrays/<dataset>/) so each arm keeps its
    # own bvals/bvecs and lines up with csd/<arm>/ by the same job_id label.
    out_dir = os.path.join(getattr(rc, "denoised_out_dir", "denoised_arms"), arm)
    return save_arm(out_dir, arm, vol_4d, affine, bvals, bvecs, nb0)
