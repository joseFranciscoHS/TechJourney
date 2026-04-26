import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import dipy.reconst.dti as dti
import numpy as np
from dipy.core.gradients import gradient_table


def compute_dti_maps(data_xyzv: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray):
    gtab = gradient_table(bvals, bvecs)
    tenfit = dti.TensorModel(gtab).fit(data_xyzv)
    return {"fa": tenfit.fa, "md": tenfit.md, "ad": tenfit.ad, "rd": tenfit.rd}


def _validate_dti_inputs(
    denoised_xyzv: np.ndarray,
    gt_xyzv: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
) -> Optional[str]:
    if denoised_xyzv.ndim != 4 or gt_xyzv.ndim != 4:
        return "dti_invalid_ndim_expected_4d"
    if denoised_xyzv.shape != gt_xyzv.shape:
        return f"dti_shape_mismatch denoised={denoised_xyzv.shape} gt={gt_xyzv.shape}"
    nv = int(denoised_xyzv.shape[-1])
    if len(bvals) != nv:
        return f"dti_bvals_len_mismatch bvals={len(bvals)} V={nv}"
    if np.asarray(bvecs).shape != (nv, 3):
        return f"dti_bvecs_shape_mismatch bvecs={np.asarray(bvecs).shape} expected=({nv},3)"
    if not np.isfinite(denoised_xyzv).all() or not np.isfinite(gt_xyzv).all():
        return "dti_non_finite_input_values"
    if np.sum(np.asarray(bvals) <= 50) < 1:
        return "dti_requires_b0_volume"
    return None


def _map_sanity(dti_maps: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    fa = np.asarray(dti_maps["fa"])
    md = np.asarray(dti_maps["md"])
    out["fa_nonfinite"] = int(np.sum(~np.isfinite(fa)))
    out["md_nonfinite"] = int(np.sum(~np.isfinite(md)))
    out["fa_range"] = [float(np.nanmin(fa)), float(np.nanmax(fa))]
    out["md_range"] = [float(np.nanmin(md)), float(np.nanmax(md))]
    return out


def compute_dti_errors(
    denoised_xyzv: np.ndarray,
    gt_xyzv: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    d_dti = compute_dti_maps(denoised_xyzv, bvals, bvecs)
    g_dti = compute_dti_maps(gt_xyzv, bvals, bvecs)
    if roi_mask is None:
        roi_mask = np.ones_like(g_dti["fa"], dtype=bool)

    def mae(a, b):
        return float(np.mean(np.abs(a[roi_mask] - b[roi_mask])))

    return {
        "fa_mae": mae(d_dti["fa"], g_dti["fa"]),
        "md_mae": mae(d_dti["md"], g_dti["md"]),
        "ad_mae": mae(d_dti["ad"], g_dti["ad"]),
        "rd_mae": mae(d_dti["rd"], g_dti["rd"]),
        "dti_sanity_denoised": _map_sanity(d_dti),
        "dti_sanity_gt": _map_sanity(g_dti),
    }


def roi_mask_from_gt_threshold(
    gt_xyzv: np.ndarray, roi_threshold: Optional[float]
) -> Optional[np.ndarray]:
    """ROI = voxels where any channel exceeds threshold (same rule as hybrid reconstruct)."""
    if roi_threshold is None:
        return None
    return (gt_xyzv > float(roi_threshold)).any(axis=-1)


def bvals_bvecs_truncated_from_dbrain_matrix(
    bvecs_path: str, bvalue: float, n_volumes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load HCP-style b-matrix file and return bvals/bvecs truncated to n_volumes (like hybrid)."""
    from utils.data import DBrainDataLoader

    dl = DBrainDataLoader(
        nii_path="__unused__", bvecs_path=bvecs_path, bvalue=float(bvalue)
    )
    gtab = dl.load_gradient_table()
    bvals = np.asarray(gtab.bvals, dtype=np.float64)[: int(n_volumes)]
    bvecs = np.asarray(gtab.bvecs, dtype=np.float64)[: int(n_volumes)]
    if len(bvals) != int(n_volumes):
        logging.warning(
            "Gradient table length %s != n_volumes %s (truncated)",
            len(gtab.bvals),
            n_volumes,
        )
    return bvals, bvecs


def try_compute_dti_errors(
    denoised_xyzv: np.ndarray,
    gt_xyzv: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    *,
    roi_threshold: Optional[float] = 0.02,
) -> Dict[str, Any]:
    """
    Compute FA/MD/AD/RD MAE vs GT on ROI; on failure return null metrics + reason.
    """
    base_null: Dict[str, Any] = {
        "fa_mae": None,
        "md_mae": None,
        "ad_mae": None,
        "rd_mae": None,
    }
    try:
        nv = int(denoised_xyzv.shape[-1])
        bvals = np.asarray(bvals)[:nv]
        bvecs = np.asarray(bvecs)[:nv]
        reason = _validate_dti_inputs(denoised_xyzv, gt_xyzv, bvals, bvecs)
        if reason is not None:
            base_null["dti_skipped_reason"] = reason
            return base_null
        roi = roi_mask_from_gt_threshold(gt_xyzv, roi_threshold)
        out = compute_dti_errors(
            denoised_xyzv.astype(np.float64),
            gt_xyzv.astype(np.float64),
            bvals,
            bvecs,
            roi_mask=roi,
        )
        out["dti_reference"] = "clean_gt"
        return out
    except Exception as exc:
        logging.warning("DTI metrics failed: %s", exc)
        base_null["dti_skipped_reason"] = str(exc)
        return base_null


def save_dti_metrics(metrics: Dict[str, Any], out_dir: str, name: str = "dti_metrics.json"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
