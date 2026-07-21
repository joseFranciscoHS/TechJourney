"""Fixed DIPY CSD + peaks pipeline and no-GT proxy fixel metrics.

Corroborates how our denoising affects CSD fixel outputs (LNNN Figs. 10-11
style): fix the CSD estimator + peak-extraction protocol, vary only the input
DWI (noisy vs denoised arms exported by `paper_eval/export_denoised.py`), and
compare with reference-free proxy metrics (vs noisy and vs MP-PCA — there is
no fixel ground truth on Stanford, so all claims here must stay qualitative /
relative; see plan "Correctness contract").

Peak-extraction convention (DIPY 1.12): unused peak slots are zero-filled —
`peak_values[..., k] == 0` and `peak_dirs[..., k] == [0, 0, 0]` — so "number
of peaks" per voxel is `count(peak_values > 0)` along the last axis.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.direction.peaks import PeaksAndMetrics
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst

# Peak protocol frozen to match LNNN (Aguayo-Gonzalez et al., Front.
# Neuroinform. 2024): max 3 peaks, relative threshold 0.2.
DEFAULT_SH_ORDER = 8
DEFAULT_NPEAKS = 3
DEFAULT_REL_PEAK_THRESHOLD = 0.2
DEFAULT_MIN_SEPARATION_ANGLE = 25.0
PEAK_VALUE_EPS = 1e-8


def fit_csd_peaks(
    data_4d: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sh_order: int = DEFAULT_SH_ORDER,
    npeaks: int = DEFAULT_NPEAKS,
    rel_thr: float = DEFAULT_REL_PEAK_THRESHOLD,
    min_sep_angle: float = DEFAULT_MIN_SEPARATION_ANGLE,
    roi_radii: int = 10,
    fa_thr: float = 0.7,
    parallel: bool = False,
) -> Tuple[PeaksAndMetrics, np.ndarray]:
    """Fit auto_response_ssst -> CSD -> peaks_from_model with a frozen protocol.

    Args:
        data_4d: (X, Y, Z, V) physical-intensity DWI volume (b0s + DWIs), as
            produced by `export_denoised.assemble_denorm_4d` + `save_arm`.
        bvals, bvecs: length-V gradient table, same volume order as data_4d.
        mask: optional (X, Y, Z) bool brain/ROI mask to restrict fitting.
        sh_order, npeaks, rel_thr, min_sep_angle: peak-extraction protocol,
            frozen across all arms so only the input DWI varies.

    Returns:
        (peaks, response) — `peaks` is a DIPY `PeaksAndMetrics` with
        `.peak_dirs` (X,Y,Z,npeaks,3), `.peak_values` (X,Y,Z,npeaks),
        `.peak_indices` (X,Y,Z,npeaks), `.shm_coeff` (X,Y,Z,nsh), `.gfa`
        (X,Y,Z); `response` is the (evals, S0) response function tuple.
    """
    if data_4d.ndim != 4:
        raise ValueError(f"fit_csd_peaks expects 4D data, got ndim={data_4d.ndim}")
    gtab = gradient_table(bvals, bvecs=bvecs)
    response, ratio = auto_response_ssst(
        gtab, data_4d, roi_radii=roi_radii, fa_thr=fa_thr
    )
    logging.info("CSD response: evals=%s, S0=%.4f, ratio=%.4f", response[0], response[1], ratio)
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=sh_order)
    peaks = peaks_from_model(
        model,
        data_4d,
        default_sphere,
        relative_peak_threshold=rel_thr,
        min_separation_angle=min_sep_angle,
        mask=mask,
        npeaks=npeaks,
        return_sh=True,
        sh_order_max=sh_order,
        parallel=parallel,
    )
    return peaks, response


def save_csd_outputs(out_dir: str, peaks: PeaksAndMetrics) -> Dict[str, str]:
    """Save peak arrays + fODF SH coefficients for one arm under `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "peaks_dirs": os.path.join(out_dir, "peaks_dirs.npy"),
        "peaks_values": os.path.join(out_dir, "peaks_values.npy"),
        "peaks_indices": os.path.join(out_dir, "peaks_indices.npy"),
        "fodf_sh": os.path.join(out_dir, "fodf_sh.npy"),
    }
    np.save(paths["peaks_dirs"], peaks.peak_dirs)
    np.save(paths["peaks_values"], peaks.peak_values)
    np.save(paths["peaks_indices"], peaks.peak_indices)
    np.save(paths["fodf_sh"], peaks.shm_coeff)
    logging.info("Saved CSD outputs to %s", out_dir)
    return paths


def load_csd_outputs(out_dir: str) -> Dict[str, np.ndarray]:
    """Load back arrays written by `save_csd_outputs` (e.g. for proxy metrics)."""
    return {
        "peak_dirs": np.load(os.path.join(out_dir, "peaks_dirs.npy")),
        "peak_values": np.load(os.path.join(out_dir, "peaks_values.npy")),
        "peak_indices": np.load(os.path.join(out_dir, "peaks_indices.npy")),
        "shm_coeff": np.load(os.path.join(out_dir, "fodf_sh.npy")),
    }


def _n_peaks_per_voxel(peak_values: np.ndarray) -> np.ndarray:
    """(X,Y,Z) int array: count of populated peak slots per voxel."""
    return np.sum(peak_values > PEAK_VALUE_EPS, axis=-1)


def compute_csd_fixel_metrics(
    peaks: PeaksAndMetrics,
    mask: Optional[np.ndarray] = None,
    *,
    protocol: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-arm CSD/fixel summary (no comparison to other arms — that's proxy metrics)."""
    n_peaks = _n_peaks_per_voxel(peaks.peak_values)
    m = mask if mask is not None else np.ones(n_peaks.shape, dtype=bool)
    n_vox = int(np.sum(m))
    out: Dict[str, Any] = {
        "n_voxels": n_vox,
        "mean_n_peaks": float(np.mean(n_peaks[m])) if n_vox else None,
        "frac_zero_peak": float(np.mean(n_peaks[m] == 0)) if n_vox else None,
        "frac_single_peak": float(np.mean(n_peaks[m] == 1)) if n_vox else None,
        "frac_multi_peak": float(np.mean(n_peaks[m] >= 2)) if n_vox else None,
        "gfa_mean": float(np.mean(np.asarray(peaks.gfa)[m])) if n_vox else None,
    }
    if protocol:
        out["protocol"] = protocol
    return out


def save_csd_fixel_metrics(
    metrics: Dict[str, Any], out_dir: str, filename: str = "csd_fixel_metrics.json"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Saved CSD fixel metrics: %s", path)
    return path


# ---------------------------------------------------------------------------
# Proxy metrics (no GT): compare two arms' peaks under the identical protocol.
# ---------------------------------------------------------------------------


def primary_peak_angular_deviation(
    peak_dirs_a: np.ndarray,
    peak_dirs_b: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Angular deviation (degrees) between the primary peak of two arms.

    Peak directions are undirected axes (not vectors), so we compare via
    `abs(dot(a, b))` to remove the sign ambiguity before taking arccos.
    Voxels where either arm has no primary peak (zero vector) are excluded.
    """
    a = peak_dirs_a[..., 0, :]
    b = peak_dirs_b[..., 0, :]
    valid = (np.linalg.norm(a, axis=-1) > PEAK_VALUE_EPS) & (
        np.linalg.norm(b, axis=-1) > PEAK_VALUE_EPS
    )
    if mask is not None:
        valid = valid & mask
    n_vox = int(np.sum(valid))
    if n_vox == 0:
        return {"mean_deg": None, "median_deg": None, "n_voxels": 0}
    dot = np.clip(np.abs(np.sum(a[valid] * b[valid], axis=-1)), 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(dot))
    return {
        "mean_deg": float(np.mean(angles_deg)),
        "median_deg": float(np.median(angles_deg)),
        "std_deg": float(np.std(angles_deg)),
        "n_voxels": n_vox,
    }


def multi_peak_stats(
    peak_values: np.ndarray, mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Mean #peaks and multi-peak voxel fraction (self-contained per-arm proxy)."""
    n_peaks = _n_peaks_per_voxel(peak_values)
    m = mask if mask is not None else np.ones(n_peaks.shape, dtype=bool)
    n_vox = int(np.sum(m))
    if n_vox == 0:
        return {"mean_n_peaks": None, "frac_multi_peak": None, "n_voxels": 0}
    return {
        "mean_n_peaks": float(np.mean(n_peaks[m])),
        "frac_multi_peak": float(np.mean(n_peaks[m] >= 2)),
        "n_voxels": n_vox,
    }


def discontinuity_rate(
    peak_dirs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    angle_thresh_deg: float = 45.0,
) -> Dict[str, Any]:
    """Fraction of voxels whose primary peak flips >angle_thresh_deg from a
    forward (+x/+y/+z) neighbor.

    PROXY / APPROXIMATION: this is a coarse discontinuity indicator, not a
    validated tractography-continuity metric — it only checks axis-aligned
    forward neighbors within `mask` (typically a crossing-fiber ROI) and does
    not account for fiber curvature. Document any claims accordingly.
    """
    primary = peak_dirs[..., 0, :]
    has_peak = np.linalg.norm(primary, axis=-1) > PEAK_VALUE_EPS
    m = mask if mask is not None else np.ones(has_peak.shape, dtype=bool)
    m = m & has_peak

    flips = np.zeros(has_peak.shape, dtype=bool)
    checked = np.zeros(has_peak.shape, dtype=bool)
    for axis in range(3):
        shifted = np.roll(primary, -1, axis=axis)
        shifted_has = np.roll(has_peak, -1, axis=axis)
        pair_valid = m & shifted_has
        # Zero out wrap-around neighbor at the axis boundary.
        idx = [slice(None)] * 3
        idx[axis] = -1
        pair_valid[tuple(idx)] = False
        dot = np.clip(np.abs(np.sum(primary * shifted, axis=-1)), 0.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        flips |= pair_valid & (angle_deg > angle_thresh_deg)
        checked |= pair_valid

    n_checked = int(np.sum(m))
    if n_checked == 0:
        return {"discontinuity_rate": None, "n_voxels": 0, "angle_thresh_deg": angle_thresh_deg}
    return {
        "discontinuity_rate": float(np.sum(flips[m]) / n_checked),
        "n_voxels": n_checked,
        "angle_thresh_deg": angle_thresh_deg,
    }


def gyral_fan_spread(
    peak_dirs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reference_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> Dict[str, Any]:
    """Spread of the primary peak orientation relative to `reference_axis`.

    PROXY / APPROXIMATION: in the absence of a true cortical-surface normal
    per voxel, this uses a single fixed reference axis (default: image
    z-axis) as a stand-in for "cortical normal" (documented assumption from
    the plan). `reference_axis` should be overridden per-ROI once the gyral
    blade ROI is scouted (Phase 5) with its locally estimated normal.
    """
    ref = np.asarray(reference_axis, dtype=np.float64)
    ref = ref / (np.linalg.norm(ref) + PEAK_VALUE_EPS)
    primary = peak_dirs[..., 0, :]
    has_peak = np.linalg.norm(primary, axis=-1) > PEAK_VALUE_EPS
    m = mask if mask is not None else np.ones(has_peak.shape, dtype=bool)
    m = m & has_peak
    n_vox = int(np.sum(m))
    if n_vox == 0:
        return {"mean_angle_deg": None, "std_angle_deg": None, "n_voxels": 0}
    dot = np.clip(np.abs(np.sum(primary[m] * ref, axis=-1)), 0.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))
    return {
        "mean_angle_deg": float(np.mean(angle_deg)),
        "std_angle_deg": float(np.std(angle_deg)),
        "n_voxels": n_vox,
    }


def compute_proxy_metrics(
    peaks_by_arm: Dict[str, Dict[str, np.ndarray]],
    mask: Optional[np.ndarray] = None,
    reference_arms: Sequence[str] = ("noisy", "mppca"),
    discontinuity_mask: Optional[np.ndarray] = None,
    angle_thresh_deg: float = 45.0,
    reference_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> Dict[str, Any]:
    """Build the full cross-arm proxy metrics dict.

    Args:
        peaks_by_arm: `{arm: load_csd_outputs(...)}` — arm name is the same
            `job_id`-derived label used by `export_denoised.save_arm`.
        mask: brain/tissue mask shared by all arms (same crop -> same mask).
        reference_arms: arms every other arm is compared against (skipped
            for themselves and for any reference arm missing from
            `peaks_by_arm`).
        discontinuity_mask: crossing-fiber ROI mask for `discontinuity_rate`
            (falls back to `mask` if not given).
        reference_axis: see `gyral_fan_spread` — override once ROI-specific
            normals are available (Phase 5).

    Returns:
        `{arm: {self: {...}, vs_<ref>: {...}, ...}}`.
    """
    disc_mask = discontinuity_mask if discontinuity_mask is not None else mask
    out: Dict[str, Any] = {}
    for arm, peaks in peaks_by_arm.items():
        arm_metrics: Dict[str, Any] = {
            "self": {
                **multi_peak_stats(peaks["peak_values"], mask=mask),
                "gyral_fan_spread": gyral_fan_spread(
                    peaks["peak_dirs"], mask=mask, reference_axis=reference_axis
                ),
                "discontinuity": discontinuity_rate(
                    peaks["peak_dirs"], mask=disc_mask, angle_thresh_deg=angle_thresh_deg
                ),
            }
        }
        for ref in reference_arms:
            if ref == arm or ref not in peaks_by_arm:
                continue
            arm_metrics[f"vs_{ref}"] = {
                "primary_peak_angular_deviation": primary_peak_angular_deviation(
                    peaks["peak_dirs"], peaks_by_arm[ref]["peak_dirs"], mask=mask
                )
            }
        out[arm] = arm_metrics
    return out


def save_proxy_metrics(
    metrics: Dict[str, Any], out_dir: str, basename: str = "proxy_metrics"
) -> Dict[str, str]:
    """Save proxy metrics as both JSON (nested) and CSV (flattened, one row/arm)."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rows = []
    for arm, arm_metrics in metrics.items():
        row: Dict[str, Any] = {"arm": arm}
        for group, group_metrics in arm_metrics.items():
            for k, v in _flatten_metric_group(group_metrics):
                row[f"{group}.{k}"] = v
        rows.append(row)
    csv_path = os.path.join(out_dir, f"{basename}.csv")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Saved proxy metrics: %s, %s", json_path, csv_path)
    return {"json": json_path, "csv": csv_path}


def _flatten_metric_group(d: Dict[str, Any], prefix: str = ""):
    """Yield (dotted_key, value) pairs for a (possibly nested) metrics dict."""
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten_metric_group(v, prefix=f"{key}.")
        else:
            yield key, v


def load_arm_volume(arrays_root: str, arm: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load `arrays/<arm>/denoised_<arm>.npy` + bvals/bvecs."""
    arm_dir = os.path.join(arrays_root, arm)
    npy = os.path.join(arm_dir, f"denoised_{arm}.npy")
    if not os.path.isfile(npy):
        raise FileNotFoundError(f"Missing denoised volume for arm '{arm}': {npy}")
    data = np.load(npy)
    bvals = np.loadtxt(os.path.join(arm_dir, "bvals")).ravel()
    bvecs = np.loadtxt(os.path.join(arm_dir, "bvecs")).T  # file is 3xV
    return data, bvals, bvecs


def run_csd_for_arm(
    arrays_root: str,
    csd_root: str,
    arm: str,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit CSD+peaks for one arm and write `csd/<arm>/` outputs."""
    data, bvals, bvecs = load_arm_volume(arrays_root, arm)
    if mask is None:
        # Simple brain-ish mask from mean b0 signal (first volume with b~0).
        b0_idx = np.where(bvals < 50)[0]
        ref = data[..., int(b0_idx[0])] if len(b0_idx) else data.mean(axis=-1)
        mask = ref > (0.05 * float(np.nanmax(ref)))
    peaks, response = fit_csd_peaks(data, bvals, bvecs, mask=mask)
    out_dir = os.path.join(csd_root, arm)
    save_csd_outputs(out_dir, peaks)
    protocol = {
        "sh_order": DEFAULT_SH_ORDER,
        "npeaks": DEFAULT_NPEAKS,
        "relative_peak_threshold": DEFAULT_REL_PEAK_THRESHOLD,
        "min_separation_angle": DEFAULT_MIN_SEPARATION_ANGLE,
        "response_evals": [float(x) for x in np.asarray(response[0]).ravel()],
        "response_S0": float(response[1]),
    }
    metrics = compute_csd_fixel_metrics(peaks, mask=mask, protocol=protocol)
    save_csd_fixel_metrics(metrics, out_dir)
    # Persist mask + GFA for figure backgrounds.
    np.save(os.path.join(out_dir, "mask.npy"), mask.astype(np.uint8))
    np.save(os.path.join(out_dir, "gfa.npy"), np.asarray(peaks.gfa))
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: run CSD on exported arms and (optionally) compute proxy metrics."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DIPY CSD+peaks on exported Stanford fixel arms"
    )
    parser.add_argument(
        "--study-root",
        default=None,
        help="tmp/paper_final_k16_stanford_fixels (default: inferred from repo)",
    )
    parser.add_argument(
        "--arms",
        default="noisy,mppca,p2s,drcnet3d,restormer3d,restormer3d_large,res_cnn_2d,restormer2d",
        help="Comma-separated arm labels matching arrays/<arm>/",
    )
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="Skip cross-arm proxy metrics (per-arm CSD only)",
    )
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    study_root = args.study_root or os.path.join(
        repo_root, "tmp", "paper_final_k16_stanford_fixels"
    )
    arrays_root = os.path.join(study_root, "arrays")
    csd_root = os.path.join(study_root, "csd")
    proxy_root = os.path.join(study_root, "proxy")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    available = [a for a in arms if os.path.isfile(os.path.join(arrays_root, a, f"denoised_{a}.npy"))]
    missing = [a for a in arms if a not in available]
    if missing:
        logging.warning("Skipping arms with no exported volume: %s", missing)
    if not available:
        raise SystemExit(f"No arms found under {arrays_root}")

    # Shared mask from the noisy arm when available.
    shared_mask = None
    if "noisy" in available:
        noisy_data, noisy_bvals, _ = load_arm_volume(arrays_root, "noisy")
        b0_idx = np.where(noisy_bvals < 50)[0]
        ref = (
            noisy_data[..., int(b0_idx[0])]
            if len(b0_idx)
            else noisy_data.mean(axis=-1)
        )
        shared_mask = ref > (0.05 * float(np.nanmax(ref)))

    for arm in available:
        logging.info("=== CSD arm=%s ===", arm)
        run_csd_for_arm(arrays_root, csd_root, arm, mask=shared_mask)

    if not args.skip_proxy:
        peaks_by_arm = {
            arm: load_csd_outputs(os.path.join(csd_root, arm)) for arm in available
        }
        proxy = compute_proxy_metrics(peaks_by_arm, mask=shared_mask)
        save_proxy_metrics(proxy, proxy_root)
        logging.info("Proxy metrics written under %s", proxy_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
