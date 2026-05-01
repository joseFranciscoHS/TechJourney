import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.data import rescale_reconstruction_to_01
from utils.experiment_runtime import hardware_info


def apply_reconstruction_eval_protocol(
    reconstructed_xyzv: np.ndarray,
    reference_xyzv: np.ndarray,
    *,
    rescale_to_01: bool,
    rescale_mode: str,
    clip_to_range: bool,
) -> np.ndarray:
    """Apply a single canonical post-processing policy before metrics."""
    out = reconstructed_xyzv
    if rescale_to_01:
        ref = reference_xyzv if rescale_mode == "match_gt" else None
        out = rescale_reconstruction_to_01(out, mode=rescale_mode, reference=ref)
    if clip_to_range:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def compute_roi_mask(
    reference_xyzv: np.ndarray, threshold: Optional[float]
) -> Optional[np.ndarray]:
    if threshold is None:
        return None
    return (reference_xyzv > float(threshold)).any(axis=-1)


def metrics_policy_dict(
    *,
    reference_name: str,
    rescale_to_01: bool,
    rescale_mode: str,
    clip_to_range: bool,
    roi_threshold: Optional[float],
) -> Dict[str, Any]:
    return {
        "reference_name": reference_name,
        "rescale_to_01": bool(rescale_to_01),
        "rescale_mode": str(rescale_mode),
        "clip_to_range": bool(clip_to_range),
        "metrics_roi_threshold": roi_threshold,
    }


def save_run_manifest(
    *,
    out_dir: str,
    filename: str = "run_manifest.json",
    seed: Optional[int],
    reproducible: Optional[bool],
    runtime_device: str,
    config: Dict[str, Any],
    metrics_policy: Dict[str, Any],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "seed": seed,
        "reproducible": reproducible,
        "runtime_device": runtime_device,
        "hardware": hardware_info(runtime_device),
        "config": config,
        "metrics_policy": metrics_policy,
    }
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Saved run manifest: %s", path)
    return path


def summarize_roi(mask: Optional[np.ndarray]) -> Tuple[int, float]:
    if mask is None:
        return 0, 0.0
    n_roi = int(np.sum(mask))
    pct = float(100.0 * n_roi / mask.size) if mask.size else 0.0
    return n_roi, pct
