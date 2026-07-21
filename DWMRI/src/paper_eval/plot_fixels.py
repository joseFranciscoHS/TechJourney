"""Matplotlib glyph panels for the Stanford CSD fixel figure (arms x ROIs).

A "glyph" here is a set of short line segments per voxel, one per detected
CSD peak, oriented along the in-plane component of `peak_dirs` and colored/
scaled by `peak_values`. This mirrors the qualitative fixel-glyph panels in
LNNN Figs. 10-11 (Aguayo-Gonzalez et al., Front. Neuroinform. 2024) without
reimplementing their pipeline — see plan "Summary".

ROI convention (`stanford_fixel_rois.yaml` schema):
    name: corona_radiata_crossing
    axis: 2            # 0=x (sagittal), 1=y (coronal), 2=z (axial)
    index: 40          # slice index along `axis`
    bbox: [[x0, x1], [y0, y1]]   # bbox in the two remaining axes, ascending
                                  # order (e.g. axis=2 -> bbox is [x-range, y-range])
    description: free-text rationale from the scout pass
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

PEAK_VALUE_EPS = 1e-8


def load_rois(yaml_path: str) -> List[Dict[str, Any]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    rois = payload.get("rois", [])
    for roi in rois:
        if "axis" not in roi or "index" not in roi or "bbox" not in roi:
            raise ValueError(f"ROI entry missing axis/index/bbox: {roi}")
    return rois


def _other_axes(axis: int) -> Sequence[int]:
    return tuple(a for a in (0, 1, 2) if a != axis)


def roi_slice(vol: np.ndarray, roi: Dict[str, Any]) -> np.ndarray:
    """Extract the 2D (+ trailing dims) slice + bbox described by `roi`.

    Works for spatial-only volumes (X,Y,Z), (X,Y,Z,npeaks), and
    (X,Y,Z,npeaks,3) — any trailing dims after the first 3 spatial dims are
    preserved as-is.
    """
    axis = int(roi["axis"])
    idx = int(roi["index"])
    (b0lo, b0hi), (b1lo, b1hi) = roi["bbox"]
    other = _other_axes(axis)

    sl = [slice(None)] * vol.ndim
    sl[axis] = idx
    sl[other[0]] = slice(b0lo, b0hi)
    sl[other[1]] = slice(b1lo, b1hi)
    return vol[tuple(sl)]


def plot_glyph_panel(
    ax,
    background_2d: np.ndarray,
    peak_dirs_2d: np.ndarray,
    peak_values_2d: np.ndarray,
    axis: int,
    title: Optional[str] = None,
    max_peaks: int = 3,
    glyph_scale: float = 0.45,
    value_thr: float = PEAK_VALUE_EPS,
    cmap: str = "gray",
) -> None:
    """Draw one glyph panel: grayscale background + per-voxel peak line segments.

    `peak_dirs_2d` is (H, W, npeaks, 3); only the two in-plane components
    (i.e. the two axes orthogonal to `axis`) are used to draw each glyph.
    """
    other = _other_axes(axis)
    h, w = background_2d.shape
    ax.imshow(
        background_2d.T,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=max(float(np.nanmax(background_2d)), 1e-6),
    )
    n_peaks = min(max_peaks, peak_dirs_2d.shape[-2])
    for i in range(h):
        for j in range(w):
            for k in range(n_peaks):
                val = float(peak_values_2d[i, j, k])
                if val <= value_thr:
                    continue
                d = peak_dirs_2d[i, j, k]
                dx, dy = float(d[other[0]]), float(d[other[1]])
                norm = (dx**2 + dy**2) ** 0.5
                if norm <= value_thr:
                    continue
                dx, dy = dx / norm, dy / norm
                ax.plot(
                    [i - glyph_scale * dx, i + glyph_scale * dx],
                    [j - glyph_scale * dy, j + glyph_scale * dy],
                    color="red" if k == 0 else "cyan" if k == 1 else "yellow",
                    linewidth=0.8,
                    solid_capstyle="round",
                )
    ax.set_xlim(-0.5, h - 0.5)
    ax.set_ylim(-0.5, w - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def plot_fixel_figure(
    rois: Sequence[Dict[str, Any]],
    arms: Sequence[str],
    peaks_by_arm: Dict[str, Dict[str, np.ndarray]],
    background_by_arm: Dict[str, np.ndarray],
    out_path: str,
    max_peaks: int = 3,
    glyph_scale: float = 0.45,
    panel_size: float = 2.2,
) -> str:
    """Build the arms x ROIs glyph grid figure and save it to `out_path`.

    Args:
        rois: list of ROI dicts (see module docstring / `load_rois`).
        arms: column order, e.g. ["noisy", "mppca", "p2s", "drcnet3d", ...].
        peaks_by_arm: `{arm: {"peak_dirs": ..., "peak_values": ...}}` — full
            (X, Y, Z, ...) arrays, same crop/shape for every arm.
        background_by_arm: `{arm: gfa_or_b0_volume}` — (X, Y, Z) background
            for each arm (typically CSD GFA on that arm's own denoised input).
    """
    n_rows, n_cols = len(rois), len(arms)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        squeeze=False,
    )
    for r, roi in enumerate(rois):
        for c, arm in enumerate(arms):
            ax = axes[r][c]
            if arm not in peaks_by_arm or arm not in background_by_arm:
                ax.axis("off")
                continue
            bg_2d = roi_slice(background_by_arm[arm], roi)
            dirs_2d = roi_slice(peaks_by_arm[arm]["peak_dirs"], roi)
            vals_2d = roi_slice(peaks_by_arm[arm]["peak_values"], roi)
            title = arm if r == 0 else None
            plot_glyph_panel(
                ax,
                bg_2d,
                dirs_2d,
                vals_2d,
                axis=int(roi["axis"]),
                title=title,
                max_peaks=max_peaks,
                glyph_scale=glyph_scale,
            )
            if c == 0:
                ax.set_ylabel(roi.get("name", f"roi_{r}"), fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved fixel glyph figure: %s", out_path)
    return out_path


def main(argv=None) -> None:
    """CLI: build arms x ROIs glyph figure from study-root CSD outputs."""
    import argparse

    from paper_eval.csd_fixels import load_csd_outputs

    parser = argparse.ArgumentParser(description="Plot Stanford CSD fixel glyph figure")
    parser.add_argument("--study-root", default=None)
    parser.add_argument(
        "--rois",
        default=None,
        help="Path to stanford_fixel_rois.yaml (default: beside this module)",
    )
    parser.add_argument(
        "--arms",
        default="noisy,mppca,p2s,drcnet3d,restormer3d,restormer3d_large,res_cnn_2d,restormer2d",
    )
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    study_root = args.study_root or os.path.join(
        repo_root, "tmp", "paper_final_k16_stanford_fixels"
    )
    rois_path = args.rois or os.path.join(script_dir, "stanford_fixel_rois.yaml")
    out_path = args.out or os.path.join(study_root, "figures", "fixel_glyphs.png")

    rois = load_rois(rois_path)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    csd_root = os.path.join(study_root, "csd")

    peaks_by_arm = {}
    background_by_arm = {}
    for arm in arms:
        arm_dir = os.path.join(csd_root, arm)
        if not os.path.isdir(arm_dir):
            logging.warning("Skipping missing CSD arm: %s", arm)
            continue
        peaks_by_arm[arm] = load_csd_outputs(arm_dir)
        gfa_path = os.path.join(arm_dir, "gfa.npy")
        if os.path.isfile(gfa_path):
            background_by_arm[arm] = np.load(gfa_path)
        else:
            # Fallback: max peak value as a cheap background.
            background_by_arm[arm] = peaks_by_arm[arm]["peak_values"].max(axis=-1)

    present = [a for a in arms if a in peaks_by_arm]
    if not present:
        raise SystemExit(f"No CSD arms found under {csd_root}")
    plot_fixel_figure(rois, present, peaks_by_arm, background_by_arm, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
