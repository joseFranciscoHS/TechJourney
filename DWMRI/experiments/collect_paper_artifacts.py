from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _infer_method_from_path(rel_path: str) -> str:
    rel = rel_path.lower()
    if "drcnet_hybrid_rgs" in rel:
        return "drcnet_hybrid_rgs"
    if "restormer_hybrid_rgs" in rel:
        return "restormer_hybrid_rgs"
    if "p2s" in rel:
        return "patch2self"
    if "mds2s" in rel:
        return "mds2s"
    if "mppca" in rel:
        return "mppca"
    return "unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.rglob("metrics.json")):
        rel = _safe_rel(metrics_path, output_root)
        metrics = _read_json(metrics_path) or {}
        roi = _read_json(metrics_path.with_name("metrics_roi.json")) or {}
        dti = _read_json(metrics_path.with_name("dti_metrics.json")) or {}
        run_manifest = _read_json(metrics_path.with_name("run_manifest.json")) or {}
        config = run_manifest.get("config", {}) if isinstance(run_manifest, dict) else {}
        rows.append(
            {
                "metrics_relpath": rel,
                "method": _infer_method_from_path(rel),
                "dataset": config.get("dataset"),
                "dimensionality": config.get("dimensionality"),
                "sampling_mode": config.get("sampling_mode"),
                "k_input": config.get("k_input"),
                "n_context_samples": config.get("n_context_samples"),
                "n_preds": config.get("n_preds"),
                "psnr": metrics.get("psnr"),
                "ssim": metrics.get("ssim"),
                "mse": metrics.get("mse"),
                "psnr_roi": roi.get("psnr"),
                "ssim_roi": roi.get("ssim"),
                "mse_roi": roi.get("mse"),
                "fa_mae": dti.get("fa_mae"),
                "md_mae": dti.get("md_mae"),
                "ad_mae": dti.get("ad_mae"),
                "rd_mae": dti.get("rd_mae"),
            }
        )
    return rows


def _registry_rows(registry_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in _read_jsonl(registry_path):
        ctrl = rec.get("control_metrics") or {}
        inf = rec.get("inference_config") or {}
        samp = rec.get("sampling_config") or {}
        rows.append(
            {
                "exp_id": rec.get("exp_id"),
                "job_id": rec.get("job_id"),
                "recipe": rec.get("recipe"),
                "status": rec.get("status"),
                "dataset": rec.get("dataset"),
                "architecture": rec.get("architecture"),
                "dimensionality": rec.get("dimensionality"),
                "sampling_mode": rec.get("sampling_mode"),
                "k_input": samp.get("k_input"),
                "g_shell": samp.get("g_shell"),
                "n_context_samples": inf.get("n_context_samples"),
                "n_preds": inf.get("n_preds"),
                "n_params": ctrl.get("n_params"),
                "sec_per_epoch": ctrl.get("sec_per_epoch"),
                "sec_per_volume": ctrl.get("sec_per_volume"),
                "peak_gpu_mem_mb": ctrl.get("peak_gpu_mem_mb"),
                "duration_s": rec.get("duration_s"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect standardized paper artifact tables from output-root."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--registry", default=None, help="Registry JSONL path (optional).")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    registry_path = (
        Path(args.registry).expanduser().resolve()
        if args.registry
        else output_root / "registry.jsonl"
    )

    metrics_rows = _metric_rows(output_root)
    runtime_rows = _registry_rows(registry_path)
    _write_csv(out_dir / "paper_metrics_summary.csv", metrics_rows)
    _write_csv(out_dir / "paper_runtime_summary.csv", runtime_rows)

    manifest = {
        "status": "ok",
        "output_root": str(output_root),
        "registry_path": str(registry_path),
        "metrics_rows": len(metrics_rows),
        "runtime_rows": len(runtime_rows),
        "outputs": {
            "metrics_csv": str(out_dir / "paper_metrics_summary.csv"),
            "runtime_csv": str(out_dir / "paper_runtime_summary.csv"),
        },
    }
    (out_dir / "paper_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
