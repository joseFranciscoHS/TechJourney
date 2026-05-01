from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paper_eval.run_paper_pilot import frozen_spec


@dataclass(frozen=True)
class ParamRow:
    parameter: str
    pilot_value: str
    paper_protocol_value: str
    paper_manifest_value: str
    paper_config_fallback: str
    notes: str


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _extract_set_overrides(job_command: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(job_command):
        item = str(job_command[i])
        if item == "--set" and i + 1 < len(job_command):
            key_val = str(job_command[i + 1])
            if "=" in key_val:
                k, v = key_val.split("=", 1)
                out[k] = v
            i += 2
            continue
        i += 1
    return out


def _find_job(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    for job in manifest.get("jobs", []):
        if isinstance(job, dict) and job.get("id") == job_id:
            return job
    return {}


def _csv_rows(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    drcnet_cfg: dict[str, Any],
) -> list[ParamRow]:
    pilot = frozen_spec(output_root="tmp")
    proto_dbrain = protocol.get("datasets", {}).get("dbrain", {})
    proto_stanford = protocol.get("datasets", {}).get("stanford", {})

    drcnet_dbrain_job = _find_job(manifest, "drcnet_dbrain_rgs_final")
    drcnet_stanford_job = _find_job(manifest, "drcnet_stanford_rgs_final")
    mds2s_job = _find_job(manifest, "mds2s_dbrain_final")
    drcnet_dbrain_set = _extract_set_overrides(drcnet_dbrain_job.get("command", []))
    drcnet_stanford_set = _extract_set_overrides(drcnet_stanford_job.get("command", []))

    cfg_dbrain = drcnet_cfg.get("dbrain", {})
    cfg_train = cfg_dbrain.get("train", {})
    cfg_data = cfg_dbrain.get("data", {})
    cfg_recon = cfg_dbrain.get("reconstruct", {})

    rows = [
        ParamRow(
            parameter="dbrain.train.seed",
            pilot_value=str(pilot.seed),
            paper_protocol_value=str(proto_dbrain.get("seed", "")),
            paper_manifest_value=drcnet_dbrain_set.get("dbrain.train.seed", ""),
            paper_config_fallback=str(cfg_train.get("seed", "")),
            notes="Pilot uses frozen short-run seed; paper final seed is protocol-fixed.",
        ),
        ParamRow(
            parameter="dbrain.train.reproducible",
            pilot_value=str(pilot.reproducible).lower(),
            paper_protocol_value=str(
                protocol.get("reproducibility", {})
                .get("deterministic", {})
                .get("reproducible", "")
            ).lower(),
            paper_manifest_value=drcnet_dbrain_set.get("dbrain.train.reproducible", ""),
            paper_config_fallback=str(cfg_train.get("reproducible", "")).lower(),
            notes="Manifest explicitly forces reproducible=true for final runs.",
        ),
        ParamRow(
            parameter="dbrain.data.num_input_volumes(K)",
            pilot_value=str(pilot.k),
            paper_protocol_value=str(proto_dbrain.get("k_input", "")),
            paper_manifest_value=drcnet_dbrain_set.get(
                "dbrain.data.num_input_volumes", ""
            ),
            paper_config_fallback=str(cfg_data.get("num_input_volumes", "")),
            notes="Paper final keeps full K=24.",
        ),
        ParamRow(
            parameter="dbrain.data.shell_gradient_volumes(G)",
            pilot_value="12",
            paper_protocol_value=str(proto_dbrain.get("g_shell", "")),
            paper_manifest_value=drcnet_dbrain_set.get(
                "dbrain.data.shell_gradient_volumes", ""
            ),
            paper_config_fallback=str(cfg_data.get("shell_gradient_volumes", "")),
            notes="Pilot speed profile reduces shell size.",
        ),
        ParamRow(
            parameter="dbrain.data.target_channel",
            pilot_value=str(pilot.k - 1),
            paper_protocol_value=str(proto_dbrain.get("target_channel", "")),
            paper_manifest_value=drcnet_dbrain_set.get(
                "dbrain.data.target_channel", ""
            ),
            paper_config_fallback=str(cfg_data.get("target_channel", "")),
            notes="Derived as K-1 in pilot; fixed to 23 in final.",
        ),
        ParamRow(
            parameter="dbrain.train.num_epochs",
            pilot_value=str(pilot.epochs_short),
            paper_protocol_value="(not pinned in protocol YAML)",
            paper_manifest_value="(inherits config)",
            paper_config_fallback=str(cfg_train.get("num_epochs", "")),
            notes="Final training length depends on config/progressive schedule.",
        ),
        ParamRow(
            parameter="dbrain.train.progressive.enabled",
            pilot_value="false",
            paper_protocol_value="progressive_vs_standard listed as ablation",
            paper_manifest_value="(inherits config)",
            paper_config_fallback=str(
                cfg_train.get("progressive", {}).get("enabled", "")
            ),
            notes="Manifest final does not yet expand ablation jobs.",
        ),
        ParamRow(
            parameter="dbrain.train.batch_size",
            pilot_value="1",
            paper_protocol_value="(not pinned in protocol YAML)",
            paper_manifest_value="(inherits config)",
            paper_config_fallback=str(cfg_train.get("batch_size", "")),
            notes="Pilot uses minimal batch for speed and CPU fallback.",
        ),
        ParamRow(
            parameter="dbrain.reconstruct.n_context_samples",
            pilot_value=str(pilot.n_context),
            paper_protocol_value="n_context_sensitivity listed as ablation",
            paper_manifest_value="(inherits config)",
            paper_config_fallback=str(cfg_recon.get("n_context_samples", "")),
            notes="Pilot fixed to 1; final defaults from config unless overridden.",
        ),
        ParamRow(
            parameter="dbrain.reconstruct.n_preds",
            pilot_value=str(pilot.n_preds),
            paper_protocol_value="(not pinned in protocol YAML)",
            paper_manifest_value="(inherits config)",
            paper_config_fallback=str(cfg_recon.get("n_preds", "")),
            notes="Pilot fixed to 1 for short inference.",
        ),
        ParamRow(
            parameter="dbrain.reconstruct.metrics_roi_threshold",
            pilot_value=str(pilot.roi_threshold),
            paper_protocol_value=str(
                protocol.get("evaluation_policy", {}).get("metrics_roi_threshold", "")
            ),
            paper_manifest_value=drcnet_dbrain_set.get(
                "dbrain.reconstruct.metrics_roi_threshold", ""
            ),
            paper_config_fallback=str(cfg_recon.get("metrics_roi_threshold", "")),
            notes="Aligned between pilot and paper.",
        ),
        ParamRow(
            parameter="dbrain.data.take_x/take_y/take_z",
            pilot_value="32/32/32",
            paper_protocol_value="canonical split from config",
            paper_manifest_value="(inherits config)",
            paper_config_fallback=f"{cfg_data.get('take_x', '')}/{cfg_data.get('take_y', '')}/{cfg_data.get('take_z', '')}",
            notes="Pilot crops aggressively for runtime.",
        ),
        ParamRow(
            parameter="dbrain.data.step",
            pilot_value="16",
            paper_protocol_value="(not pinned in protocol YAML)",
            paper_manifest_value="(inherits config)",
            paper_config_fallback=str(cfg_data.get("step", "")),
            notes="Pilot uses sparse patch sampling.",
        ),
        ParamRow(
            parameter="dbrain.data.shell_sampling_mode",
            pilot_value="rgs + sequential (phase C matrix)",
            paper_protocol_value=str(proto_dbrain.get("shell_sampling_mode", "")),
            paper_manifest_value=drcnet_dbrain_set.get(
                "dbrain.data.shell_sampling_mode", ""
            ),
            paper_config_fallback=str(cfg_data.get("shell_sampling_mode", "")),
            notes="Final manifest currently runs RGS main jobs only.",
        ),
        ParamRow(
            parameter="stanford.data.num_input_volumes(K)",
            pilot_value=str(pilot.k),
            paper_protocol_value=str(proto_stanford.get("k_input", "")),
            paper_manifest_value=drcnet_stanford_set.get(
                "stanford.data.num_input_volumes", ""
            ),
            paper_config_fallback="(see drcnet config stanford profile)",
            notes="Pilot smoke keeps small K; final uses K=24.",
        ),
        ParamRow(
            parameter="stanford.data.shell_gradient_volumes(G)",
            pilot_value="(inherits stanford config; pilot only crops xyz)",
            paper_protocol_value=str(proto_stanford.get("g_shell", "")),
            paper_manifest_value=drcnet_stanford_set.get(
                "stanford.data.shell_gradient_volumes", ""
            ),
            paper_config_fallback="(see drcnet config stanford profile)",
            notes="Final stanford jobs explicitly set G=150.",
        ),
        ParamRow(
            parameter="p2s paths",
            pilot_value="optional --nii-path / --bvecs-path",
            paper_protocol_value="fixed split strategy",
            paper_manifest_value="(no path flags in manifest)",
            paper_config_fallback="p2s/config.yaml",
            notes="Pilot supports local/cloud path overrides directly.",
        ),
        ParamRow(
            parameter="mds2s device + epochs flags",
            pilot_value="--device auto, --num-epochs=2",
            paper_protocol_value="(not pinned in protocol YAML)",
            paper_manifest_value=f"{'--device present' if '--device' in mds2s_job.get('command', []) else 'no --device'}; {'--num-epochs present' if '--num-epochs' in mds2s_job.get('command', []) else 'no --num-epochs'}",
            paper_config_fallback="mds2s/config.yaml",
            notes="Final manifest currently relies on config defaults.",
        ),
        ParamRow(
            parameter="mppca runner flavor",
            pilot_value="paper_eval.baselines.mppca_dbrain",
            paper_protocol_value="mppca baseline",
            paper_manifest_value="paper_eval.baselines.mppca_run over shared npy export",
            paper_config_fallback="n/a",
            notes="Paper final inserts export_dbrain_npy step.",
        ),
    ]
    return rows


def _compute_manifest_gap(
    protocol: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    ablations = protocol.get("matrix", {}).get("ablations", [])
    job_ids = [
        str(job.get("id", ""))
        for job in manifest.get("jobs", [])
        if isinstance(job, dict)
    ]
    manifest_blob = " ".join(job_ids)
    missing: list[str] = []
    matched: list[str] = []
    heuristic = {
        "sampling_sequential_vs_rgs": ["sequential", "rgs"],
        "conv2d_vs_conv3d": ["2d", "conv2d", "conv3d"],
        "k_sweep": ["_k", "k_sweep", "k5", "k10", "num_input_volumes"],
        "mask_p_sensitivity": ["mask", "mask_p"],
        "n_context_sensitivity": ["n_context", "ncontext"],
        "progressive_vs_standard": ["progressive", "standard"],
    }
    for abl in ablations:
        terms = heuristic.get(str(abl), [str(abl)])
        if any(term in manifest_blob for term in terms):
            matched.append(str(abl))
        else:
            missing.append(str(abl))
    recommendation = (
        "Manifest ablation coverage matches protocol matrix."
        if not missing
        else "Generate/extend paper manifest with explicit ablation jobs to fully "
        "satisfy paper_protocol_final.yaml matrix.ablations."
    )
    return {
        "protocol_ablations": ablations,
        "manifest_job_count": len(job_ids),
        "manifest_ids": job_ids,
        "matched_ablations_heuristic": matched,
        "missing_ablations_heuristic": missing,
        "recommendation": recommendation,
    }


def _write_csv(rows: list[ParamRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "parameter",
                "pilot_value",
                "paper_protocol_value",
                "paper_manifest_value",
                "paper_config_fallback",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "parameter": row.parameter,
                    "pilot_value": row.pilot_value,
                    "paper_protocol_value": row.paper_protocol_value,
                    "paper_manifest_value": row.paper_manifest_value,
                    "paper_config_fallback": row.paper_config_fallback,
                    "notes": row.notes,
                }
            )


def _write_gap_report(gap: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(gap, fh, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pilot-vs-paper parameter comparison CSV and manifest-gap report."
    )
    parser.add_argument(
        "--protocol",
        default="experiments/paper_protocol_final.yaml",
    )
    parser.add_argument(
        "--manifest",
        default="experiments/paper_manifest_final.yaml",
    )
    parser.add_argument(
        "--drcnet-config",
        default="src/drcnet_hybrid_rgs/config.yaml",
    )
    parser.add_argument(
        "--out-csv",
        default="experiments/pilot_vs_paper_params.csv",
    )
    parser.add_argument(
        "--out-gap-report",
        default="experiments/paper_manifest_gap_report.yaml",
    )
    args = parser.parse_args()

    protocol = _load_yaml(Path(args.protocol))
    manifest = _load_yaml(Path(args.manifest))
    drcnet_cfg = _load_yaml(Path(args.drcnet_config))

    rows = _csv_rows(protocol=protocol, manifest=manifest, drcnet_cfg=drcnet_cfg)
    _write_csv(rows, Path(args.out_csv))

    gap = _compute_manifest_gap(protocol=protocol, manifest=manifest)
    _write_gap_report(gap, Path(args.out_gap_report))

    print(f"wrote_csv={args.out_csv}")
    print(f"wrote_gap_report={args.out_gap_report}")


if __name__ == "__main__":
    main()
