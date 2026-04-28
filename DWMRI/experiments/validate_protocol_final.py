import argparse
import json
from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _job_map(manifest):
    return {j["id"]: j for j in manifest.get("jobs", [])}


def _command_text(job):
    return " ".join(str(x) for x in job.get("command", []))


def validate(protocol_path: Path, manifest_path: Path):
    protocol = _load_yaml(protocol_path)
    manifest = _load_yaml(manifest_path)
    jobs = _job_map(manifest)
    errors = []

    required_jobs = [
        "export_dbrain_npy_final",
        "mppca_dbrain_final",
        "drcnet_dbrain_rgs_final",
        "restormer_dbrain_rgs_final",
        "drcnet_stanford_rgs_final",
        "restormer_stanford_rgs_final",
        "p2s_dbrain_dipy_final",
        "p2s_dbrain_sklearn_reference_final",
        "mds2s_dbrain_final",
        "mds2s_stanford_final",
    ]
    for jid in required_jobs:
        if jid not in jobs:
            errors.append(f"missing_job:{jid}")

    required_ablation_jobs = {
        "sampling_sequential_vs_rgs": [
            "drcnet_dbrain_seq_k24_ablation",
            "restormer_dbrain_seq_k24_ablation",
        ],
        "conv2d_vs_conv3d": [
            "drcnet_dbrain_2d_rgs_k24_ablation",
            "restormer_dbrain_2d_rgs_k24_ablation",
        ],
        "k_sweep": ["drcnet_dbrain_k10_ablation"],
        "mask_p_sensitivity": ["drcnet_dbrain_maskp_02_ablation"],
        "n_context_sensitivity": ["drcnet_dbrain_ncontext_12_ablation"],
        "progressive_vs_standard": [
            "drcnet_dbrain_progressive_off_ablation",
            "restormer_dbrain_progressive_off_ablation",
        ],
    }
    protocol_ablations = [str(a) for a in protocol.get("matrix", {}).get("ablations", [])]
    for ablation in protocol_ablations:
        for jid in required_ablation_jobs.get(ablation, []):
            if jid not in jobs:
                errors.append(f"missing_ablation_job:{ablation}:{jid}")

    dbrain_seed = protocol["datasets"]["dbrain"]["seed"]
    stanford_seed = protocol["datasets"]["stanford"]["seed"]
    roi_thr = protocol["evaluation_policy"]["metrics_roi_threshold"]
    rescale_mode = protocol["evaluation_policy"]["rescale_mode"]

    for jid in ("drcnet_dbrain_rgs_final", "restormer_dbrain_rgs_final"):
        if jid not in jobs:
            continue
        cmd = _command_text(jobs[jid])
        for token in (
            f"dbrain.train.seed={dbrain_seed}",
            "dbrain.train.reproducible=true",
            f"dbrain.reconstruct.metrics_roi_threshold={roi_thr}",
            f"dbrain.reconstruct.rescale_mode={rescale_mode}",
            "dbrain.reconstruct.compute_dti=true",
        ):
            if token not in cmd:
                errors.append(f"{jid}:missing_token:{token}")

    for jid in ("drcnet_stanford_rgs_final", "restormer_stanford_rgs_final"):
        if jid not in jobs:
            continue
        cmd = _command_text(jobs[jid])
        for token in (
            f"stanford.train.seed={stanford_seed}",
            "stanford.train.reproducible=true",
            f"stanford.reconstruct.metrics_roi_threshold={roi_thr}",
            f"stanford.reconstruct.rescale_mode={rescale_mode}",
            "stanford.reconstruct.compute_dti=false",
        ):
            if token not in cmd:
                errors.append(f"{jid}:missing_token:{token}")

    p2s_jobs = {
        "p2s_dbrain_dipy_final": "dipy",
        "p2s_dbrain_sklearn_reference_final": "sklearn_reference",
    }
    for jid, backend in p2s_jobs.items():
        if jid not in jobs:
            continue
        cmd = _command_text(jobs[jid])
        for token in (
            "--dataset dbrain",
            f"--backend {backend}",
            f"--seed {dbrain_seed}",
            "--reproducible true",
        ):
            if token not in cmd:
                errors.append(f"{jid}:missing_token:{token}")

    for jid, seed, ds in (
        ("mds2s_dbrain_final", dbrain_seed, "dbrain"),
        ("mds2s_stanford_final", stanford_seed, "stanford"),
    ):
        if jid not in jobs:
            continue
        cmd = _command_text(jobs[jid])
        for token in (f"--dataset {ds}", f"--seed {seed}", "--reproducible true"):
            if token not in cmd:
                errors.append(f"{jid}:missing_token:{token}")

    status = "ok" if not errors else "failed"
    return {
        "status": status,
        "protocol_path": str(protocol_path),
        "manifest_path": str(manifest_path),
        "num_jobs": len(manifest.get("jobs", [])),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate final paper protocol and manifest coherence.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = validate(Path(args.protocol), Path(args.manifest))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
