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


def _mask_tag(mask_p: float) -> str:
    return str(mask_p).replace("0.", "0").replace(".", "")


def _sigma_tag(sigma: float) -> str:
    return f"{int(round(float(sigma) * 1000)):03d}"


def validate(protocol_path: Path, manifest_path: Path):
    protocol = _load_yaml(protocol_path)
    manifest = _load_yaml(manifest_path)
    jobs = _job_map(manifest)
    errors = []
    warnings = []

    required_jobs = [
        "export_dbrain_npy_final",
        "mppca_dbrain_final",
        "drcnet_dbrain_rgs_final",
        "restormer_dbrain_rgs_final",
        "drcnet_dbrain_supervised_upperbound_final",
        "restormer_dbrain_supervised_upperbound_final",
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
            "drcnet_dbrain_seq_k16_ablation",
            "restormer_dbrain_seq_k16_ablation",
        ],
        "conv2d_vs_conv3d": [
            "drcnet_dbrain_2d_rgs_k16_ablation",
            "restormer_dbrain_2d_rgs_k16_ablation",
        ],
        "progressive_vs_standard": [
            "drcnet_dbrain_progressive_off_ablation",
            "restormer_dbrain_progressive_off_ablation",
        ],
    }
    protocol_ablations = [
        str(a) for a in protocol.get("matrix", {}).get("ablations", [])
    ]
    for ablation in protocol_ablations:
        for jid in required_ablation_jobs.get(ablation, []):
            if jid not in jobs:
                errors.append(f"missing_ablation_job:{ablation}:{jid}")

    sweep_cfg = protocol.get("matrix", {}).get("sweep_coverage", {})
    sweep_archs = [str(a) for a in sweep_cfg.get("architectures", [])]
    k_sweep_cfg = sweep_cfg.get("k_sweep", {})
    mask_values = [float(v) for v in sweep_cfg.get("mask_p_sensitivity", [])]
    nctx_values = [int(v) for v in sweep_cfg.get("n_context_sensitivity", [])]

    for arch in sweep_archs:
        for dataset, k_values in k_sweep_cfg.items():
            for k in [int(v) for v in k_values]:
                jid = f"{arch}_{dataset}_k{k}_ablation"
                if jid not in jobs:
                    errors.append(f"missing_ablation_job:k_sweep:{jid}")
                    continue
                cmd = _command_text(jobs[jid])
                for token in (
                    f"{dataset}.data.shell_sampling_mode=rgs",
                    f"{dataset}.model.in_channel={k}",
                    f"{dataset}.data.num_input_volumes={k}",
                    f"{dataset}.data.target_channel={k - 1}",
                ):
                    if token not in cmd:
                        errors.append(f"{jid}:missing_token:{token}")

    for arch in sweep_archs:
        for dataset in k_sweep_cfg.keys():
            for mask_p in mask_values:
                jid = f"{arch}_{dataset}_maskp_{_mask_tag(mask_p)}_ablation"
                if jid not in jobs:
                    errors.append(f"missing_ablation_job:mask_p_sensitivity:{jid}")
                    continue
                cmd = _command_text(jobs[jid])
                for token in (
                    f"{dataset}.data.shell_sampling_mode=rgs",
                    f"{dataset}.train.mask_p={mask_p}",
                    f"{dataset}.reconstruct.mask_p={mask_p}",
                ):
                    if token not in cmd:
                        errors.append(f"{jid}:missing_token:{token}")

    for arch in sweep_archs:
        for dataset in k_sweep_cfg.keys():
            for nctx in nctx_values:
                jid = f"{arch}_{dataset}_ncontext_{nctx}_ablation"
                if jid not in jobs:
                    errors.append(f"missing_ablation_job:n_context_sensitivity:{jid}")
                    continue
                cmd = _command_text(jobs[jid])
                for token in (
                    f"{dataset}.data.shell_sampling_mode=rgs",
                    f"{dataset}.reconstruct.n_context_samples={nctx}",
                ):
                    if token not in cmd:
                        errors.append(f"{jid}:missing_token:{token}")

    npreds_cfg = sweep_cfg.get("n_preds_sensitivity", {})
    npreds_values = [int(v) for v in npreds_cfg.get("dbrain", [])]
    npreds_archs = [str(v) for v in npreds_cfg.get("architectures", [])]
    fixed_nctx = int(npreds_cfg.get("fixed_n_context_samples", 16))
    for arch in npreds_archs:
        for npreds in npreds_values:
            jid = f"{arch}_dbrain_npreds_{npreds}_ablation"
            if jid not in jobs:
                errors.append(f"missing_ablation_job:n_preds_sensitivity:{jid}")
                continue
            cmd = _command_text(jobs[jid])
            for token in (
                "dbrain.data.shell_sampling_mode=rgs",
                f"dbrain.reconstruct.n_context_samples={fixed_nctx}",
                f"dbrain.reconstruct.n_preds={npreds}",
            ):
                if token not in cmd:
                    errors.append(f"{jid}:missing_token:{token}")

    dbrain_seed = protocol["datasets"]["dbrain"]["seed"]
    stanford_seed = protocol["datasets"]["stanford"]["seed"]
    roi_thr = protocol["evaluation_policy"]["metrics_roi_threshold"]
    rescale_mode = protocol["evaluation_policy"]["rescale_mode"]

    sigma_cfg = sweep_cfg.get("noise_sigma_sensitivity", {})
    sigma_values = [float(v) for v in sigma_cfg.get("dbrain", [])]
    sigma_archs = [
        str(v) for v in sigma_cfg.get("methods", {}).get("proposed_architectures", [])
    ]
    sigma_baselines = [
        str(v) for v in sigma_cfg.get("methods", {}).get("baselines", [])
    ]

    for sigma in sigma_values:
        stag = _sigma_tag(sigma)
        export_jid = f"export_dbrain_npy_sigma_{stag}_final"
        if export_jid not in jobs:
            errors.append(f"missing_ablation_job:noise_sigma_sensitivity:{export_jid}")
        else:
            cmd = _command_text(jobs[export_jid])
            for token in (
                "experiments/paper_export_dbrain_volume_pair.py",
                f"--noise-sigma {sigma:.2f}",
            ):
                if token not in cmd:
                    errors.append(f"{export_jid}:missing_token:{token}")

        if "mppca" in sigma_baselines:
            jid = f"mppca_dbrain_sigma_{stag}_final"
            if jid not in jobs:
                errors.append(f"missing_ablation_job:noise_sigma_sensitivity:{jid}")
            else:
                cmd = _command_text(jobs[jid])
                for token in (
                    "paper_eval.baselines.mppca_run",
                    f"../tmp/paper_final_shared_npy_sigma_{stag}/noisy_dwi_xyzv.npy",
                ):
                    if token not in cmd:
                        errors.append(f"{jid}:missing_token:{token}")

        if "p2s_dipy" in sigma_baselines:
            jid = f"p2s_dbrain_dipy_sigma_{stag}_final"
            if jid not in jobs:
                errors.append(f"missing_ablation_job:noise_sigma_sensitivity:{jid}")
            else:
                cmd = _command_text(jobs[jid])
                for token in ("--backend dipy", f"--noise-sigma {sigma:.2f}"):
                    if token not in cmd:
                        errors.append(f"{jid}:missing_token:{token}")

        if "p2s_sklearn_reference" in sigma_baselines:
            jid = f"p2s_dbrain_sklearn_reference_sigma_{stag}_final"
            if jid not in jobs:
                errors.append(f"missing_ablation_job:noise_sigma_sensitivity:{jid}")
            else:
                cmd = _command_text(jobs[jid])
                for token in (
                    "--backend sklearn_reference",
                    f"--noise-sigma {sigma:.2f}",
                ):
                    if token not in cmd:
                        errors.append(f"{jid}:missing_token:{token}")

        if "mds2s" in sigma_baselines:
            jid = f"mds2s_dbrain_sigma_{stag}_final"
            if jid not in jobs:
                errors.append(f"missing_ablation_job:noise_sigma_sensitivity:{jid}")
            else:
                cmd = _command_text(jobs[jid])
                for token in (
                    "--dataset dbrain",
                    f"--noise-sigma {sigma:.2f}",
                    f"--seed {dbrain_seed}",
                    "--reproducible true",
                ):
                    if token not in cmd:
                        errors.append(f"{jid}:missing_token:{token}")

        for arch in sigma_archs:
            jid = f"{arch}_dbrain_sigma_{stag}_ablation"
            if jid not in jobs:
                errors.append(f"missing_ablation_job:noise_sigma_sensitivity:{jid}")
                continue
            cmd = _command_text(jobs[jid])
            for token in (
                "dbrain.data.shell_sampling_mode=rgs",
                f"dbrain.data.noise_sigma={sigma:.2f}",
            ):
                if token not in cmd:
                    errors.append(f"{jid}:missing_token:{token}")

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

    supervised_expected_dirs = {
        "drcnet_dbrain_supervised_upperbound_final": (
            "dbrain.train.checkpoint_dir=drcnet_hybrid_rgs/checkpoints/dbrain_supervised",
            "dbrain.reconstruct.metrics_dir=drcnet_hybrid_rgs/metrics/dbrain_supervised",
        ),
        "restormer_dbrain_supervised_upperbound_final": (
            "dbrain.train.checkpoint_dir=restormer_hybrid_rgs/checkpoints/dbrain_supervised",
            "dbrain.reconstruct.metrics_dir=restormer_hybrid_rgs/metrics/dbrain_supervised",
        ),
    }
    for jid in supervised_expected_dirs:
        if jid not in jobs:
            continue
        cmd = _command_text(jobs[jid])
        for token in (
            f"dbrain.train.seed={dbrain_seed}",
            "dbrain.train.reproducible=true",
            "dbrain.train.supervised=true",
            f"dbrain.reconstruct.metrics_roi_threshold={roi_thr}",
            f"dbrain.reconstruct.rescale_mode={rescale_mode}",
            "dbrain.reconstruct.compute_dti=true",
            "--regime supervised",
        ):
            if token not in cmd:
                errors.append(f"{jid}:missing_token:{token}")
        for token in supervised_expected_dirs[jid]:
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

    parity_cfg = protocol.get("matrix", {}).get("parity_2d_vs_3d", {})
    parity_jobs = [str(j) for j in parity_cfg.get("required_jobs", [])]
    pc = parity_cfg.get("constraints", {})
    for jid in parity_jobs:
        if jid not in jobs:
            warnings.append(f"parity2d_missing_job:{jid}")
            continue
        cmd = _command_text(jobs[jid])
        expected_tokens = (
            f"dbrain.train.seed={pc.get('seed', dbrain_seed)}",
            f"dbrain.data.shell_sampling_mode={pc.get('shell_sampling_mode', 'rgs')}",
            f"dbrain.data.num_input_volumes={pc.get('k_input', 16)}",
            f"dbrain.data.target_channel={pc.get('target_channel', 15)}",
            f"dbrain.reconstruct.mask_p={pc.get('mask_p', 0.3)}",
            f"dbrain.reconstruct.metrics_roi_threshold={pc.get('metrics_roi_threshold', roi_thr)}",
            f"dbrain.reconstruct.rescale_mode={pc.get('rescale_mode', rescale_mode)}",
            f"dbrain.reconstruct.clip_to_range={str(pc.get('clip_to_range', True)).lower()}",
            f"dbrain.reconstruct.compute_dti={str(pc.get('compute_dti', True)).lower()}",
            "--regime self_supervised",
        )
        for token in expected_tokens:
            if token not in cmd:
                warnings.append(f"parity2d_warning:{jid}:missing_token:{token}")

    status = "ok" if not errors else "failed"
    return {
        "status": status,
        "protocol_path": str(protocol_path),
        "manifest_path": str(manifest_path),
        "num_jobs": len(manifest.get("jobs", [])),
        "errors": errors,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate final paper protocol and manifest coherence."
    )
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
