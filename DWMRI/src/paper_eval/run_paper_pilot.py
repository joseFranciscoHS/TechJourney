from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch


def _resolve_run_device(prefer_mps: bool) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class PilotSpec:
    dataset_primary: str
    dataset_smoke: str
    seed: int
    reproducible: bool
    k: int
    mask_p: float
    n_context: int
    n_preds: int
    roi_threshold: float
    epochs_short: int
    output_root: str
    registry_path: str
    run_device: str
    dbrain_nii_path: str | None
    dbrain_bvecs_path: str | None

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset_primary": self.dataset_primary,
            "dataset_smoke": self.dataset_smoke,
            "seed": self.seed,
            "reproducible": self.reproducible,
            "k": self.k,
            "mask_p": self.mask_p,
            "n_context": self.n_context,
            "n_preds": self.n_preds,
            "roi_threshold": self.roi_threshold,
            "epochs_short": self.epochs_short,
            "output_root": self.output_root,
            "registry_path": self.registry_path,
            "run_device": self.run_device,
            "dbrain_nii_path": self.dbrain_nii_path,
            "dbrain_bvecs_path": self.dbrain_bvecs_path,
        }


def frozen_spec(
    output_root: str,
    run_device: str,
    dbrain_nii_path: str | None = None,
    dbrain_bvecs_path: str | None = None,
) -> PilotSpec:
    pilot_root = os.path.join(output_root, "paper_pilot")
    return PilotSpec(
        dataset_primary="dbrain",
        dataset_smoke="stanford",
        seed=42,
        reproducible=True,
        k=10,
        mask_p=0.3,
        n_context=1,
        n_preds=1,
        roi_threshold=0.02,
        epochs_short=2,
        output_root=pilot_root,
        registry_path=os.path.join(pilot_root, "registry", "pilot_runtime.jsonl"),
        run_device=run_device,
        dbrain_nii_path=dbrain_nii_path,
        dbrain_bvecs_path=dbrain_bvecs_path,
    )


def _add_overrides(cmd: List[str], overrides: List[str]) -> List[str]:
    out = list(cmd)
    for override in overrides:
        out.extend(["--set", override])
    return out


def _base_overrides(
    spec: PilotSpec, sampling_mode: str, dataset_scope: str, run_device: str
) -> List[str]:
    prefix = f"{dataset_scope}."
    overrides = [
        f"{prefix}train.device={run_device}",
        f"{prefix}reconstruct.device={run_device}",
        f"{prefix}train.seed={spec.seed}",
        f"{prefix}train.reproducible={'true' if spec.reproducible else 'false'}",
        f"{prefix}train.progressive.enabled=false",
        f"{prefix}train.num_epochs={spec.epochs_short}",
        f"{prefix}train.batch_size=1",
        f"{prefix}data.shell_sampling_mode={sampling_mode}",
        f"{prefix}data.num_input_volumes={spec.k}",
        f"{prefix}model.in_channel={spec.k}",
        f"{prefix}data.target_channel={spec.k - 1}",
        f"{prefix}train.mask_p={spec.mask_p}",
        f"{prefix}reconstruct.mask_p={spec.mask_p}",
        f"{prefix}reconstruct.n_context_samples={spec.n_context}",
        f"{prefix}reconstruct.n_preds={spec.n_preds}",
        f"{prefix}reconstruct.metrics_roi_threshold={spec.roi_threshold}",
        # Pilot speed profile: sparse patch sampling to keep runs short.
        f"{prefix}data.step=16",
        f"{prefix}data.patch_2d_step=16",
        f"{prefix}data.patch_filter_method=none",
        f"{prefix}data.min_signal_threshold=0.0",
    ]
    if spec.dbrain_nii_path:
        overrides.extend(
            [
                f"{prefix}data.nii_path={spec.dbrain_nii_path}",
                f"{prefix}data.nii_path_lightning={spec.dbrain_nii_path}",
            ]
        )
    if spec.dbrain_bvecs_path:
        overrides.extend(
            [
                f"{prefix}data.bvecs_path={spec.dbrain_bvecs_path}",
                f"{prefix}data.bvecs_path_lightning={spec.dbrain_bvecs_path}",
            ]
        )
    if dataset_scope == "dbrain":
        overrides.extend(
            [
                f"{prefix}data.shell_gradient_volumes=12",
                f"{prefix}data.take_x=32",
                f"{prefix}data.take_y=32",
                f"{prefix}data.take_z=32",
            ]
        )
    if dataset_scope == "stanford":
        overrides.extend(
            [
                f"{prefix}data.take_x=32",
                f"{prefix}data.take_y=32",
                f"{prefix}data.take_z=32",
            ]
        )
    return overrides


def _run_cmd(cmd: List[str], execute: bool) -> int:
    print("$ " + " ".join(cmd))
    if not execute:
        return 0
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _with_cpu_fallback(cmd: List[str], run_device: str) -> List[str]:
    if run_device != "mps":
        return cmd
    out: List[str] = []
    idx = 0
    while idx < len(cmd):
        token = cmd[idx]
        if token == "--set" and idx + 1 < len(cmd):
            override = cmd[idx + 1]
            if override.endswith("train.device=mps"):
                override = override.replace("train.device=mps", "train.device=cpu")
            elif override.endswith("reconstruct.device=mps"):
                override = override.replace(
                    "reconstruct.device=mps", "reconstruct.device=cpu"
                )
            out.extend([token, override])
            idx += 2
            continue
        if token == "--device" and idx + 1 < len(cmd):
            out.extend([token, "cpu"])
            idx += 2
            continue
        out.append(token)
        idx += 1
    return out


def _phase_b_close_eval_gaps(spec: PilotSpec) -> List[List[str]]:
    common = [
        sys.executable,
        "-m",
        "restormer_hybrid_rgs.run2d_hybrid",
        "--dataset",
        spec.dataset_primary,
        "--output-root",
        spec.output_root,
        "--registry-path",
        spec.registry_path,
        "--no-wandb",
        "--recipe",
        "pilot_phase_b_restormer2d_gapcheck",
    ]
    return [
        _add_overrides(
            common,
            _base_overrides(
                spec,
                sampling_mode="rgs",
                dataset_scope=spec.dataset_primary,
                run_device=spec.run_device,
            ),
        )
    ]


def _phase_c_core_matrix(spec: PilotSpec) -> List[List[str]]:
    cmds: List[List[str]] = []
    for arch, module_3d, module_2d in [
        ("drcnet", "drcnet_hybrid_rgs.run", "drcnet_hybrid_rgs.run2d_hybrid"),
        ("restormer", "restormer_hybrid_rgs.run", "restormer_hybrid_rgs.run2d_hybrid"),
    ]:
        for sampling in ("rgs", "sequential"):
            base = [
                sys.executable,
                "-m",
                module_3d,
                "--dataset",
                spec.dataset_primary,
                "--output-root",
                spec.output_root,
                "--registry-path",
                spec.registry_path,
                "--no-wandb",
                "--no-images",
                "--recipe",
                f"pilot_phase_c_{arch}_3d_{sampling}",
            ]
            cmds.append(
                _add_overrides(
                    base,
                    _base_overrides(
                        spec,
                        sampling_mode=sampling,
                        dataset_scope=spec.dataset_primary,
                        run_device=spec.run_device,
                    ),
                )
            )

            base2d = [
                sys.executable,
                "-m",
                module_2d,
                "--dataset",
                spec.dataset_primary,
                "--output-root",
                spec.output_root,
                "--registry-path",
                spec.registry_path,
                "--no-wandb",
                "--recipe",
                f"pilot_phase_c_{arch}_2d_{sampling}",
            ]
            cmds.append(
                _add_overrides(
                    base2d,
                    _base_overrides(
                        spec,
                        sampling_mode=sampling,
                        dataset_scope=spec.dataset_primary,
                        run_device=spec.run_device,
                    ),
                )
            )

    p2s_dipy = [
        sys.executable,
        "-m",
        "p2s.run",
        "--dataset",
        spec.dataset_primary,
        "--backend",
        "dipy",
        "--seed",
        str(spec.seed),
        "--reproducible",
        "true" if spec.reproducible else "false",
        "--no-images",
        "--no-wandb",
    ]
    if spec.dbrain_nii_path:
        p2s_dipy.extend(["--nii-path", spec.dbrain_nii_path])
    if spec.dbrain_bvecs_path:
        p2s_dipy.extend(["--bvecs-path", spec.dbrain_bvecs_path])
    p2s_ref = [
        sys.executable,
        "-m",
        "p2s.run",
        "--dataset",
        spec.dataset_primary,
        "--backend",
        "sklearn_reference",
        "--seed",
        str(spec.seed),
        "--reproducible",
        "true" if spec.reproducible else "false",
        "--no-images",
        "--no-wandb",
    ]
    if spec.dbrain_nii_path:
        p2s_ref.extend(["--nii-path", spec.dbrain_nii_path])
    if spec.dbrain_bvecs_path:
        p2s_ref.extend(["--bvecs-path", spec.dbrain_bvecs_path])
    mds2s = [
        sys.executable,
        "-m",
        "mds2s.run",
        "--dataset",
        spec.dataset_primary,
        "--no-images",
        "--no-wandb",
        "--device",
        spec.run_device,
        "--num-epochs",
        str(spec.epochs_short),
        "--seed",
        str(spec.seed),
        "--reproducible",
        "true" if spec.reproducible else "false",
    ]
    if spec.dbrain_nii_path:
        mds2s.extend(["--nii-path", spec.dbrain_nii_path])
    if spec.dbrain_bvecs_path:
        mds2s.extend(["--bvecs-path", spec.dbrain_bvecs_path])
    mppca = [
        sys.executable,
        "-m",
        "paper_eval.baselines.mppca_dbrain",
        "--output-root",
        spec.output_root,
        "--seed",
        str(spec.seed),
    ]
    if spec.dbrain_nii_path:
        mppca.extend(["--nii-path", spec.dbrain_nii_path])
    if spec.dbrain_bvecs_path:
        mppca.extend(["--bvecs-path", spec.dbrain_bvecs_path])
    cmds.extend([p2s_dipy, p2s_ref, mds2s, mppca])
    return cmds


def _phase_d_core_ablations(spec: PilotSpec) -> List[List[str]]:
    cmds: List[List[str]] = []
    for k in (5, 10):
        for arch, module in [
            ("drcnet", "drcnet_hybrid_rgs.run"),
            ("restormer", "restormer_hybrid_rgs.run"),
        ]:
            base = [
                sys.executable,
                "-m",
                module,
                "--dataset",
                spec.dataset_primary,
                "--output-root",
                spec.output_root,
                "--registry-path",
                spec.registry_path,
                "--no-wandb",
                "--no-images",
                "--recipe",
                f"pilot_phase_d_{arch}_k{k}",
            ]
            ov = _base_overrides(
                spec,
                sampling_mode="rgs",
                dataset_scope=spec.dataset_primary,
                run_device=spec.run_device,
            )
            ov.extend(
                [
                    f"{spec.dataset_primary}.data.num_input_volumes={k}",
                    f"{spec.dataset_primary}.model.in_channel={k}",
                    f"{spec.dataset_primary}.data.target_channel={k - 1}",
                ]
            )
            cmds.append(_add_overrides(base, ov))

    for mask_p in (0.2, 0.3, 0.5):
        for arch, module in [
            ("drcnet", "drcnet_hybrid_rgs.run"),
            ("restormer", "restormer_hybrid_rgs.run"),
        ]:
            base = [
                sys.executable,
                "-m",
                module,
                "--dataset",
                spec.dataset_primary,
                "--output-root",
                spec.output_root,
                "--registry-path",
                spec.registry_path,
                "--no-wandb",
                "--no-images",
                "--recipe",
                f"pilot_phase_d_{arch}_mask{mask_p}",
            ]
            ov = _base_overrides(
                spec,
                sampling_mode="rgs",
                dataset_scope=spec.dataset_primary,
                run_device=spec.run_device,
            )
            ov.extend(
                [
                    f"{spec.dataset_primary}.train.mask_p={mask_p}",
                    f"{spec.dataset_primary}.reconstruct.mask_p={mask_p}",
                ]
            )
            cmds.append(_add_overrides(base, ov))
    return cmds


def _phase_e_stanford_smoke(spec: PilotSpec) -> List[List[str]]:
    cmds: List[List[str]] = []
    for arch, module in [
        ("drcnet", "drcnet_hybrid_rgs.run"),
        ("restormer", "restormer_hybrid_rgs.run"),
    ]:
        base = [
            sys.executable,
            "-m",
            module,
            "--dataset",
            spec.dataset_smoke,
            "--output-root",
            spec.output_root,
            "--registry-path",
            spec.registry_path,
            "--no-wandb",
            "--no-images",
            "--recipe",
            f"pilot_phase_e_stanford_{arch}",
        ]
        cmds.append(
            _add_overrides(
                base,
                _base_overrides(
                    spec,
                    sampling_mode="rgs",
                    dataset_scope=spec.dataset_smoke,
                    run_device=spec.run_device,
                ),
            )
        )
    return cmds


def _phase_f_consolidate(spec: PilotSpec) -> List[List[str]]:
    summary_root = os.path.join(spec.output_root, "paper_tables_dryrun")
    metrics_root = spec.output_root
    return [
        [
            sys.executable,
            "-m",
            "paper_eval.consolidate_results",
            "--root",
            metrics_root,
            "--out",
            os.path.join(summary_root, "all_metrics.csv"),
        ],
        [
            sys.executable,
            "-m",
            "paper_eval.summarize_registry",
            "--registry",
            spec.registry_path,
            "--out",
            os.path.join(summary_root, "registry_summary.csv"),
        ],
        [
            sys.executable,
            "-m",
            "paper_eval.pilot_tables",
            "--metrics-csv",
            os.path.join(summary_root, "all_metrics.csv"),
            "--registry-csv",
            os.path.join(summary_root, "registry_summary.csv"),
            "--out-dir",
            summary_root,
        ],
    ]


def _phase_commands(spec: PilotSpec) -> Dict[str, List[List[str]]]:
    return {
        "B_close_eval_gaps": _phase_b_close_eval_gaps(spec),
        "C_core_matrix": _phase_c_core_matrix(spec),
        "D_core_ablations": _phase_d_core_ablations(spec),
        "E_stanford_smoke": _phase_e_stanford_smoke(spec),
        "F_consolidate": _phase_f_consolidate(spec),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end short pilot for paper."
    )
    parser.add_argument(
        "--output-root",
        default="tmp",
        help="Parent output root; pilot artifacts are written to <output-root>/paper_pilot.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute commands. Default prints runbook only.",
    )
    parser.add_argument(
        "--only-phase",
        default=None,
        choices=[
            "B_close_eval_gaps",
            "C_core_matrix",
            "D_core_ablations",
            "E_stanford_smoke",
            "F_consolidate",
        ],
    )
    parser.add_argument("--dbrain-nii-path", default=None)
    parser.add_argument("--dbrain-bvecs-path", default=None)
    parser.add_argument(
        "--prefer-mps",
        action="store_true",
        help="Prefer MPS on Apple Silicon when CUDA is unavailable (else fallback to CPU).",
    )
    args = parser.parse_args()

    run_device = _resolve_run_device(prefer_mps=bool(args.prefer_mps))
    spec = frozen_spec(
        output_root=args.output_root,
        dbrain_nii_path=args.dbrain_nii_path,
        dbrain_bvecs_path=args.dbrain_bvecs_path,
        run_device=run_device,
    )
    Path(spec.output_root).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(spec.registry_path)).mkdir(parents=True, exist_ok=True)
    spec_path = os.path.join(spec.output_root, "pilot_spec_locked.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec.as_dict(), f, indent=2)
    print(json.dumps({"pilot_spec": spec.as_dict(), "spec_path": spec_path}, indent=2))

    phases = _phase_commands(spec)
    selected = [args.only_phase] if args.only_phase else list(phases.keys())
    failures = []
    for phase in selected:
        print(f"\n=== {phase} ===")
        for cmd in phases[phase]:
            rc = _run_cmd(cmd, execute=bool(args.execute))
            # TODO: tech debt — classify MPS failures by exception type instead of retrying all non-zero exits.
            if rc != 0 and args.execute and spec.run_device == "mps":
                fallback_cmd = _with_cpu_fallback(cmd, run_device=spec.run_device)
                if fallback_cmd != cmd:
                    print("Retrying failed MPS command on CPU fallback.")
                    rc = _run_cmd(fallback_cmd, execute=True)
            if rc != 0:
                failures.append({"phase": phase, "cmd": cmd, "exit_code": rc})
                if args.execute:
                    break
        if failures and args.execute:
            break

    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2))
        raise SystemExit(1)
    print(
        json.dumps(
            {"status": "ok", "phases": selected, "executed": bool(args.execute)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
