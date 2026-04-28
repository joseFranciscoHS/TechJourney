# Pilot vs Full-Test Gap Checklist

Scope: this checklist is based on the clean pilot execution under `tmp/paper_pilot` with synthetic DBrain input and Stanford sample data.

## Done

- `done` Pilot end-to-end executed for phases B/C/D/E/F with `status: ok`.
- `done` All pilot jobs recorded as successful (`21/21`) in `tmp/paper_pilot/registry/pilot_runtime.jsonl`.
- `done` Consolidated pilot tables were generated in `tmp/paper_pilot/paper_tables_dryrun`.
- `done` Paper artifact consolidation generated:
  - `tmp/paper_pilot/paper_artifacts/paper_metrics_summary.csv`
  - `tmp/paper_pilot/paper_artifacts/paper_runtime_summary.csv`
- `done` Protocol/manifest validation passed with no errors or warnings:
  - `tmp/paper_pilot/paper_artifacts/protocol_validation.json`
- `done` Dependency probe passed (`dipy`, `wandb`, `numpy`, `torch` available):
  - `tmp/paper_pilot/paper_artifacts/protocol_dependencies.json`
- `done` Protocol closure report generated:
  - `tmp/paper_pilot/paper_artifacts/protocol_closure_report.json`

## Missing for full test

- `missing` Run the full matrix on real DBrain paths (pilot used synthetic DBrain volume pair).
- `missing` Complete publication-grade evidence on real data for:
  - main metrics table (PSNR/SSIM/MSE full+ROI),
  - DTI errors (FA/MD/AD/RD),
  - runtime/per-method comparisons.
- `missing` Final figures and manuscript-ready tables referenced in the paper plan.
- `missing` Explicit final pass confirming evaluation policy consistency on real runs (ROI threshold, rescaling policy, seed/split/crop reporting).

## Blocked / limitations

- `blocked` Backend traceability in registry is incomplete for some jobs (`runtime_device` not consistently populated), so per-job MPS-vs-CPU evidence is not fully machine-readable from registry alone.
- `blocked` MPS fallback reason classification is coarse (current fallback retries any failing MPS command on CPU in pilot runner).

## Recommended next actions

1. Re-run the same pilot/full manifests against real DBrain inputs using `--dbrain-nii-path` and `--dbrain-bvecs-path`.
2. Export a device audit artifact (per recipe: requested device, effective device, fallback yes/no).
3. Freeze paper tables/figures from real-data outputs after the protocol validator remains `status: ok`.
