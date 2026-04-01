# Discussion: J-invariance and self-supervised DWMRI denoising

This directory is for **methods discussion**, peer-review notes, open questions, and experiment rationales. Implementation code lives under [`src/`](../) in package folders (`restormer_hybrid`, `drcnet_hybrid`, etc.); link to those files from here instead of duplicating behavior.

## Canonical technical report

Stable write-up of the framework and Scheme 2 (spatial–angular hybrid):

- [`../J_invariance_DWMRI_denoising_report.md`](../J_invariance_DWMRI_denoising_report.md)

## What belongs here

| Here (`src/discussion/`) | Elsewhere (`src/<package>/`) |
|--------------------------|------------------------------|
| Assumption audits, threats to validity | Training scripts, models, configs |
| “Is this J-invariant?” reasoning threads | `data.py`, `fit.py`, `config.yaml` |
| Low-data / transfer-learning strategy notes | Runnable pipelines |
| Peer-review responses, dated decision logs | Checkpoints, outputs |

Optional long-running threads or dated notes: see [`threads/README.md`](threads/README.md).

## Cursor

The project rule **J-invariance peer reviewer** activates when you work in this tree, in the technical report, or in hybrid implementation files. The skill **j-invariance-dwmri-reviewer** adds checklists and citation guidance (`.cursor/skills/j-invariance-dwmri-reviewer/`).
