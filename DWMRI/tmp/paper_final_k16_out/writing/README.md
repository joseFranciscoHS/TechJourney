# Writing Prompts for DWMRI Hybrid RGS Paper

This directory contains **sequentially numbered LLM prompts** for drafting sections of the paper. Each prompt is a self-contained instruction file designed to be fed to a writing-assistant LLM (e.g. ChatGPT, Claude) along with the listed attachment files.

---

## How to use

1. Open the prompt file (e.g. `001_orientation_conditioning.md`).
2. Attach every file listed in the **Files to attach** section of that prompt.
3. Paste or send the **Prompt** section to the LLM.
4. The LLM should produce a draft you can iterate on.

### Attaching files

Each prompt lists files as paths relative to the repository root (`DWMRI/`). Some LLM interfaces let you upload files directly; others accept pasted text. When pasting, wrap each file in a clear delimiter so the LLM can distinguish them:

```
=== FILE: src/J_invariance_DWMRI_denoising_report_hybrid_rgs.md ===
<contents>
=== END FILE ===
```

---

## Naming convention

Prompts are numbered with a zero-padded three-digit prefix:

```
NNN_<short_slug>.md
```

- **NNN** determines execution order (001, 002, ...).
- **short_slug** is a lowercase, underscore-separated label describing the section.

### Adding a new prompt

1. Pick the next available number (check existing files).
2. Create `NNN_<slug>.md` following the template below.
3. List **all** files the LLM needs to see, even if they were attached in a previous prompt (each prompt should be self-contained).

---

## Prompt template

Copy the block below to start a new prompt file:

```markdown
# NNN — Section Title

## Context for the LLM

<Brief paragraph telling the LLM what this paper is about and where this
section fits in the overall structure. Include any decisions already made
(e.g. "we decided not to include X").>

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `path/to/file.md` | Description of why the LLM needs it |

## Prompt

<The actual instruction to the LLM. Be specific about:
  - Target length (words / pages / paragraphs)
  - Tone and style (technical, concise, IEEE-style, etc.)
  - What to include and what to omit
  - LaTeX conventions if applicable
  - Any tables or figures to reference / produce
>

## Expected output

<Describe what the LLM should hand back: a LaTeX snippet, a markdown
draft, a table, etc.>
```

---

## Prompt index

| # | Slug | Section | Status |
|---|------|---------|--------|
| 001 | `orientation_conditioning` | Gradient Direction Conditioning via FiLM (ablation subsection) | ready |
| 002 | `main_comparison_table` | Main Comparison Table and Results (primary experimental results) | ready |
| 003 | `sampling_ablations` | Sequential vs RGS and K Sweep (sampling strategy ablations) | ready |
| 004 | `3d_vs_2d_ablation` | 3D vs 2D Convolutions (critical architectural ablation) | ready |
| 005 | `sigma_robustness` | Robustness to Noise Level (sigma sweep across SNR regimes) | ready |
| 006 | `stanford_generalization` | Generalization to Real Scanner Noise (Stanford HARDI evaluation) | ready |
| 007 | `objective_controlled_ablation` | Objective-Controlled DRCNet Ablation (angular/spatial/random component isolation) | ready |
| 020 | `k16_rerun_results_update` | Canonical K=16 rerun update (main comparison + ablation K=16 rows + FiLM baseline + Stanford) | ready |
| 021 | `sigma_baseline_backfill` | Backfill Patch2Self & MD-S2S σ=0.15/0.20 in sigma sweep table + Figure 2 | ready |
| 022 | `restormer_architecture_modifications` | Restormer Architecture Modifications (3D/2D variants, detailed comparison with original) | ready |
| 023 | `architecture_diagram_tikz` | Architecture Diagram (TikZ three-panel figure: Hybrid RGS pipeline, Restormer3D, 2D variant) | ready |

### Registry provenance (K=16 canonical baselines)

As of the **June 28, 2026** rerun (`exp_id=paper_final_k16_rerun`), authoritative metrics for jobs `*_rgs_final` come from:

- `tmp/paper_final_k16_rerun_20260628T042410Z/registry.jsonl` (lines 3–6)
- `tmp/paper_final_k16_rerun_20260628T042410Z/paper_tables/`

This supersedes May-2026 values in `tmp/paper_final_k16_out/registry.jsonl` **only** for those four canonical jobs. FiLM-conditioned runs (`*_film_conditioning`) remain in the old registry until a dedicated rerun. Use prompt **020** to sync the paper.

---

## General writing guidelines (shared across all prompts)

These conventions apply to every prompt unless explicitly overridden:

- **Language:** English, academic tone, concise.
- **Math notation:** LaTeX inline `$...$` and display `$$...$$`.
- **Tables:** LaTeX `tabular` or `booktabs` format.
- **Figures:** Reference by `\ref{fig:label}`; figure generation is separate.
- **Citations:** Use `\cite{key}` with BibTeX keys. When the key is unknown, use a descriptive placeholder like `\cite{perez2018film}`.
- **Acronyms:** Define on first use; afterwards use the acronym only.
- **Paper context:**
  - Method name: Hybrid RGS (Random Gradient Subset).
  - Two backbone architectures: DRCNet (3D CNN) and Restormer (3D Transformer).
  - Self-supervised denoising of Diffusion-Weighted MRI via J-invariance.
  - Datasets: D-Brain (synthetic Rician noise, has GT) and Stanford HARDI (real scanner noise, no GT).
