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
