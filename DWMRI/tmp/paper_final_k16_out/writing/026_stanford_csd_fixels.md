# 026 — Stanford CSD fixel corroboration (LNNN Figs. 10–11 style)

## Context for the LLM

You are helping revise a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using **Hybrid RGS**. The tutor request (Aguayo-González et al., Front. Neuroinform. 2024) asks us to corroborate how **our denoising** affects **CSD fixel outputs** of the kind shown in their Figs. 10–11 (crossing-fiber continuity / interruptions; gyral-blade fanning).

We do **not** reimplement LNNN / AxonNet. Instead we:

1. Export physical-intensity Stanford DWI crops for eight arms (noisy, MP-PCA, P2S, DRCNet-3D, Restormer-3D, Restormer-3D large, Res-CNN-2D, Restormer-2D).
2. Fit a **frozen** DIPY CSD + peaks protocol (`npeaks=3`, `relative_peak_threshold=0.2`, cite Aguayo-González et al.).
3. Produce qualitative glyph figures + no-GT proxy metrics (vs noisy and vs MP-PCA).

**Important language constraint:** there is **no fixel ground truth** on Stanford. All claims must stay **qualitative / relative** (e.g. “closer to MP-PCA”, “fewer primary-peak flips in the crossing ROI”). Do **not** claim “improved microstructure” without GT.

Study root: `tmp/paper_final_k16_stanford_fixels/`
Orchestration: `experiments/rerun_k16_stanford_fixel_arms.sh`

---

## Files to attach

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Manuscript to patch |
| `src/paper_eval/csd_fixels.py` | Frozen CSD+peaks + proxy metrics |
| `src/paper_eval/plot_fixels.py` | Glyph figure builder |
| `src/paper_eval/stanford_fixel_rois.yaml` | Frozen ROI boxes (after scout) |
| `tmp/paper_final_k16_stanford_fixels/proxy/proxy_metrics.csv` | Proxy table (after run) |
| `tmp/paper_final_k16_stanford_fixels/figures/fixel_glyphs.png` | Main figure (after run) |

---

## Prompt

You are revising `paper/Sepulveda_dwmri_restormer.tex`. Perform the following tasks. Produce LaTeX snippets that can be inserted or substituted directly.

### Task 1 — Methods: CSD fixel evaluation protocol

Add a short Methods subsection (or paragraph under evaluation) that:

- States that we evaluate denoising impact on single-shell CSD fixels on the Stanford HARDI crop used for training (`take_x/y/z` Stanford profile), citing Aguayo-González et al. for the peak protocol.
- Specifies DIPY `auto_response_ssst` → `ConstrainedSphericalDeconvModel` → `peaks_from_model` with `npeaks=3`, `relative_peak_threshold=0.2`, `min_separation_angle=25°`, SH order 8.
- Lists the eight input arms and notes that all volumes are denormalized to physical intensities with matching b0s/bvals/bvecs before CSD.
- Mentions the proxy metrics (primary-peak angular deviation vs noisy and vs MP-PCA; multi-peak fraction; crossing-ROI discontinuity rate; gyral fan spread) and that they are **reference-free**.

### Task 2 — Results: qualitative figure + small metrics table

- Insert a multi-column glyph figure (arms × ROIs) using `tmp/paper_final_k16_stanford_fixels/figures/fixel_glyphs.png` (or the final paper figures path once copied).
- Caption must emphasize qualitative comparison and that ROIs are in the training crop FOV.
- Add a small table of proxy metrics from `proxy/proxy_metrics.csv` once available; leave `TODO` placeholders if the CSV is not yet filled.

### Task 3 — Claims discipline

Audit any new sentences so they do **not** assert absolute microstructural improvement. Prefer formulations like “reduces primary-peak discontinuity relative to the noisy input in the crossing ROI” or “aligns more closely with the MP-PCA reference”.

---

## Placeholder LaTeX (fill numbers after the study run)

```latex
\subsection{Stanford CSD fixel corroboration}
\label{sec:stanford-csd-fixels}

To corroborate how Hybrid RGS denoising affects constrained spherical
deconvolution (CSD) fixel outputs of the kind shown by
Aguayo-Gonz\'alez et~al.~\cite{aguayo2024lnnn}, we recompute single-shell
CSD peaks on the Stanford HARDI training crop under a frozen DIPY protocol
(\texttt{auto\_response\_ssst} $\rightarrow$
\texttt{ConstrainedSphericalDeconvModel} $\rightarrow$
\texttt{peaks\_from\_model}; at most three peaks,
relative peak threshold $0.2$, minimum separation $25^\circ$).
Only the input DWI varies across arms (noisy, MP-PCA, Patch2Self,
DRCNet-3D, Restormer-3D, Restormer-3D-large, Res-CNN-2D, Restormer-2D);
all volumes are denormalized to physical intensities with matching $b_0$s
before fitting. Because Stanford has no fixel ground truth, we report
qualitative glyph panels (Fig.~\ref{fig:stanford-csd-fixels}) and
reference-free proxies versus the noisy and MP-PCA arms
(Table~\ref{tab:stanford-csd-proxies}).

\begin{figure*}[t]
  \centering
  % \includegraphics[width=\textwidth]{figures/stanford_csd_fixels.png}
  \caption{CSD peak glyphs on the Stanford training crop for eight
  denoising arms (columns) and scouted ROIs (rows). Peak slots are colored
  by rank (primary/secondary/tertiary). Qualitative comparison only;
  no fixel ground truth is available.}
  \label{fig:stanford-csd-fixels}
\end{figure*}

\begin{table}[t]
  \centering
  \caption{Proxy fixel metrics on Stanford (vs noisy / vs MP-PCA).
  Values are placeholders until
  \texttt{tmp/paper\_final\_k16\_stanford\_fixels/proxy/proxy\_metrics.csv}
  is filled.}
  \label{tab:stanford-csd-proxies}
  \begin{tabular}{lcccc}
    \toprule
    Arm & $\Delta\theta$ vs noisy ($^\circ$) & $\Delta\theta$ vs MP-PCA ($^\circ$)
      & multi-peak frac. & disc.\ rate \\
    \midrule
    noisy & -- & TODO & TODO & TODO \\
    MP-PCA & TODO & -- & TODO & TODO \\
    P2S & TODO & TODO & TODO & TODO \\
    DRCNet-3D & TODO & TODO & TODO & TODO \\
    Restormer-3D & TODO & TODO & TODO & TODO \\
    Restormer-3D-large & TODO & TODO & TODO & TODO \\
    Res-CNN-2D & TODO & TODO & TODO & TODO \\
    Restormer-2D & TODO & TODO & TODO & TODO \\
    \bottomrule
  \end{tabular}
\end{table}
```
