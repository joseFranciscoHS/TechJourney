# 023 — Architecture Diagram (TikZ)

## Context for the LLM

You are helping create a **comprehensive architecture figure** for a research paper on **Hybrid RGS** (Random Gradient Subset), a self-supervised framework for denoising Diffusion-Weighted MRI (DWI). The paper currently has no architecture diagram — only plot figures (K-sweep, sigma-robustness). This figure will visually explain:

1. The **Hybrid RGS training and inference pipeline** (data flow, sampling, masking, loss computation, Monte Carlo ensembling)
2. The **internal architecture of Restormer3D** (3-level encoder-decoder transformer with optional FiLM conditioning)
3. The **2D ablation variant** (lightweight residual CNN used in the 3D-vs-2D comparison)

The paper manuscript is [`paper/Sepulveda_dwmri_restormer.tex`](../../paper/Sepulveda_dwmri_restormer.tex). The figure will be inserted in Section 3 (Materials and Methods), likely after the Hybrid RGS formulation subsection or in the Architecture-Agnostic Instantiations subsection.

**Key notation from the paper** (use these symbols in the diagram for consistency):

- `G`: total number of diffusion-weighted volumes in the acquisition
- `t`: physical target diffusion volume index
- `C_t = (c_1, ..., c_{K-1})`: randomly sampled context gradient indices
- `K`: total number of input channels (context + target)
- `y_t`: noisy target volume
- `\tilde{y}_t`: Bernoulli-masked target volume
- `z_t`: concatenated input stack (context volumes + masked target)
- `m`: Bernoulli mask with probability `p=0.3`
- `\Omega_m`: set of masked voxel indices
- `f_\theta`: trained denoiser network (DRCNet or Restormer)
- `\hat{y}_t`: single prediction
- `N_c`: number of random context samples at inference
- `N_p`: number of spatial mask draws per context
- `\bar{y}_t`: final ensemble-averaged reconstruction
- Equations: masked loss (Eq. 4), Monte Carlo ensemble (Eq. 5)

**Design constraints:**

- **Single-column width** for `elsarticle` document class (~\linewidth)
- **Three panels** stacked vertically, labeled (a), (b), (c)
- **Compilable TikZ code** using `\usetikzlibrary{positioning,arrows.meta,fit,backgrounds,shapes.geometric}`
- Labels must be **legible** at print resolution (prefer stacked over cramped side-by-side layout)

---

## Files to attach

Attach these files when you send this prompt. Paths are relative to `DWMRI/`.

| File | Purpose |
|------|---------|
| `paper/Sepulveda_dwmri_restormer.tex` | Current manuscript (for notation: `y_t`, `C_t`, `z_t`, `\Omega_m`, `f_\theta`, `N_c`, `N_p`, Eq. 4, Eq. 5) |
| `src/restormer_hybrid_rgs/model.py` | `Restormer3D` implementation (3-level encoder-decoder, FiLM insertion points, channel dimensions) |
| `src/restormer_hybrid_rgs/model2d.py` | `Restormer2D` implementation (lightweight residual CNN, no attention) |
| `src/restormer_hybrid_rgs/config.yaml` | Hyperparameters: `dim=12`, `num_blocks=[1,2,2]`, `heads=[1,2,2]`, `K=16` default |
| `src/utils/film_layer.py` | FiLM conditioning layer (γ, β prediction from 4D acquisition vector) |

---

## Prompt

Generate a **single compilable TikZ figure** with three stacked panels (a), (b), (c) for the Hybrid RGS architecture. The figure should be self-contained (one `\begin{figure}...\end{figure}` environment) and insertable into the LaTeX manuscript.

---

### Panel (a): Hybrid RGS Training and Inference Pipeline

**Training flow (left-to-right, top row):**

1. **Input**: Box labeled "`G` DWI volumes" (representing the full 4D acquisition: `Y \in \mathbb{R}^{N_x \times N_y \times N_z \times G}`)

2. **Random sampling**: Arrow to a box showing:
   - "Sample target `t`"
   - "Sample context `C_t` (K-1 gradients)"
   - Use `\in` symbol: `C_t \subset \{1,...,G\}\setminus\{t\}`

3. **Masking**: Arrow to a split showing:
   - Top path: "`K-1` context volumes (unmasked)"
   - Bottom path: "Target volume `y_t`" → "Apply Bernoulli mask `m`" → "`\tilde{y}_t = (1-m) \odot y_t`"
   - Annotate mask: "`p=0.3`"

4. **Concatenation**: Both paths merge into "`K`-channel stack `z_t`" with a small annotation showing the target is at channel index `K-1`

5. **Network**: Arrow to a box labeled "`f_\theta`" with subtitle "(DRCNet or Restormer)"

6. **Prediction**: Arrow to "`\hat{y}_t`"

7. **Loss**: Arrow to a box showing masked MSE: 
   - Show Eq. 4 reference or write: `\mathcal{L} = \frac{1}{|\Omega_m|} \sum_{i \in \Omega_m} (\hat{y}_{t,i} - y_{t,i})^2`
   - Annotate: "Loss on masked voxels only"

**Inference flow (dashed box below training, or parallel):**

- Start from same "`G` DWI volumes" or reference training flow
- Show nested loop structure:
  - Outer loop: "`N_c` context samples" (draw `C_t^{(1)}, ..., C_t^{(N_c)}`)
  - Inner loop: "`N_p` mask draws" per context
- Multiple `f_\theta` boxes (or single box with annotation "× `N_c × N_p` forward passes")
- Average symbol: `\bar{y}_t = \frac{1}{N_c N_p} \sum_{a,b} f_\theta(z_t^{(a,b)})`
- Show Eq. 5 reference or write the formula

**Visual style:**
- Use solid arrows for training flow
- Use dashed arrows/box for inference branch
- Color suggestion: light blue for data boxes, light orange for network, light red for loss
- Annotate dimensions where helpful: e.g., "`z_t \in \mathbb{R}^{N_x \times N_y \times N_z \times K}`"

---

### Panel (b): Restormer3D Internal Architecture

Show a **U-Net-style encoder-decoder** with 3 levels (not 4):

**Encoder (left side, downward):**

1. **Patch embed**: Arrow from "`K` input channels" → box labeled "Patch Embed (Conv3d 3×3×3)" → "`dim=12` features"

2. **Encoder Level 1** (full resolution):
   - Box containing "N=1 Transformer Blocks"
   - Each block: "MDTA (heads=1) + GDFN (FFN 1.5×)"
   - Skip connection arrow pointing right to decoder side

3. **Downsample 1→2**: Arrow labeled "Downsample3D (strided Conv3d)" → "×2 spatial↓, channels→`2·dim`"

4. **Encoder Level 2** (half resolution):
   - Box: "N=2 Transformer Blocks (heads=2)"
   - Skip connection to decoder
   - **FiLM insertion point #1**: Small tag/box on the skip or output: "FiLM(cond)" with annotation "`[cos_x, cos_y, cos_z, b_norm]`" pointing to it

5. **Downsample 2→latent**: Arrow labeled "Downsample3D" → "×2 spatial↓, channels→`4·dim`"

**Bottleneck (center, lowest):**

- Box: "Latent: N=2 Transformer Blocks (heads=2)"
- Channel annotation: "`4·dim=48` features"
- **FiLM insertion point #2**: Tag "FiLM(cond)"

**Decoder (right side, upward):**

1. **Upsample latent→2**: Arrow labeled "Upsample3D (ConvTranspose3d)" → "×2 spatial↑, channels→`2·dim`"

2. **Concatenation**: Merge with encoder level 2 skip (show concatenation symbol or merged arrow)

3. **Channel reduction**: Small box "Conv3d 1×1×1: `4·dim` → `2·dim`"

4. **Decoder Level 2**:
   - Box: "N=2 Transformer Blocks (heads=2)"
   - **FiLM insertion point #3**: Tag "FiLM(cond)"

5. **Upsample 2→1**: Arrow labeled "Upsample3D" → "×2 spatial↑, channels→`dim`"

6. **Concatenation** with encoder level 1 skip

7. **Decoder Level 1**: Box "N=1 Transformer Blocks (heads=1)"

**Output head (right, exit):**

- Box: "Refinement: N=2 Transformer Blocks"
- Arrow to box: "Conv3d 3×3×3 + PReLU"
- Arrow to box: "Learned affine: `γ·x + β`"
- Final output: "`1` channel (denoised volume)"

**Visual style:**
- Rectangular boxes for processing stages
- Arrows with dimension labels (`dim`, `2·dim`, `4·dim`)
- Skip connections as curved arrows arcing from encoder to decoder
- FiLM tags as small colored boxes (e.g., green) with dashed lines to the 4D conditioning vector shown once (top or side)
- Annotate input: "`K` channels", output: "`1` channel"

---

### Panel (c): 2D Ablation Variant Inset

Small comparison box (can be a callout, side panel, or footer of panel (b)):

**Title**: "2D Variant (Table 3d_vs_2d ablation)"

**Side-by-side or stacked mini-diagrams:**

**Left box: Restormer3D (3-level)**
- Small schematic: Encoder (3 levels) → Bottleneck → Decoder (3 levels)
- "MDTA + GDFN transformer blocks"
- "0.178M parameters"
- "Attention, hierarchical, FiLM optional"

**Right box: Restormer2D (lightweight CNN)**
- Linear flow: "Embed Conv2d → N Residual Blocks → Output Conv2d"
- "Residual block = Conv2d-GELU-Conv2d + skip"
- "0.015M parameters (12× smaller)"
- "**Not a transformer** — no attention, no hierarchy"

**Annotation**: "Both use same Hybrid RGS objective (random sampling, masked loss)"

**Visual style:**
- Minimal, schematic (no need for full detail)
- Emphasize parameter count difference
- Highlight "not a transformer" text (bold or color)

---

## Formatting requirements

1. **TikZ libraries**: Must declare `\usetikzlibrary{positioning,arrows.meta,fit,backgrounds,shapes.geometric}` (include this in the output as a comment or note)

2. **Preamble additions**: Note that the paper preamble must have:
   ```latex
   \usepackage{tikz}
   \usetikzlibrary{positioning,arrows.meta,fit,backgrounds,shapes.geometric}
   ```

3. **Figure environment**:
   ```latex
   \begin{figure}[ht]
     \centering
     % TikZ code here
     \caption{...}
     \label{fig:architecture}
   \end{figure}
   ```

4. **Caption** (150-200 words): Explain that panel (a) shows the Hybrid RGS framework (training with random gradient subset sampling and Bernoulli target masking, inference with Monte Carlo ensemble), panel (b) details the Restormer3D architecture (3-level encoder-decoder with MDTA/GDFN transformer blocks, optional FiLM conditioning at three stages), and panel (c) contrasts the 3D transformer with the lightweight 2D residual CNN used in the 3D-vs-2D ablation (Table `\ref{tab:3d_vs_2d}`). Mention that K=16 is the default configuration, N_c=16 and N_p=12 for D-Brain (N_p=23 for Stanford).

5. **Label**: `\label{fig:architecture}`

6. **Size**: Target `\linewidth` for single-column `elsarticle`. If panels are too cramped, consider making two separate figures instead of three panels.

7. **Math mode**: Use `$...$` for inline math in TikZ node text (e.g., `node {$\hat{y}_t$}`)

8. **Color**: Optional light background colors for boxes (e.g., `fill=blue!10`, `fill=orange!10`) — use sparingly for clarity, not decoration

9. **Arrows**: Use `arrows.meta` library for clean arrow heads (e.g., `-Stealth`)

10. **Legibility**: Font size in nodes should be `\small` or `\footnotesize` if needed; all labels must be readable at print size

---

## Expected output

Return:

1. **Complete TikZ figure code**: One `\begin{figure}...\end{figure}` block with all three panels (a), (b), (c) as TikZ `\begin{tikzpicture}...\end{tikzpicture}` environments or sub-environments

2. **Preamble note**: Exact lines to add to the paper preamble (`\usepackage{tikz}`, `\usetikzlibrary{...}`)

3. **Caption and label**: Full caption text (~150-200 words) referencing the paper's equations, tables, and hyperparameters

4. **Insertion point**: Brief instruction on where in the paper this figure should go (e.g., "Insert after Section 3.2 (Hybrid RGS Formulation) or within Section 3.4 (Architecture-Agnostic Instantiations)")

**Important**: The TikZ code should compile without errors when inserted into the paper. Test with a standalone `\documentclass{article} \usepackage{tikz} \usetikzlibrary{...} \begin{document} ... \end{document}` wrapper if unsure.

**Fallback**: If three panels in one figure become too complex, suggest splitting into two figures: (1) Hybrid RGS pipeline (panel a) as `fig:hybrid_rgs_pipeline`, and (2) Architecture details (panels b+c) as `fig:restormer_architecture`. Provide both options.
