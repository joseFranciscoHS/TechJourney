# 008 — Author Metadata

## Context for the LLM

You are helping finalize a research paper on **self-supervised denoising of Diffusion-Weighted MRI (DWI)** using **Hybrid RGS** (Random Gradient Subset). The manuscript is ready except for author metadata placeholders.

## Files to attach

None needed — all information is provided below.

## Prompt

Replace the placeholder author metadata in the paper with the following final information:

### Authors

1. **Francisco Hernandez Sepulveda** (first author)
   - Email: jose.sepulveda@cimat.mx
   - Affiliation: CIMAT

2. **Mariano Rivera Meraz** (second author, corresponding author)
   - Email: mrivera@cimat.mx
   - Affiliation: CIMAT

### Affiliation

- **CIMAT**: Centro de Investigacion en Matematicas, Guanajuato, Mexico

### LaTeX Format

Use the following `elsarticle` format:

```latex
\author[inst1]{Francisco Hernandez Sepulveda}
\ead{jose.sepulveda@cimat.mx}

\author[inst1]{Mariano Rivera Meraz\corref{cor1}}
\ead{mrivera@cimat.mx}
\cortext[cor1]{Corresponding author.}

\address[inst1]{Centro de Investigacion en Matematicas, Guanajuato, Mexico}
```

## Expected output

The LaTeX author block with correct names, emails, and affiliation, ready to paste into lines 20-29 of the manuscript.
