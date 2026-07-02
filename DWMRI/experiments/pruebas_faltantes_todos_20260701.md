# Pruebas Faltantes y TODOs Pendientes — 1 julio 2026

Este documento es el sucesor de [`pruebas_faltantes_todos_20260628.md`](pruebas_faltantes_todos_20260628.md). Registra el estado actual tras la auditoría de correcciones 015–019 (28 jun – 1 jul 2026) y las acciones que siguen pendientes antes de submission.

**Referencia de auditoría:** `tmp/paper_final_k16_out/writing/015_diffusivity_units_correction.md` … `019_stanford_bvalue_correction.md`

---

## Completado desde 20260628

| # | Tema | Estado | Evidencia |
|---|------|--------|-----------|
| 1 | Unidades difusividad (015) | **Aplicado en paper** | Footnotes `$^\dagger$` en tablas; TODOs eliminados; Methods L144 correcto |
| 2 | Métricas FiLM (016) | **Aplicado en paper + soporte** | `tab:film_ablation` con ROI -0.64/-0.53 dB; narrativa corregida; aux files actualizados 1-jul |
| 3 | Conflicto registry (017) | **Documentado y aplicado** | Valores canónicos en paper; párrafo provenance L408–409; sin PSNR-ROI corruptos (9.538) |
| 7 | Stanford b-value (019) | **Aplicado en paper + configs** | Paper dice `b=2000` en todas las menciones; `config.yaml` corregidos 1-jul |

**Archivos del paper verificados:** `paper/Sepulveda_dwmri_restormer.tex`

---

## Prioridad alta — pendiente (figuras cualitativas)

Estos items **no se resuelven con ediciones de texto** y requieren generación de visualizaciones o decisión explícita de remover figuras.

### 4. Figura cualitativa — Main Comparison D-Brain

**TODO en paper (línea 453):**
> Replace this planned qualitative comparison with final image panels or remove the figure before submission. The final figure should include noisy input, MP-PCA, Patch2Self, MD-S2S, DRCNet-Hybrid-RGS, Restormer-Hybrid-RGS, reference image when available, and error maps.

**Estado:** No verificado si paneles existen en `tmp/paper_final_k16_out/`.

**Acción requerida:**
1. Buscar outputs en `tmp/paper_final_k16_out/drcnet_hybrid_rgs/images/dbrain/`, `mppca/`, `patch2self/`, etc.
2. Si existen: compilar figura multi-método (6–8 columnas).
3. Si NO existen: ejecutar scripts de visualización **o** remover el `\TODO` y la referencia a figura.

**Decisión necesaria:** ¿Generar imágenes o remover la figura?

**Esfuerzo:** 1–2 h (generar) | 15 min (remover TODO + texto conservador)

**Archivos afectados:**
- `paper/Sepulveda_dwmri_restormer.tex` línea 453
- `tmp/paper_final_k16_out/figures/` (nuevo, si se genera)

---

### 5. Figura Stanford HARDI FA-map

**TODO en paper (línea 495):**
> Replace the planned Stanford HARDI FA-map comparison with final image panels or remove the figure before submission. The final figure should show a representative axial slice crossing major white-matter tracts for noisy input, Patch2Self, MD-S2S, DRCNet-Hybrid-RGS, and Restormer-Hybrid-RGS.

**Estado:** No hay archivos `stanford_fa*` en el repo. Mapas FA/MD en `tmp/.../stanford_film_conditioning/` no verificados.

**Acción requerida:**
1. Verificar si `tmp/paper_final_k16_out/drcnet_hybrid_rgs/metrics/stanford_film_conditioning/` contiene `fa_map.npy`.
2. Si existen: generar slice axial (corpus callosum / corona radiata).
3. Si NO: post-procesamiento DTI o remover figura.

**Decisión necesaria:** ¿Generar mapas FA/MD de Stanford o limitar claims a feasibility sin figura?

**Esfuerzo:** 1–2 h | 15 min (remover)

**Archivos afectados:**
- `paper/Sepulveda_dwmri_restormer.tex` línea 495
- §4.5 Stanford generalization

---

### 6. Evidencia cualitativa Stanford (FA/residual/ROI variance)

**TODO en paper (línea 465):**
> Add visual Stanford FA/MD panels, residual maps, and/or homogeneous-ROI variance summaries. Without those panels, keep Stanford claims limited to feasibility and scalability.

**Estado:** Texto cualitativo presente (L461–463) pero **sin figura referenciada**. Sin ground truth en Stanford.

**Opciones:**

| Opción | Contenido | Esfuerzo |
|--------|-----------|----------|
| **A (mínimo viable)** | 1 panel FA (noisy, DRCNet, Restormer) + mapas residuales | ~1.5 h |
| **B** | Variance en ROI homogéneo (centrum semiovale) | ~1 h adicional |
| **C** | Remover TODOs; mantener claims de feasibility | 15 min |

**Recomendación:** Opción A si mapas `.npy` existen; Opción C si deadline apretado.

**Archivos afectados:**
- `paper/Sepulveda_dwmri_restormer.tex` líneas 465, 495
- `tmp/paper_final_k16_out/figures/stanford_fa_comparison.png` (nuevo, si Opción A)

---

## Prioridad media — completado 1 jul 2026

### Corrección archivos de soporte FiLM (016)

**Problema:** `orientation_conditioning_metrics_summary.md` tenía PSNR-ROI inflados (30.40, 27.94) copiados incorrectamente.

**Valores autoritativos** (`registry.jsonl` líneas 12–13 baseline, 131–134 FiLM):

| Métrica | DRCNet baseline | DRCNet FiLM | Restormer baseline | Restormer FiLM |
|---------|-----------------|-------------|-------------------|----------------|
| PSNR (full) | 23.93 | 25.40 | 22.83 | 23.60 |
| PSNR-ROI | **26.88** | **26.24** | **23.22** | **22.69** |
| Delta ROI | — | **-0.64 dB** | — | **-0.53 dB** |
| FA-MAE | 0.2587 | 0.2405 | 0.2378 | 0.2603 |
| MD-MAE | 3449 | 3452 | 3455 | 3423 |

**Hallazgo crítico:** FiLM **mejora PSNR full-volume** (+1.47 dB DRCNet, +0.77 dB Restormer) pero **degrada PSNR-ROI** (-0.64 dB DRCNet, -0.53 dB Restormer).

**Archivos corregidos:**
- `tmp/paper_final_k16_out/orientation_conditioning_metrics_summary.md`
- `tmp/paper_final_k16_out/writing/001_orientation_conditioning.md`
- `tmp/paper_final_k16_out/writing/README_corrections.md` (progress tracking)

---

## Prioridad baja — completado 1 jul 2026

### Stanford b-value en configs

**Problema:** `config.yaml` decía `bvalue: 1000`; DIPY usa **b=2000 s/mm²** (Rokem et al., 2015).

**Nota:** El código carga bvals desde DIPY; el config solo afecta path naming y logging. DTI fitting ya era correcto.

**Archivos corregidos:**
- `src/drcnet_hybrid_rgs/config.yaml` línea 181: `bvalue: 2000`
- `src/restormer_hybrid_rgs/config.yaml` línea 170: `bvalue: 2000`

---

## Resumen de acciones

| # | TODO | Prioridad | Estado | Esfuerzo restante |
|---|------|-----------|--------|-------------------|
| 1 | Unidades difusividad | Crítica | ✅ Completado | — |
| 2 | Métricas FiLM (paper + soporte) | Crítica | ✅ Completado | — |
| 3 | Registry canónico | Info | ✅ Documentado | — |
| 4 | Figura D-Brain main comparison | **Alta** | ⏳ Pendiente | 1–2 h o 15 min |
| 5 | Figura Stanford FA-map | **Alta** | ⏳ Pendiente | 1–2 h o 15 min |
| 6 | Evidencia Stanford (FA/residual) | **Alta** | ⏳ Pendiente | ~1.5 h o 15 min |
| 7 | Stanford b-value | Crítica | ✅ Completado | — |

**Próximo paso recomendado:** Decidir sobre items #4–#6 (generar figuras vs. remover TODOs y mantener texto conservador).

---

## Comandos de verificación

```bash
# TODOs restantes en paper (esperado: 3 hasta resolver figuras)
grep -n '\\TODO{' TechJourney/DWMRI/paper/Sepulveda_dwmri_restormer.tex

# b-value incorrecto en paper (debe estar vacío)
grep -n 'b=1000' TechJourney/DWMRI/paper/Sepulveda_dwmri_restormer.tex

# Valores FiLM incorrectos en soporte (debe estar vacío)
grep -n '30\.40\|27\.94\|28\.48\|1\.92 dB\|1\.38 dB' \
  TechJourney/DWMRI/tmp/paper_final_k16_out/orientation_conditioning_metrics_summary.md

# Configs corregidos (debe estar vacío)
grep -n 'bvalue.*1000' \
  TechJourney/DWMRI/src/drcnet_hybrid_rgs/config.yaml \
  TechJourney/DWMRI/src/restormer_hybrid_rgs/config.yaml

# Valores canónicos baseline
grep -A 5 'drcnet_dbrain_rgs_final\|restormer_dbrain_rgs_final' \
  TechJourney/DWMRI/tmp/paper_final_k16_out/registry.jsonl | grep 'psnr_roi'
```

---

## Notas finales

- Valores **autoritativos** en `registry.jsonl` líneas 12–13 (baselines), 131–134 (FiLM).
- **No usar** `paper_metrics_summary.csv` para filas baseline σ=0.1 K=16 (paths corruptos por `inference_time_grid`).
- Los **writing prompts 008–014** siguen válidos; prompt `001_orientation_conditioning.md` actualizado con narrativa ROI degradation.
- Stanford PSNR en registry FiLM es **noisy-vs-noisy** (sin GT); no comparar con PSNR D-Brain.

**Contacto:** Francisco / Mariano

**Última actualización:** 2026-07-01
