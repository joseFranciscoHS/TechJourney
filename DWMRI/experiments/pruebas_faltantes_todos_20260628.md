# Pruebas Faltantes y TODOs Pendientes — 28 junio 2026

Este documento registra los TODOs del paper que **no se pueden resolver únicamente con el código/datos existentes** y requieren decisiones explícitas, nuevas corridas experimentales, o generación de visualizaciones.

---

## 1. Unidades de Difusividad (líneas 190, 397)

**Problema:** Las tablas reportan MD-MAE, AD-MAE, RD-MAE con valores ~3400-8500 para D-Brain. Estos valores NO son mm²/s físicos.

**Hallazgos:**

1. **Código:** `src/paper_eval/dti_metrics.py` aplica `invert_normalization` antes de tensor fitting, lo cual devuelve los DWIs a la escala de intensidad original del phantom.

2. **Escala de D-Brain:** Es un phantom digital simulado con intensidades arbitrarias. El `dti_sanity_gt.md_range` del registry muestra `max ≈ 879,827`, órdenes de magnitud arriba de valores físicos cerebrales típicos (~0.7-1.0 × 10⁻³ mm²/s).

3. **Conclusión:** Los valores reportados son **MAE en unidades del phantom simulado**, NO en mm²/s. Solo son comparables entre métodos en el mismo pipeline.

4. **Stanford HARDI (valores ~10⁻⁵ a 10⁻⁶):** SÍ están en mm²/s físicos (DIPY calcula difusividades reales a partir de datos in-vivo).

**Opciones para el paper:**

### Opción A (recomendada para deadline): Reportar "como está" con disclaimer

- Mantener valores actuales en tablas
- Agregar nota al pie: "MD-MAE, AD-MAE, and RD-MAE are reported in arbitrary units derived from the D-Brain phantom intensity scale and should be interpreted as relative comparisons across methods. Stanford HARDI diffusivity errors are in physical mm²/s units."
- Evitar etiquetas como "×10⁻⁶ mm²/s" o "μm²/s" en columnas D-Brain

### Opción B (requiere re-run completo): Escalar a mm²/s físicos

- Determinar factor de escala entre intensidades D-Brain y difusividades físicas
- Re-computar DTI metrics en todas las corridas con escala correcta
- **Costo:** Varias horas de compute + riesgo de inconsistencias

**Decisión necesaria:** ¿Cuál opción prefieres?

**Archivos afectados:**
- `paper/Sepulveda_dwmri_restormer.tex` líneas 190 (Ablation table footnote), 397 (Main table footnote)
- Todas las tablas con MD-MAE / AD-MAE / RD-MAE

---

## 2. Discrepancia FiLM en `orientation_conditioning_metrics_summary.md`

**Problema resuelto parcialmente — requiere corrección de documento.**

Los valores autoritativos de FiLM (de `registry.jsonl`, corroborados por `paper_metrics_summary.csv`) son:

### D-Brain, σ=0.1, K=16

| Metric | DRCNet FiLM | Restormer FiLM |
|--------|-------------|----------------|
| PSNR | **25.403** dB | **23.603** dB |
| SSIM | **0.4552** | **0.4380** |
| FA-MAE | **0.2405** | **0.2603** |
| MD-MAE | **3452.19** | **3423.42** |

**Sin embargo**, `orientation_conditioning_metrics_summary.md` tiene:
- PSNR-ROI valores **inflados/incorrectos**: 30.40 (DRCNet), 27.94 (Restormer), 45.06 (Stanford DRCNet), 32.56 (Stanford Restormer)
- Baseline ROI PSNR = 28.48 (parece copiado de MPPCA, no del baseline DRCNet real que es 26.88)

**Los valores correctos de ROI PSNR** (de registry):
- DRCNet FiLM: **26.244**
- Restormer FiLM: **22.694**
- DRCNet baseline: **26.882**
- Restormer baseline: **23.220**

**Acción:** Actualizar `orientation_conditioning_metrics_summary.md` y cualquier borrador de tabla FiLM con los valores del registry.

**Archivos afectados:**
- `tmp/paper_final_k16_out/orientation_conditioning_metrics_summary.md`
- Prompt `001_orientation_conditioning.md` (si usa summary como fuente)

---

## 3. Conflicto Registry σ=0.10 K=16 (línea 282)

**Problema resuelto — documentado para referencia.**

El `registry.jsonl` tiene múltiples corridas con σ=0.1, K=16:

| Job ID | Recipe | Registry line | PSNR (full) | PSNR-ROI | Notas |
|--------|--------|---------------|-------------|----------|-------|
| `drcnet_dbrain_rgs_final` | `drcnet_main` | **12** | **23.927** | **26.882** | **Canonical baseline** (16 mayo) |
| `drcnet_dbrain_k16_ablation` | `k_sweep` | 26 | 23.878 | 26.909 | K-sweep training separado |
| `drcnet_dbrain_inference_time_ctx48_pred20` | `inference_time_grid` | 121 | **5.372** | **9.538** | **Sobrescribió métricas path** |

**Causa:** Los jobs `inference_time_grid` (ctx=48, pred=20) usaron los mismos `metrics_dir` paths que los baselines, sobrescribiendo `metrics.json` con valores de una configuración diferente.

**Valores autoritativos (del registry, líneas 12-13):**

- **DRCNet baseline:** PSNR 23.927, SSIM 0.4404, FA-MAE 0.2587, MD-MAE 3449.38, **PSNR-ROI 26.882**
- **Restormer baseline:** PSNR 22.835, SSIM 0.4271, FA-MAE 0.2378, MD-MAE 3454.58, **PSNR-ROI 23.220**

**Consecuencia para paper:** `paper_metrics_summary.csv` contiene valores corruptos en las rutas baseline. Para escribir tablas del paper, **usar valores del `registry.jsonl`** directamente, NO de `paper_metrics_summary.csv` baseline paths.

**No requiere acción adicional** — solo documentación para escritura.

---

## 4. Figuras Cualitativas — Main Comparison (líneas 399)

**TODO original:**
> Replace this planned qualitative comparison with final image panels or remove the figure before submission. The final figure should include noisy input, MP-PCA, Patch2Self, MD-S2S, DRCNet-Hybrid-RGS, Restormer-Hybrid-RGS, reference image when available, and error maps.

**Estado:** No se ha verificado si estas visualizaciones existen en `tmp/paper_final_k16_out/`.

**Acción requerida:**

1. **Verificar existencia:** Buscar en directorios de output:
   - `tmp/paper_final_k16_out/drcnet_hybrid_rgs/images/dbrain/`
   - `tmp/paper_final_k16_out/restormer_hybrid_rgs/images/dbrain/`
   - `tmp/paper_final_k16_out/mppca/`, `tmp/paper_final_k16_out/patch2self/`, etc.

2. **Si existen:** Compilar paneles en figura multi-método (6-8 columnas: noisy, MPPCA, P2S, MDS2S, DRCNet-RGS, Restormer-RGS, GT, error map).

3. **Si NO existen:** Ejecutar scripts de visualización o decidir remover la figura.

**Decisión necesaria:** ¿Generar las imágenes o remover la figura?

**Archivos afectados:**
- Texto del paper referenciando la figura (si se remueve)
- `tmp/paper_final_k16_out/figures/` (nuevo directorio para figuras compiladas)

---

## 5. Figuras Cualitativas — Stanford HARDI FA/MD (línea 441)

**TODO original:**
> Replace the planned Stanford HARDI FA-map comparison with final image panels or remove the figure before submission. The final figure should show a representative axial slice crossing major white-matter tracts for noisy input, Patch2Self, MD-S2S, DRCNet-Hybrid-RGS, and Restormer-Hybrid-RGS.

**Estado:** No verificado si mapas FA/MD de Stanford están generados.

**Acción requerida:**

1. Verificar si `tmp/paper_final_k16_out/drcnet_hybrid_rgs/metrics/stanford_film_conditioning/` contiene FA/MD `.npy` maps.

2. Si existen: Generar visualización de slice axial representativo con corpus callosum / corona radiata visibles.

3. Si NO: Ejecutar post-procesamiento DTI en Stanford outputs o remover figura.

**Decisión necesaria:** ¿Generar mapas FA/MD de Stanford o limitar claims a feasibility sin figura?

**Archivos afectados:**
- §4.5 Stanford generalization (texto + figura ref)

---

## 6. Stanford Qualitative Evidence (línea 411)

**TODO original:**
> Add visual Stanford FA/MD panels, residual maps, and/or homogeneous-ROI variance summaries. Without those panels, keep Stanford claims limited to feasibility and scalability.

**Estado:** Sin ground truth en Stanford, claims cuantitativos son limitados. Se necesita evidencia cualitativa.

**Opciones:**

### A. Mapas FA/MD (overlap con TODO #5)

Mostrar que las estructuras anatómicas (corpus callosum, cápsulas internas) tienen valores FA plausibles (~0.6-0.8 en WM, ~0.1-0.2 en GM).

### B. Residual maps

Calcular `residual = noisy_input - denoised` y verificar que se parece a ruido (no contiene bordes anatómicos).

### C. Variance en ROIs homogéneas

Medir `std(FA)` en regiones homogéneas (ej. centrum semiovale) para comparar métodos. Menor varianza = mejor denoising en zonas uniformes.

**Acción mínima viable:** Agregar 1 panel de FA map comparativo (Stanford noisy vs DRCNet-RGS vs Restormer-RGS) en §4.5.

**Decisión necesaria:** ¿Qué nivel de evidencia Stanford incluir?

---

## 7. Stanford b-value Discrepancia (config vs DIPY) — RESUELTO

**Problema identificado:**

- **Config:** `src/drcnet_hybrid_rgs/config.yaml` Stanford section dice `bvalue: 1000`
- **DIPY documentation:** Stanford HARDI dataset oficial usa **b=2000 s/mm²** con 160 direcciones
- **Usuario confirmó:** El valor correcto es **b=2000**

**Análisis de código:**

```python
# src/utils/data.py
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames("stanford_hardi")
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)
```

El código carga los bvals **directos de DIPY**, ignorando el `config.yaml bvalue: 1000`. El valor en config solo se usa para path naming, NO para tensor fitting. Entonces:

- **DTI fitting es correcto** (usa b=2000 de DIPY)
- **Texto del paper debe decir b=2000** (no b=1000)
- **Config tiene error de documentación** (debería decir 2000 para consistencia)

**Acción completada:**

Documentado en `pruebas_faltantes_todos_20260628.md`. El writing prompt 009 (dataset/preprocessing) **ya especifica correctamente b=2000** para Stanford HARDI.

**Archivos a corregir (opcional, no crítico para paper):**
- `src/drcnet_hybrid_rgs/config.yaml` línea 181: cambiar `bvalue: 1000` → `bvalue: 2000` (o agregar comment)
- `src/restormer_hybrid_rgs/config.yaml` línea 170: mismo cambio

**Para el paper:** Usar **b=2000, 150 directions** en toda mención de Stanford HARDI (como está en prompt 009).

---

## Resumen de Acciones Pendientes

| # | TODO | Tipo | Esfuerzo estimado |
|---|------|------|-------------------|
| 1 | Decidir convención unidades difusividad | Decisión editorial | 10 min |
| 2 | Corregir valores FiLM en orientation summary | Corrección doc | 5 min |
| 3 | Usar registry (no CSV) para baselines | Documentado | — |
| 4 | Generar o remover figura main comparison | Viz o decisión | 1-2 hrs o remove |
| 5 | Generar o remover figura Stanford FA/MD | Viz o decisión | 1-2 hrs o remove |
| 6 | Evidencia cualitativa Stanford (mínima) | Viz | 30 min - 1 hr |
| 7 | Verificar/corregir Stanford b-value | Verificación + corrección texto | 15 min |

**Prioridad para deadline paper:**

1. **Crítico:** TODO #1 (unidades), #2 (FiLM values), #7 (b-value texto)
2. **Recomendado:** TODO #6 (1 panel FA Stanford)
3. **Opcional si tiempo permite:** TODO #4, #5 (figuras completas)

---

## Notas Finales

- Todos los **writing prompts 008-014** están completos y listos para uso.
- Valores **autoritativos** están en `registry.jsonl` líneas 12-13 (baselines), 131-134 (FiLM).
- El **inference method** (línea 122) fue corregido en prompt 011 — full-volume averaging, NO masked-voxel-only.
- **Diffusivity units** necesitan decisión explícita antes de finalizar tablas.

**Contacto para resolver estos TODOs:** Francisco / Mariano
