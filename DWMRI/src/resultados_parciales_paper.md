# Resultados parciales — paper DWMRI Hybrid RGS (vivo)

Documento vivo para ir consolidando números antes de redactar el paper. **No** sustituye al plan de experimentos en [`plan_para_escribir_el_paper.md`](plan_para_escribir_el_paper.md); solo refleja lo que ya salió de la corrida.

**Convenciones:** `⚠` = métrica registrada pero con alerta abierta (no usar en el paper sin revisar). `—pend—` = job pendiente o en curso. `—n/a—` = no aplica. `—` = no disponible en el artefacto actual.

---

## Estado de la corrida (snapshot)

| Campo | Valor |
| --- | --- |
| **Última actualización de este doc** | 2026-05-14 |
| **exp_id** | `paper_final_k16` |
| **Comando típico** | Ver [`experiments/checklist_pruebas_paper.md`](../experiments/checklist_pruebas_paper.md) §0 (`driver.py` + `paper_manifest_final.yaml`) |
| **Registry** | [`tmp/paper_final_k16_out/registry.jsonl`](../tmp/paper_final_k16_out/registry.jsonl) |
| **Driver state** | [`tmp/paper_final_k16_out/driver_state.json`](../tmp/paper_final_k16_out/driver_state.json) |
| **Driver events** | [`tmp/paper_final_k16_out/driver_events.jsonl`](../tmp/paper_final_k16_out/driver_events.jsonl) |
| **Job en curso (driver_state)** | `restormer_dbrain_seq_k16_ablation` — `running` (intento 2 al momento del snapshot) |

### Resumen de jobs (`driver_state.json`)

Totales sobre **143** claves en estado (= **138** jobs del manifiesto + **5** entradas huérfanas `p2s_dbrain_sklearn_reference_*` sin manifiesto; ver eventos `manifest_mismatch` en `driver_events.jsonl`).

| status | count |
| --- | ---: |
| succeeded | 14 |
| failed | 1 |
| running | 1 |
| pending | 127 |

El job `failed` corresponde a `p2s_dbrain_sklearn_reference_final` (SIGKILL / OOM); omitido por diseño en el manifiesto actual.

### Progreso por bloque (solo jobs listados en `paper_manifest_final.yaml`)

| Bloque | succeeded | running | pending |
| --- | ---: | ---: | ---: |
| Tabla principal + baselines (`*_final` núcleo) | 12 | 0 | 4 |
| Ablación sampling / progressive (`seq_k16`, `progressive_off`) | 2 | 1 | 1 |
| Ablación K (`*_k*_ablation`) | 0 | 0 | 22 |
| Ablación 2D vs 3D (`*_2d_rgs_k16_ablation`) | 0 | 0 | 2 |
| Ablación `mask_p` | 0 | 0 | 20 |
| Ablación `n_context` | 0 | 0 | 24 |
| Ablación `n_preds` | 0 | 0 | 12 |
| Sensibilidad σ (`*_sigma_*`) | 0 | 0 | 24 |
| Inferencia / gap / seeds / tiempos / resumen | 0 | 0 | 16 |

Los **4** pendientes del núcleo son: `drcnet_dbrain_arch_compare_parity_final`, `restormer_dbrain_arch_compare_parity_final`, `summarize_registry_final`, `collect_paper_artifacts_final`.

---

## Alertas / flags abiertos

Revisar **antes** de copiar cifras al manuscrito.

- **Alerta A — DRCNet-RGS D-Brain por debajo de baselines clásicos**  
  PSNR_full ≈ **13.33** para `drcnet_dbrain_rgs_final` está ~9 dB por debajo de MP-PCA (~22.59) y ~7 dB por debajo de DRCNet-sequential (~20.78). La cota supervisada DRCNet (~13.32) es casi idéntica al self-sup. Hipótesis: rescaling en reconstrucción RGS, checkpoint del último stage progressive, o `n_context`/`n_preds` insuficientes frente al entrenamiento.

- **Alerta B — SSIM negativo en Restormer (D-Brain)**  
  `ssim ≈ -0.071` (RGS) y `≈ -0.069` (supervisado). Suele indicar `data_range` mal fijado en `skimage.metrics.structural_similarity` o rango de tensores inconsistente frente al GT.

- **Alerta C — Progressive OFF >> progressive ON (DRCNet)**  
  `drcnet_dbrain_progressive_off_ablation`: PSNR_full ≈ **23.83** vs `drcnet_dbrain_rgs_final` (progressive ON, 360 épocas): **13.33**. Hay que verificar stages finales (patch size, batch) y que la corrida “principal” no esté en un estado degenerado.

- **Alerta D — DRCNet vs Restormer en Stanford (PSNR respecto al input)**  
  `drcnet_stanford_rgs_final`: PSNR_full ≈ **3.03** vs `restormer_stanford_rgs_final`: **26.99** (mismo `n_context`/`n_preds` en registry). Sin GT, el PSNR mide coherencia con el input ruidoso; la magnitud sugiere revisar si DRCNet altera demasiado la señal o si hay asimetría de pipeline.

- **Alerta E — Rutas de métricas P2S / MD-S2S D-Brain (resuelta para localización)**  
  Las métricas **sí** existen; el job corre con `cwd` = `src/`, pero también hay duplicado bajo la raíz del repo (`DWMRI/p2s/...`). Referencia canónica usada aquí:  
  - P2S DIPY D-Brain: `src/p2s/metrics/dbrain/bvalue_2500/noise_sigma_0.1/backend_dipy_model_ols/`  
  - MD-S2S D-Brain (paper `num_volumes=60`): `src/mds2s/metrics/dbrain/bvalue_2500/num_volumes_60/noise_sigma_0.1/learning_rate_0.0001/`

---

## Tabla principal — D-Brain (Rician σ = 0.1, K = 16 donde aplica)

| Método | PSNR_full | SSIM_full | PSNR_ROI | SSIM_ROI | FA_MAE | Params | s/vol | Notas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Noisy (referencia) | — | — | — | — | — | — | — | Definido en pipeline; no volcado en esta tabla |
| MP-PCA | 22.59 | 0.435 | 28.44 | 0.521 | 0.0898 | — | — | [`tmp/paper_final_k16_out/baselines/mppca/dbrain/`](../tmp/paper_final_k16_out/baselines/mppca/dbrain/) |
| P2S DIPY | 17.65 | 0.381 | 21.31 | 0.465 | 0.239 | — | — | `src/p2s/metrics/dbrain/.../backend_dipy_model_ols/` |
| MD-S2S (Self2self, 60 vol) | 15.26 | 0.425 | 15.40 | 0.508 | 0.230 | — | — | `src/mds2s/metrics/dbrain/.../num_volumes_60/...` |
| DRCNet-hybrid-RGS | 13.33⚠ | 0.005⚠ | 14.04⚠ | 0.005⚠ | 0.249⚠ | 116002 | 33.4 | [`registry.jsonl`](../tmp/paper_final_k16_out/registry.jsonl) — Alerta A |
| Restormer-hybrid-RGS | 10.08⚠ | -0.071⚠ | 12.55⚠ | -0.088⚠ | 0.247⚠ | 177883 | 129.3 | 1ª corrida OOM en T4; éxito en reintento — Alertas A/B |
| DRCNet supervisado (upper bound) | 13.32⚠ | 0.007⚠ | 14.09⚠ | 0.007⚠ | 0.258⚠ | 116002 | 34.3 | Alerta A |
| Restormer supervisado (upper bound) | 10.12⚠ | -0.069⚠ | 12.58⚠ | -0.085⚠ | 0.268⚠ | 177883 | 129.0 | Alerta B |

---

## Tabla — Stanford HARDI (sin GT limpio)

PSNR/SSIM aquí son **respecto al volumen de entrada** (coherencia), no calidad absoluta vs GT.

| Método | PSNR_full | SSIM_full | PSNR_ROI | Notas |
| --- | ---: | ---: | ---: | --- |
| P2S DIPY | 29.88 | 0.706 | 29.33 | `src/p2s/metrics/stanford/bvalue_2000/noise_sigma_0.01/backend_dipy_model_ols/` |
| MD-S2S | 19.07 | 0.690 | 18.88 | `src/mds2s/metrics/stanford/bvalue_2500/num_volumes_150/noise_sigma_0.01/learning_rate_0.0003/` |
| DRCNet-hybrid-RGS | 3.03⚠ | -0.067⚠ | 3.82⚠ | Alerta D |
| Restormer-hybrid-RGS | 26.99 | 0.617 | 26.05 | |

---

## Ablaciones ya registradas en `registry.jsonl` (D-Brain)

### §1.1 — Sequential vs RGS (K = 16)

| Configuración | PSNR_full | SSIM_full | PSNR_ROI | FA_MAE |
| --- | ---: | ---: | ---: | ---: |
| DRCNet `sequential` | 20.78 | 0.396 | 23.60 | 0.239 |
| DRCNet `rgs` (mismo job que fila principal) | 13.33⚠ | 0.005⚠ | 14.04⚠ | 0.249⚠ |
| Restormer `sequential` | —pend— | —pend— | —pend— | —pend— |

**Estado:** `restormer_dbrain_seq_k16_ablation` estaba **running** al 2026-05-14. Al terminar, añadir una fila leyendo la última entrada `success` de `registry.jsonl` para ese `job_id` y mover esta fila a números firmes.

### §1.7 — Progressive vs estándar (DRCNet)

| Configuración | PSNR_full | SSIM_full | PSNR_ROI | FA_MAE | Train |
| --- | ---: | ---: | ---: | ---: | --- |
| Progressive **OFF** (`drcnet_dbrain_progressive_off_ablation`) | 23.83 | 0.434 | 25.83 | 0.261 | 300 épocas, RGS |
| Progressive **ON** (`drcnet_dbrain_rgs_final`) | 13.33⚠ | 0.005⚠ | 14.04⚠ | 0.249⚠ | 360 épocas, RGS |
| Restormer progressive OFF | —pend— | —pend— | —pend— | —pend— | Job `restormer_dbrain_progressive_off_ablation` |

---

## Próximos resultados esperados

1. Completar **`restormer_dbrain_seq_k16_ablation`** → cierra la tabla §1.1 para Restormer.
2. **`restormer_dbrain_progressive_off_ablation`** → cierra §1.7 para Restormer.
3. **K-sweep** (20 jobs) → curva PSNR vs K (plan §1.2).
4. **2D vs 3D** (`drcnet_dbrain_2d_rgs_k16_ablation`, `restormer_dbrain_2d_rgs_k16_ablation`) → plan §1.4.
5. **mask_p**, **n_context**, **n_preds** → plan §1.5–1.6 y §5.
6. **Bloque σ** (24 jobs) → plan §4.2.
7. Jobs de **inferencia / gap / tiempos** y **`summarize_registry_final`** / **`collect_paper_artifacts_final`**.

---

## Changelog

| Fecha | Cambio |
| --- | --- |
| 2026-05-14 | Creación del documento: tablas desde `registry.jsonl`, MP-PCA desde `tmp/.../baselines/`, P2S/MD-S2S D-Brain desde `src/p2s/metrics/...` y `src/mds2s/metrics/...`; alertas A–E; estado del driver; Restormer-seq marcado como en curso. |
