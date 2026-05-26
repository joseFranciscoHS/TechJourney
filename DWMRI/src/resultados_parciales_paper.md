# Resultados parciales — paper DWMRI Hybrid RGS (vivo)

Documento vivo para ir consolidando números antes de redactar el paper. **No** sustituye al plan de experimentos en [`plan_para_escribir_el_paper.md`](plan_para_escribir_el_paper.md); solo refleja lo que ya salió de la corrida.

**Convenciones:** `⚠` = métrica registrada pero con alerta abierta (no usar en el paper sin revisar). `—pend—` = job pendiente o en curso. `—n/a—` = no aplica. `—` = no disponible en el artefacto actual.

---

## Estado de la corrida (snapshot)

| Campo | Valor |
| --- | --- |
| **Última actualización de este doc** | 2026-05-16 |
| **exp_id** | `paper_final_k16` |
| **Comando típico** | Ver [`experiments/checklist_pruebas_paper.md`](../experiments/checklist_pruebas_paper.md) §0 (`driver.py` + `paper_manifest_final.yaml`) |
| **Registry** | [`tmp/paper_final_k16_out/registry.jsonl`](../tmp/paper_final_k16_out/registry.jsonl) |
| **Driver state** | [`tmp/paper_final_k16_out/driver_state.json`](../tmp/paper_final_k16_out/driver_state.json) |
| **Driver events** | [`tmp/paper_final_k16_out/driver_events.jsonl`](../tmp/paper_final_k16_out/driver_events.jsonl) |
| **Job en curso (driver_state)** | `restormer_dbrain_progressive_off_ablation` — `running` |

### Resumen de jobs (`driver_state.json`)

Totales sobre **143** claves en estado (= **138** jobs del manifiesto + **5** entradas huérfanas `p2s_dbrain_sklearn_reference_*` sin manifiesto; ver eventos `manifest_mismatch` en `driver_events.jsonl`).

| status | count |
| --- | ---: |
| succeeded | 15 |
| failed | 1 |
| running | 1 |
| pending | 126 |

El job `failed` corresponde a `p2s_dbrain_sklearn_reference_final` (SIGKILL / OOM); omitido por diseño en el manifiesto actual.

### Progreso por bloque (solo jobs listados en `paper_manifest_final.yaml`)

| Bloque | succeeded | running | pending |
| --- | ---: | ---: | ---: |
| Tabla principal + baselines (`*_final` núcleo) | 12 | 0 | 4 |
| Ablación sampling / progressive (`seq_k16`, `progressive_off`) | 3 | 1 | 0 |
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

- **Alertas A–D (cerradas, 2026-05-16)** — Las métricas pobres de DRCNet y Restormer (PSNR ~13, SSIM negativo, discrepancia progressive ON/OFF, DRCNet Stanford ~3 dB) se debían a reconstrucción **sin** cargar el checkpoint entrenado. Tras regenerar con pesos entrenados, las cifras de las tablas siguientes son las válidas para el paper.

- **Alerta E — Rutas de métricas P2S / MD-S2S D-Brain (resuelta)**  
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
| DRCNet-hybrid-RGS | 23.93 | 0.440 | 26.88 | 0.525 | 0.259 | 116002 | 34.0 | [`registry.jsonl`](../tmp/paper_final_k16_out/registry.jsonl) — reconstrucción con checkpoint |
| Restormer-hybrid-RGS | 22.83 | 0.427 | 23.22 | 0.509 | 0.238 | 177883 | 126.2 | reconstrucción con checkpoint |
| DRCNet supervisado (upper bound) | 21.37 | 0.369 | 22.92 | 0.446 | 0.239 | 116002 | 34.0 | |
| Restormer supervisado (upper bound) | 21.57 | 0.370 | 22.60 | 0.447 | 0.237 | 177883 | 125.3 | |

---

## Tabla — Stanford HARDI (sin GT limpio)

PSNR/SSIM aquí son **respecto al volumen de entrada** (coherencia), no calidad absoluta vs GT.

| Método | PSNR_full | SSIM_full | PSNR_ROI | Notas |
| --- | ---: | ---: | ---: | --- |
| P2S DIPY | 29.88 | 0.706 | 29.33 | `src/p2s/metrics/stanford/bvalue_2000/noise_sigma_0.01/backend_dipy_model_ols/` |
| MD-S2S | 19.07 | 0.690 | 18.88 | `src/mds2s/metrics/stanford/bvalue_2500/num_volumes_150/noise_sigma_0.01/learning_rate_0.0003/` |
| DRCNet-hybrid-RGS | 39.80 | 0.969 | 38.07 | reconstrucción con checkpoint |
| Restormer-hybrid-RGS | 27.02 | 0.618 | 26.08 | |

---

## Ablaciones ya registradas en `registry.jsonl` (D-Brain)

### §1.1 — Sequential vs RGS (K = 16)

| Configuración | PSNR_full | SSIM_full | PSNR_ROI | FA_MAE |
| --- | ---: | ---: | ---: | ---: |
| DRCNet `sequential` | 20.78 | 0.396 | 23.60 | 0.239 |
| DRCNet `rgs` (mismo job que fila principal) | 23.93 | 0.440 | 26.88 | 0.259 |
| Restormer `sequential` | 20.49 | 0.385 | 21.64 | 0.246 |

### §1.7 — Progressive vs estándar (DRCNet)

| Configuración | PSNR_full | SSIM_full | PSNR_ROI | FA_MAE | Train |
| --- | ---: | ---: | ---: | ---: | --- |
| Progressive **OFF** (`drcnet_dbrain_progressive_off_ablation`) | 23.91 | 0.440 | 26.91 | 0.238 | 300 épocas, RGS |
| Progressive **ON** (`drcnet_dbrain_rgs_final`) | 23.93 | 0.440 | 26.88 | 0.259 | 360 épocas, RGS |
| Restormer progressive OFF | —pend— | —pend— | —pend— | —pend— | Job `restormer_dbrain_progressive_off_ablation` |

---

## Próximos resultados esperados

1. **`restormer_dbrain_progressive_off_ablation`** → cierra §1.7 para Restormer.
2. **K-sweep** (20 jobs) → curva PSNR vs K (plan §1.2).
3. **2D vs 3D** (`drcnet_dbrain_2d_rgs_k16_ablation`, `restormer_dbrain_2d_rgs_k16_ablation`) → plan §1.4.
4. **mask_p**, **n_context**, **n_preds** → plan §1.5–1.6 y §5.
5. **Bloque σ** (24 jobs) → plan §4.2.
6. Jobs de **inferencia / gap / tiempos** y **`summarize_registry_final`** / **`collect_paper_artifacts_final`**.

---

## Changelog

| Fecha | Cambio |
| --- | --- |
| 2026-05-14 | Creación del documento: tablas desde `registry.jsonl`, MP-PCA desde `tmp/.../baselines/`, P2S/MD-S2S D-Brain desde `src/p2s/metrics/...` y `src/mds2s/metrics/...`; alertas A–E; estado del driver; Restormer-seq marcado como en curso. |
| 2026-05-16 | Métricas DRCNet y Restormer actualizadas tras reconstrucción con checkpoint entrenado; `restormer_dbrain_seq_k16_ablation` completado; alertas A–D cerradas; driver: 15 succeeded, job en curso `restormer_dbrain_progressive_off_ablation`. |
