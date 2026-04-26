# Checklist Fase 1 — Experimentos para el Artículo DWMRI Hybrid RGS

> **Método propuesto:** un único esquema — el Híbrido RGS (Scheme 2): red con convoluciones 3D que recibe K volúmenes aleatorios del shell completo G, aplica máscara de Bernoulli solo sobre el canal objetivo (`target_channel = K-1`), y supervisa únicamente en los voxeles enmascarados (pérdida MSE enmascarada, J-invariante). Se evalúa con dos arquitecturas backbone: **DRCNet-hybrid-RGS** y **Restormer-hybrid-RGS**.
>
> **Datasets disponibles:**
>
> - **D-Brain** — b=2500, G=60 volúmenes DWI, 6 b0s. Ruido Rician sintético (tiene GT limpio).
> - **Stanford HARDI** — b=1000, G=150 volúmenes DWI, 10 b0s. Ruido real de escáner (sin GT).

---

## Sección 1 — Ablaciones del Método

---

### 1.1 Sequential vs RGS con mismo K

**Pregunta que responde:**
¿El muestreo aleatorio de gradientes en cada paso de entrenamiento (RGS) aporta mejor calidad que fijar un subconjunto secuencial de K volúmenes y rotar el objetivo entre ellos? En otras palabras, ¿la diversidad de contexto angular durante el entrenamiento tiene valor propio, o lo que importa es simplemente cuántos volúmenes se usan?

**Configuraciones a comparar:**

| Modo         | Descripción                                                                                                                                       | K   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| `sequential` | El stack de entrada siempre contiene los mismos K volúmenes fijos (los primeros K del shell). El índice objetivo rota de 0 a K-1 en cada muestra. | 24  |
| `rgs`        | En cada muestra se sortean K índices distintos sin reemplazo del shell completo G. El objetivo siempre está en `target_channel = K-1`.            | 24  |

**Valor de K a usar:** K=24. Es el valor configurado en ambas arquitecturas para D-Brain (G=60) y es lo suficientemente grande para que la diferencia entre "siempre los mismos 24" y "24 aleatorios de 60" sea medible. Si los recursos lo permiten, repetir con K=10.

**Arquitecturas involucradas:** Ambas — DRCNet-hybrid-RGS y Restormer-hybrid-RGS. Correr el experimento con cada una de forma independiente para verificar que el hallazgo es robusto a la arquitectura.

**Dataset:** D-Brain (b=2500, tiene GT para calcular todas las métricas). Stanford como verificación secundaria si los tiempos lo permiten.

**Protocolo:** Mismos hiperparámetros en todo (learning rate, batch size, épocas, mask_p=0.3, progressive stages si están activados). La única variable que cambia es `shell_sampling_mode: sequential` vs `rgs`. Los checkpoints deben guardarse en rutas separadas por modo.

**Salida esperada:**

- Tabla con PSNR / SSIM / MSE (full image y ROI) para `sequential` vs `rgs`, separada por arquitectura.
- Si `rgs` supera a `sequential` con el mismo K: argumento directo de que la diversidad angular actúa como data augmentation implícito durante el entrenamiento. Esto es un aporte del método.
- Si son equivalentes: el RGS se justifica igualmente por eficiencia de memoria (permite escalar a shells grandes G >> K sin que el modelo crezca).

---

### 1.2 Variación de K (tamaño del subconjunto de gradientes)

**Pregunta que responde:**
¿Cuántos volúmenes de gradiente son necesarios en el stack de entrada para que el modelo aprenda a denoisar correctamente? ¿Existe un K mínimo práctico? ¿Hay punto de rendimientos decrecientes?

**Valores de K a evaluar:**

| Dataset  | G (shell completo) | Valores de K          |
| -------- | ------------------ | --------------------- |
| D-Brain  | 60                 | 5, 10, 15, 20, 24, 30 |
| Stanford | 150                | 5, 10, 20, 30, 50     |

**Arquitectura de referencia:** DRCNet-hybrid-RGS (más rápido de entrenar que Restormer, apto para barridos). Una vez encontrado el K óptimo, validar con Restormer.

**Implicación de K en el modelo:** En modo RGS, `model.in_channel = K`. Cada valor de K requiere instanciar y entrenar un modelo diferente (los pesos de la capa de entrada cambian de dimensión). Esto no es un hiperparámetro que se pueda evaluar con el mismo checkpoint.

**Protocolo:** Mantener fijo todo lo demás (mask_p=0.3, learning_rate, épocas por stage si se usa progressive). Entrenar cada K desde cero. Para D-Brain con K=30 verificar que el modelo cabe en GPU (in_channel=30 × base_filters=32 puede ser pesado en patches grandes).

**Salida esperada:**

- Curva de PSNR vs K (eje x) para cada dataset. Se espera una curva con forma de "codo": calidad sube rápidamente con K pequeño y se satura a partir de cierto punto.
- Tabla complementaria con tiempo de entrenamiento y tiempo de inferencia por volumen en función de K. Cuantifica el tradeoff calidad/costo.
- Este experimento define el K recomendado que se reporta como configuración principal en el paper.
- Hipótesis de trabajo: K=10 debería ser suficiente para D-Brain (G=60 tiene mucha redundancia angular a b=2500). Para Stanford (G=150) puede requerirse K mayor.

---

### 1.3 DRCNet-hybrid-RGS vs Restormer-hybrid-RGS

**Pregunta que responde:**
¿Cuál de las dos arquitecturas backbone ofrece mejor tradeoff calidad/costo para esta tarea? ¿El mecanismo de atención del Restormer aporta sobre las convoluciones factorizadas del DRCNet cuando el dominio es 3D y los patches son pequeños (16³–32³)?

**Condiciones de la comparación (deben ser idénticas):**

- K fijo al valor óptimo encontrado en 1.2.
- Mismo dataset, mismos splits, misma semilla.
- Mismo número total de épocas de entrenamiento (sumando stages si se usa progressive).
- Mismo mask_p, mismo learning rate base (ambos configs tienen ~0.00045).
- Mismo protocolo de reconstrucción: mismos n_preds y n_context.

**Métricas a reportar por arquitectura:**

- PSNR / SSIM / MSE (full y ROI)
- FA-error y MD-error (requiere pipeline DTI, ver Sección 3)
- Número total de parámetros entrenables
- Tiempo de entrenamiento por época (en segundos)
- Tiempo de inferencia por volumen (en segundos)
- Pico de memoria GPU durante entrenamiento (MB)

**Salida esperada:**

- Una tabla que muestre que ninguna arquitectura domina en todas las dimensiones: probablemente Restormer tendrá mejor PSNR pero mayor costo; DRCNet será más rápido y apto para datasets o recursos limitados. Esta tensión es material de discusión valiosa.
- Si los resultados son muy similares en calidad: argumentar DRCNet como opción práctica (menos parámetros, más rápido).
- Si Restormer es claramente superior en métricas downstream (FA/MD): argumentar su uso cuando la precisión clínica es prioritaria.

---

### 1.4 Convoluciones 3D vs Conv2D slice-by-slice (ablación crítica)

**Pregunta que responde:**
¿El beneficio del método proviene de usar convoluciones 3D (que capturan contexto volumétrico entre cortes) o del esquema de entrenamiento híbrido RGS en sí? Sin este experimento, un revisor puede atribuir cualquier mejora a la arquitectura más expresiva y no al esquema propuesto.

**Cómo construir la versión 2D:**
Crear una variante del mismo esquema de entrenamiento (RGS + máscara de Bernoulli + pérdida MSE enmascarada) pero usando una red con convoluciones 2D. La forma más directa es procesar cada corte axial (z) del patch de forma independiente con convoluciones 2D, tal como hace MD-S2S original. El stack de entrada sería (K, X, Y) para cada z, y la red predice (1, X, Y) por corte. No se comparte información entre cortes durante la inferencia.

**Esto NO es:** un modelo distinto. Es el mismo esquema híbrido RGS con la única diferencia de que la red no conecta voxeles a lo largo del eje z.

**Arquitectura base para esta ablación:** DRCNet, porque tiene una versión 2D documentada en la literatura (el DRCNet original usa Conv2D).

**Condiciones fijas:** K óptimo de 1.2, mismo mask_p, mismo dataset (D-Brain), mismos hiperparámetros de entrenamiento.

**Salida esperada:**

- Tabla comparando 2D vs 3D en PSNR / SSIM / FA-error.
- Si 3D > 2D: el paper puede afirmar explícitamente que la incorporación de contexto volumétrico (entre cortes) es responsable de parte del beneficio, independientemente del esquema.
- Si 2D ≈ 3D: el claim del paper se recentra en el esquema RGS híbrido como contribución principal, y la elección de 3D se justifica por la completitud del contexto espacial aunque no por una brecha de calidad grande.
- Este experimento es **obligatorio para responder a revisores**. Sin él, la revisión puede rechazar el claim de que "las convoluciones 3D aportan sobre el estado del arte".

---

### 1.5 Sensibilidad a mask_p

**Pregunta que responde:**
¿Qué tan sensible es el método al hiperparámetro `mask_p` (fracción de voxeles enmascarados en el canal objetivo durante el entrenamiento)? ¿El valor 0.3 actual es robusto o crítico?

**Valores a evaluar:** mask_p ∈ {0.1, 0.2, 0.3, 0.5, 0.7}

**Relación teórica con J-invarianza:**

- mask_p bajo (ej. 0.1): solo el 10% de voxeles están enmascarados. La red tiene acceso a casi todo el volumen objetivo en el input, lo que reduce la J-invarianza (la predicción en los pocos voxeles enmascarados puede seguir dependiendo de los vecinos no enmascarados muy cercanos). Puede introducir bias.
- mask_p alto (ej. 0.7): el 70% está enmascarado. La supervisión usa muchos voxeles (bueno para varianza), pero la red pierde el contexto espacial propio del volumen objetivo, dependiendo más de los volúmenes de contexto. Puede aumentar bias si la redundancia angular sola no es suficiente.
- mask_p=0.3–0.5: zona teóricamente equilibrada.

**Arquitectura:** DRCNet-hybrid-RGS (más rápido). Dataset: D-Brain. K fijo al óptimo.

**Salida esperada:**

- Tabla o curva de PSNR/SSIM vs mask_p.
- Si el rendimiento es estable en [0.2, 0.5]: reportar que el método es robusto al hiperparámetro, lo cual simplifica el uso práctico.
- Si hay un pico claro: reportar el valor óptimo y justificarlo con el argumento teórico de bias-varianza de J-invarianza (ya documentado en el reporte técnico).

---

### 1.6 Sensibilidad a n_context en inferencia

**Pregunta que responde:**
¿Cuántos contextos aleatorios (tuples de K-1 gradientes) son necesarios durante la inferencia RGS para obtener calidad estable? Dado el costo O(V × n_context × n_preds × t_forward), encontrar el mínimo n_context que no degrada la reconstrucción es importante para la usabilidad práctica.

**Valores a evaluar:** n_context ∈ {1, 3, 5, 12, 24, 48}

**Distinción con n_preds:** `n_context` controla cuántos subconjuntos distintos de K-1 gradientes de contexto se prueban por volumen objetivo. `n_preds` controla cuántas máscaras de Bernoulli distintas se aplican por contexto. Son dos fuentes de varianza distintas (angular vs espacial). Este experimento fija n_preds=10 y varía n_context.

**Modelo:** El mejor checkpoint ya entrenado (no se re-entrena nada). Se aplica directamente sobre D-Brain.

**Salida esperada:**

- Curva de PSNR vs n_context y curva de tiempo de inferencia vs n_context (en el mismo gráfico o lado a lado).
- Se espera que la calidad sature después de cierto n_context (hipótesis: ~12–24 para D-Brain con G=60). Esto da la recomendación práctica del paper.
- Tabla de tiempo total de reconstrucción del volumen completo en función de n_context. Cuantifica el costo de escalar.

---

### 1.7 Progressive training vs training estándar

**Pregunta que responde:**
¿El entrenamiento progresivo (patches grandes → chicos con batch creciente) mejora la calidad final respecto a entrenar directamente con el patch size final?

**Configuraciones:**

| Modo        | Descripción                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------------------- |
| Progressive | 3 stages: 32³ (batch=128, 100 épocas) → 24³ (batch=256, 100 épocas) → 16³ (batch=256, 100 épocas)              |
| Estándar    | Entrena directamente con patch=16³, batch equivalente al total de épocas del progressive (300 épocas en total) |

**Racional:** El entrenamiento progresivo expone al modelo primero a contexto espacial amplio (patches grandes) y luego refina con patches finos. La hipótesis es que esto mejora la convergencia similar a un currículo de aprendizaje.

**Arquitectura:** DRCNet-hybrid-RGS. Dataset: D-Brain. K y mask_p fijos.

**Salida esperada:**

- Comparación de pérdida de validación durante el entrenamiento (curva de loss vs época).
- Tabla de métricas finales: PSNR / SSIM / MSE en test.
- Si progressive > estándar: incluir en el paper como componente del método. Si son equivalentes: simplificar el método y omitir progressive del protocolo recomendado.

---

## Sección 2 — Baselines de Comparación

El método propuesto se compara contra estos baselines en todos los experimentos que reportan métricas finales. La tabla principal del paper incluirá todas las filas.

---

### 2.1 MP-PCA (Veraart et al., 2016)

**Qué es:** Denoising basado en análisis de componentes principales con umbralización de valores propios. Estándar de facto en preprocessing de DWMRI. Implementado en DIPY (`dipy.denoise.localpca`).

**Cómo correrlo:**

```python
from dipy.denoise.localpca import mppca
denoised, sigma = mppca(data_4d, patch_radius=2, return_sigma=True)
```

El `data_4d` debe estar en unidades originales (no normalizado a [0,1]); aplicar el denoising sobre los datos crudos y luego normalizar para métricas.

**Datasets:** D-Brain y Stanford. Para D-Brain: aplicar sobre los datos con ruido sintético Rician σ=0.1. Para Stanford: aplicar sobre los datos del escáner directamente.

**Salida esperada:** Un array denoised (X, Y, Z, V) para cada dataset. Calcular todas las métricas de la Sección 3. MP-PCA es una cota baja-media para los métodos basados en aprendizaje; esperamos que nuestro método lo supere en PSNR y en métricas de tensor (FA/MD).

---

### 2.2 Patch2Self — backend DIPY (OLS)

**Qué es:** El Patch2Self original (Fadnavis et al., 2020). Regresión lineal OLS donde para cada volumen DWI se usan los demás como predictores. Implementado directamente en DIPY.

**Cómo correrlo:** Ya está implementado en `p2s/run.py` con `backend: dipy`. Verificar que se corre con `model: ols`, `shift_intensity: true`, `b0_threshold: 50`.

**Datasets:** D-Brain y Stanford. Usar exactamente la misma configuración que ya tienes.

**Nota sobre normalización:** El pipeline de `p2s/run.py` ya maneja la normalización. Verificar que las métricas se computan con los mismos rangos que el método propuesto (ambos normalizados a [0,1] por volumen).

**Salida esperada:** `denoised_patch2self.nii.gz` + métricas en `metrics.json` y `metrics_roi.json`. Patch2Self-OLS es el baseline angular más directo de comparar con nuestro método, porque la innovación principal es reemplazar la regresión lineal por una red con convoluciones 3D usando el mismo principio de J-invarianza angular.

---

### 2.3 Patch2Self — backend sklearn_reference

**Qué es:** Reimplementación del Patch2Self siguiendo el estilo MD-S2S (Kang et al.) con regresores sklearn. Ya implementado en `p2s/sklearn_patch2self.py`.

**Configuración a correr:**

```yaml
backend: sklearn_reference
sklearn_model: ols
patch_radius: [0, 0, 0]
patch_stride: 1
use_b0_as_predictors: true
```

Esta configuración reproduce el P2S de referencia de Kang et al. con un solo voxel como patch (sin contexto espacial).

**Datasets:** Solo D-Brain (Stanford con G=150 y stride=1 sería computacionalmente prohibitivo; usar stride=4 para Stanford si se incluye).

**Salida esperada:** Métricas comparables con 2.2. Si difieren significativamente, documentar la diferencia e investigar la causa (normalización, b0 handling, etc.).

---

### 2.4 MD-S2S Conv2D (Kang et al., 2021)

**Qué es:** El método Multidimensional Self2Self original. Aplica enmascaramiento de Bernoulli sobre todos los volúmenes simultáneamente (no solo el canal objetivo) usando Conv2D sobre cortes axiales. Es el predecesor directo del método propuesto.

**Cómo obtenerlo:** El repo oficial de Kang et al. está en GitHub (`B9Kang/MD-S2S-Multidimensional-Self2Self`). Dos opciones:

- Usar el repo oficial directamente sobre los mismos datos.
- Usar la implementación `mds2s/` que ya tienes (Self2self con Conv2D, que es el mismo principio).

**Diferencia clave con el método propuesto:** MD-S2S usa Conv2D (procesa cada corte axial independientemente), aplica máscara a todos los canales a la vez, y no usa el concepto de K-subset (usa todos los volúmenes). Nuestro método usa Conv3D, aplica máscara solo al canal objetivo, y usa K-subset aleatorio.

**Datasets:** D-Brain y Stanford. Misma normalización.

**Salida esperada:** Tabla de métricas. MD-S2S Conv2D + nuestro método 3D debe mostrar la brecha que cierra el uso de convoluciones 3D y el esquema híbrido. Si el gap es pequeño, refuerza el argumento de que el esquema importa más que la arquitectura (ver ablación 1.4).

---

### 2.5 MDS2S — implementación Self2self (ya disponible)

**Qué es:** La implementación `mds2s/` del repositorio. Usa la red `Self2self` con Conv2D, Dropout ensemble, y máscara de Bernoulli sobre todos los volúmenes por corte axial (Z, V, X, Y).

**Cómo correrlo:** Ya está en `mds2s/run.py`. Documentar exactamente los hiperparámetros usados (`num_volumes`, `mask_p`, `dropout_p`, `num_epochs`). Para D-Brain: `num_volumes=10` o el mismo número que K en el método propuesto para comparación justa.

**Datasets:** D-Brain y Stanford.

**Salida esperada:** Este es el baseline más cercano al método propuesto en términos de entrenamiento self-supervised, pero con arquitectura 2D y sin el esquema RGS. Esperamos que nuestro método lo supere especialmente en métricas downstream (FA/MD).

---

### 2.6 Modelo supervisado equivalente (cota superior)

**Qué es:** El mismo DRCNet-hybrid o Restormer-hybrid entrenado con acceso al ground truth, usando pérdida MSE estándar (no enmascarada) sobre todos los voxeles del volumen objetivo. Representa el techo de calidad alcanzable con la misma arquitectura.

**Protocolo de entrenamiento:** Igual que el método propuesto excepto: sin máscara de Bernoulli, la pérdida es `MSE(prediction, GT_volume)` sobre todos los voxeles. No se necesita RGS (puede usar sequential con todos los volúmenes).

**Dataset:** Solo D-Brain (necesita GT limpio). No aplicable a Stanford.

**Salida esperada:** Tabla con las métricas del modelo supervisado. El gap entre el supervisado y el self-supervised es el "costo de la auto-supervisación". Si el gap es pequeño (< 1 dB en PSNR), es un argumento muy fuerte a favor del método propuesto para uso clínico real donde GT no está disponible.

---

### 2.7 BM4D (opcional)

**Qué es:** Block-Matching 4D, una extensión del clásico BM3D para datos 4D. Baseline no-aprendido que explota redundancia espacio-angular.

**Por qué incluirlo:** Algunos revisores de journals de imagen médica (MRM, NeuroImage) lo esperan como referencia de métodos clásicos. No aprendido = sin riesgo de overfitting.

**Cómo correrlo:** Implementación en MATLAB (oficial) o en Python (bm4d en PyPI). Si el tiempo no alcanza, puede omitirse con la justificación de que MP-PCA es el estándar actual en DWMRI y BM4D no fue diseñado específicamente para datos de difusión.

---

## Sección 3 — Métricas de Evaluación

---

### 3.1 PSNR, SSIM, MSE — imagen completa

**Estado:** Ya implementado en `utils/metrics.py`.

**Verificación pendiente:** Confirmar que las métricas se computan sobre los mismos datos normalizados a [0,1] por volumen para todos los métodos. Un error común es comparar métodos cuyos outputs tienen rangos distintos. El pipeline de `rescale_to_01` en `utils/data.py` debe aplicarse de manera consistente antes de calcular métricas para todos los baselines y el método propuesto.

---

### 3.2 PSNR, SSIM, MSE — ROI (solo tejido cerebral)

**Estado:** Ya implementado. ROI definido como voxeles donde `original_data > 0.02` en cualquier volumen.

**Verificación pendiente:** Que el mismo umbral (0.02) se use para todos los métodos y ambos datasets. Documentar cuántos voxeles quedan en el ROI como porcentaje del total (ya se loggea). Esta métrica es más relevante clínicamente que la métrica full-image.

---

### 3.3 FA (Fractional Anisotropy) — error respecto a GT

**Qué es:** FA mide la anisotropía del tensor de difusión. Es la métrica más reportada en estudios de white matter. Un método que mejora PSNR pero degrada FA no aporta clínicamente.

**Pipeline a implementar:**

```python
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table

def compute_fa(data_xyzv, gtab):
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data_xyzv)
    return tenfit.fa  # shape (X, Y, Z)
```

**Lo que se necesita:**

- `data_xyzv`: el volumen 4D incluyendo los b0s (el tensor necesita b0s para ajustarse). Esto requiere concatenar los b0s que se omiten durante el denoising con los DWI denoised.
- `gtab`: gradient table de D-Brain o Stanford.
- Correr para: GT limpio, datos ruidosos, y cada método denoised.
- Calcular `FA_error = mean(|FA_denoised - FA_GT|)` sobre el ROI cerebral.

**Salida esperada:** Tabla de FA_error (MAE) por método. Incluir mapas de FA en la figura del paper (comparación visual GT vs noisy vs denoised). Esta métrica es la que diferencia un paper de DWMRI de uno de restauración genérica.

---

### 3.4 MD (Mean Diffusivity) — error respecto a GT

**Pipeline:** Mismo que FA. `tenfit.md` del mismo ajuste de tensor.

**Métrica:** `MD_error = mean(|MD_denoised - MD_GT|)` sobre ROI.

**Unidades:** MD está en mm²/s (~0.7–1.0 × 10⁻³ mm²/s en tejido). Reportar el error en las mismas unidades o en % relativo respecto a GT.

---

### 3.5 AD y RD (Axial y Radial Diffusivity)

**Pipeline:** `tenfit.ad` y `tenfit.rd`. Mismo tensor ajustado.

**Cuándo incluirlos:** Si la tabla ya es larga con FA y MD, AD/RD pueden ir en el suplementario. Incluirlos en el análisis interno aunque no estén en la tabla principal.

---

### 3.6 Calidad de tractografía (opcional, alta dificultad)

**Qué implicaría:**

- Ajustar tractografía determinística o probabilística (EuDX o iFOD2) sobre el FA/MD denoised vs GT.
- Métricas de comparación de streamlines: overlap, longitud media, conectividad entre ROIs.
- Implementación en DIPY (tracking) o FSL (probtrack).

**Decisión:** Incluir solo si el tiempo lo permite y si el venue objetivo lo requiere (Nature Methods o MRM lo valorarían; NeuroImage Anual podría no exigirlo). Si se omite, mencionar en limitaciones que queda como trabajo futuro.

---

### 3.7 Tabla de tiempos de inferencia

**Qué medir:**

- Tiempo de entrenamiento total (horas) por método, incluyendo arquitectura y dataset.
- Tiempo de inferencia por volumen DWI (segundos) para el método propuesto con distintos n_context y n_preds.
- Tiempo de inferencia total para reconstruir el volumen completo (V volúmenes).
- Para baselines: tiempo de denoising de MP-PCA y P2S sobre el mismo volumen.

**Cómo medirlo:** Usar `time.time()` antes y después de la llamada a inferencia. Para el método propuesto, medir `reconstruct_dwis_rgs` con distintas configuraciones de n_context y n_preds. Hardware a documentar: GPU model, RAM, CPU.

**Salida esperada:** Tabla en el paper con columnas: Método | Tiempo entrenamiento | Tiempo inferencia/volumen | Tiempo total | GPU memory. Esta tabla es clave para que los revisores evalúen usabilidad práctica.

---

## Sección 4 — Datasets y Protocolo

---

### 4.1 D-Brain b=2500 — protocolo completo

**Estado:** Disponible y funcional.

**Lo que falta documentar para el paper:**

- Semilla aleatoria fija para generación del ruido Rician (ya existe `np.random.seed(91021)` en `data.py`, verificar que se usa en todos los experimentos).
- Número exacto de volúmenes usados para entrenamiento vs los omitidos (b0s).
- Dimensiones espaciales usadas: take_x=128, take_y=128, take_z=96 — documentar que es un crop, no el volumen completo, y por qué.
- Confirmar que el mismo crop y la misma semilla se usan para todos los métodos comparados.

---

### 4.2 D-Brain — variación del nivel de ruido

**Qué hacer:** Entrenar y evaluar el método propuesto (y los baselines) con σ ∈ {0.05, 0.10, 0.15} Rician.

**Por qué:** Demuestra que el método es robusto y consistente en distintos regímenes de SNR. Un método que solo funciona bien a σ=0.1 (el valor de entrenamiento) puede no generalizar.

**Protocolo:** Para cada σ, tanto el entrenamiento como la inferencia usan ese nivel de ruido. No se hace transfer entre niveles de ruido (no se entrena a σ=0.1 y se testa a σ=0.15, aunque ese análisis también sería interesante para demostrar robustez).

**Salida esperada:** Curva de PSNR vs σ para el método propuesto y los baselines principales (MP-PCA y P2S). El método propuesto debe mantener ventaja en todos los niveles.

---

### 4.3 Stanford HARDI b=1000 — protocolo completo

**Estado:** Disponible. Sin GT, el loader retorna `original_data=None`.

**Estrategia de evaluación sin GT:**

- Para métricas PSNR/SSIM: no aplican directamente. Se puede usar el volumen menos ruidoso como proxy (promedio de múltiples adquisiciones del mismo gradiente si estuviera disponible, que no es el caso aquí).
- Para métricas downstream: ajustar tensor DTI sobre los datos denoised y evaluar coherencia (mapas FA deben ser suaves y anatómicamente plausibles). No se puede cuantificar el error, pero sí comparar visualmente y reportar FA medio en ROIs conocidos (cuerpo calloso, corona radiata).
- Opcionalmente: usar SSIM entre el denoised y el original ruidoso como métrica de coherencia (no de calidad absoluta).

**Lo que sí se puede reportar para Stanford:** Mapas FA cualitativos comparativos, y tiempo de inferencia sobre un shell más grande (G=150 vs G=60 en D-Brain).

---

### 4.4 Dataset con ruido real y GT (aspiracional)

**Por qué es importante:** El mayor punto débil del paper es que D-Brain usa ruido sintético Rician, que puede no capturar la distribución real del ruido de escáner (structured noise, Gibbs ringing, partial volume effects). Un dataset real con GT elevaría significativamente el impacto.

**Opciones a explorar:**

- **Phantom:** Adquirir o conseguir datos de phantom DWI con múltiples promedios (ground truth = promedio de N adquisiciones). Algunos datasets públicos HCP incluyen esto.
- **HCP test-retest:** El Human Connectome Project tiene adquisiciones repetidas de los mismos sujetos. Uno puede actuar como "GT" y el otro como "noisy" (aunque ambos tienen ruido real).
- **Simulaciones con Gibbs:** Agregar artefactos de Gibbs al ruido sintético para modelar mejor las violaciones de las asunciones de J-invarianza.

**Decisión editorial:** Si no es posible obtener un dataset real para esta iteración, documentar explícitamente en la sección de limitaciones del paper.

---

### 4.5 Protocolo de reproducibilidad

**Qué documentar antes de escribir:**

- Semilla aleatoria para todas las operaciones estocásticas (generación de ruido, generación de masks, splits, RGS sampling): verificar que se puede reproducir exactamente.
- Versiones de librerías: PyTorch, DIPY, scikit-image, NumPy — fijar en `requirements.txt`.
- Hardware: GPU model, VRAM, CUDA version.
- Número exacto de parámetros de cada modelo entrenado.
- Los checkpoints finales de cada experimento deben guardarse con una convención de nombres que incluya: dataset / arquitectura / K / noise_type / sigma / learning_rate (ya implementado en las rutas de checkpoint).

---

## Sección 5 — Análisis de Inferencia RGS

---

### 5.1 Curva calidad vs n_context

Ver descripción detallada en Sección 1.6. Complementar con el análisis de la varianza de la reconstrucción: correr n_context=24 varias veces con distintas semillas y reportar la desviación estándar del PSNR resultante. Esto cuantifica la estabilidad del estimador Monte Carlo.

---

### 5.2 Curva calidad vs n_preds (passes espaciales)

**Pregunta que responde:** Fijado n_context, ¿cuántas masks de Bernoulli distintas por contexto son necesarias para estabilizar la reconstrucción espacial?

**Valores a evaluar:** n_preds ∈ {1, 3, 5, 10, 15, 20}

**Protocolo:** Mismo checkpoint entrenado. Variar solo n_preds en `reconstruct_dwis_rgs`. Medir PSNR y tiempo.

**Relación con el espíritu Self2Self:** n_preds es el ensemble espacial análogo al dropout ensemble de S2S. Más n_preds → menos varianza en la estimación espacial → mejor calidad, pero más tiempo. Documentar cuántos n_preds son suficientes (hipótesis: 10 es suficiente para convergencia, 5 para uso práctico rápido).

---

### 5.3 Análisis del gap training/inference

**Qué es el problema:** Durante el entrenamiento, la red solo ve voxeles enmascarados en `target_channel` (los demás están a cero). Durante la inferencia, el volumen objetivo se ve parcialmente (con máscara) pero se predice sobre todos los voxeles. Esta ligera inconsistencia es compartida con Noise2Void y P2S.

**Cómo analizarlo cuantitativamente:**

- Opción 1: Durante inferencia, aplicar la misma máscara que en training (solo predecir voxeles enmascarados, para los visibles usar el valor original). Comparar esta versión "pura" con la versión estándar de inferencia.
- Opción 2: Medir el PSNR por separado en voxeles que estaban enmascarados vs no enmascarados en un pass de máscara dado. Si la red generaliza bien, ambos PSNRs deberían ser similares.

**Salida esperada:** Un párrafo en la sección de método del paper justificando por qué el gap no invalida el método (mismo argumento que usa Noise2Void: la red aprende el estimador de Bayes óptimo bajo la distribución de máscaras, y en inferencia sin máscara el error adicional es menor que el ruido original).

---

### 5.4 Tabla de tiempos completa (por configuración)

**Qué medir exactamente:**

| Configuración            | Tiempo por volumen (s) | Tiempo total (min) |
| ------------------------ | ---------------------- | ------------------ |
| n_context=5, n_preds=5   | ...                    | ...                |
| n_context=12, n_preds=10 | ...                    | ...                |
| n_context=24, n_preds=12 | ...                    | ...                |
| n_context=48, n_preds=20 | ...                    | ...                |

Medir sobre D-Brain (V=60) en la misma GPU que se usó para entrenamiento. Incluir el tiempo de carga de datos y del checkpoint.

---

## Sección 6 — Escritura y Figuras

---

### 6.1 Figura 1 — Diagrama del método

**Qué mostrar:**

- Stack de K volúmenes de gradiente seleccionados aleatoriamente del shell G.
- El canal `target_channel = K-1` con la máscara de Bernoulli aplicada (voxeles en negro = enmascarados).
- La red (DRCNet o Restormer, mostrar como caja negra con "3D Conv" dentro).
- La salida: predicción de 1 volumen.
- La pérdida: MSE solo sobre los voxeles enmascarados (flecha roja de los voxeles enmascarados a la función de pérdida).
- Diagrama secundario pequeño: la inferencia RGS con múltiples contextos → promedio.

**Herramienta sugerida:** Inkscape, Affinity Designer, o TikZ para publicación. Para borrador: draw.io o Figma.

---

### 6.2 Figura 2 — Resultados visuales cualitativos

**Layout sugerido (6 columnas × N filas de cortes):**

| GT | Noisy | MP-PCA | P2S | MD-S2S 2D | DRCNet-RGS (nuestro) | Restormer-RGS (nuestro) |

**Cortes a mostrar:** Al menos un corte axial (z medio) y un corte sagital del mismo volumen. Mostrar el mismo corte para todos los métodos.

**Volumen a visualizar:** Elegir uno representativo (no el mejor ni el peor). Típicamente se elige el volumen con el gradiente más "difícil" (mayor señal de difusión, más ruidoso).

**Incluir mapa de error:** Columna adicional con `|GT - denoised|` escalado para visibilidad. El mapa de error debe usar colormap divergente (ej. bwr o seismic).

---

### 6.3 Figura 3 — Mapas de FA comparativos

**Layout:** 1 fila por método, mostrando el mapa FA en un corte axial que cruce el cuerpo calloso (estructura de referencia clásica).

**Qué mostrar:** FA(GT), FA(noisy), FA(MP-PCA), FA(P2S), FA(DRCNet-RGS), FA(Restormer-RGS). Usar colormap `hot` o `jet` con rango [0, 1]. Incluir barra de color.

**Por qué es la figura más importante del paper:** FA es lo que el clínico mira. Si FA(denoised) se ve más suave y anatómicamente coherente que FA(noisy) y comparable a FA(GT), el paper tiene impacto clínico directo.

---

### 6.4 Tabla principal — comparación de todos los métodos

**Estructura:**

| Método                      | PSNR ↑ | SSIM ↑ | MSE ↓ | PSNR-ROI ↑ | FA-error ↓ | MD-error ↓ | Tiempo/vol (s) |
| --------------------------- | ------ | ------ | ----- | ---------- | ---------- | ---------- | -------------- |
| Noisy (sin denoising)       |        |        |       |            |            |            | —              |
| MP-PCA                      |        |        |       |            |            |            |                |
| P2S DIPY                    |        |        |       |            |            |            |                |
| MD-S2S Conv2D               |        |        |       |            |            |            |                |
| DRCNet-hybrid-RGS (K=?)     |        |        |       |            |            |            |                |
| Restormer-hybrid-RGS (K=?)  |        |        |       |            |            |            |                |
| Supervisado (cota superior) |        |        |       |            |            |            | —              |

Negritas en el mejor valor por columna (excluyendo supervisado). Segunda mejor subrayada.

---

### 6.5 Tabla de ablaciones

**Estructura compacta (solo D-Brain, solo PSNR-ROI y FA-error para brevedad):**

| Ablación     | Variante                | PSNR-ROI ↑ | FA-error ↓ |
| ------------ | ----------------------- | ---------- | ---------- |
| Sampling     | Sequential K=24         |            |            |
|              | RGS K=24 (propuesto)    |            |            |
| Tamaño K     | K=5                     |            |            |
|              | K=10                    |            |            |
|              | K=24                    |            |            |
| Arquitectura | DRCNet 3D               |            |            |
|              | Restormer 3D            |            |            |
|              | DRCNet 2D (ablación)    |            |            |
| Mask p       | 0.1                     |            |            |
|              | 0.3 (propuesto)         |            |            |
|              | 0.5                     |            |            |
| Training     | Estándar                |            |            |
|              | Progressive (propuesto) |            |            |

---

### 6.6 Tabla de tiempos y complejidad

**Estructura:**

| Método        | Parámetros (M) | GPU mem. train (GB) | Tiempo entrenamiento | Tiempo inferencia/vol (s) | Tiempo total vol. completo (min) |
| ------------- | -------------- | ------------------- | -------------------- | ------------------------- | -------------------------------- |
| MP-PCA        | —              | —                   | —                    |                           |                                  |
| P2S DIPY      | —              | —                   | —                    |                           |                                  |
| MD-S2S Conv2D |                |                     |                      |                           |                                  |
| DRCNet-RGS    |                |                     |                      |                           |                                  |
| Restormer-RGS |                |                     |                      |                           |                                  |

---

### 6.7 Sección de método — formalización J-invarianza en inferencia

**Qué escribir:** Un párrafo o subsección que explique explícitamente la pequeña inconsistencia entre training e inferencia (durante training los voxeles enmascarados son cero en el input; durante inferencia el volumen objetivo se ve parcialmente). El argumento estándar (compartido con Noise2Void, P2S, y S2S) es:

> Durante el entrenamiento, la red minimiza el riesgo esperado sobre la distribución de máscaras aleatorias, aprendiendo efectivamente el estimador de mínimo MSE de la señal limpia dado el contexto. En inferencia, al promediar múltiples passes con distintas máscaras y contextos (n_context, n_preds), la estimación converge al mismo estimador Bayes que el entrenamiento minimiza. La inconsistencia residual (voxeles que en inferencia el modelo ve pero durante training estaban enmascarados) contribuye con un sesgo de segundo orden menor que el ruido original, como demuestran Krull et al. (2020) y Batson & Royer (2019) en el contexto de Noise2Void.

Citar: Noise2Void (Krull 2020), J-invariance (Batson 2019), Patch2Self (Fadnavis 2020).

---

### 6.8 Sección de limitaciones

**Puntos a cubrir obligatoriamente:**

1. **Ruido sintético vs real:** Los experimentos principales usan ruido Rician sintético. El ruido real de escáner incluye artefactos estructurados (Gibbs ringing, motion artifacts, corrientes de Eddy) que violan las asunciones de independencia estadística de J-invarianza. El método puede introducir bias en presencia de ruido correlacionado espacialmente.

2. **Costo computacional de inferencia:** El costo O(V × n_context × n_preds × t_forward) hace que la inferencia RGS sea más costosa que MP-PCA o P2S para shells grandes. Para G=60, n_context=24, n_preds=12: ~17,000 forward passes. Cuantificar el tiempo exacto.

3. **Ausencia de GT real:** Stanford no tiene ground truth, limitando la evaluación cuantitativa a métricas downstream (FA/MD) que son indirectas.

4. **Dependencia del K óptimo:** El K óptimo puede variar con el protocolo de adquisición (b-value, número de gradientes, SNR base). El paper reporta K=? como recomendación para los datasets evaluados, pero puede requerir ajuste en otros protocolos.
