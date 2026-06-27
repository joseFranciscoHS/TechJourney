**Objetivo General**

Validar que **Hybrid RGS es un framework self-supervised robusto para DWI denoising**, no solo una arquitectura específica.

La hipótesis principal sería:

> A fixed DRCNet backbone trained with the full Hybrid RGS objective outperforms angular-only, spatial-only, deterministic-subset, and non-random alternatives, indicating that the gain comes from the coupled random-gradient-subset and target-masking objective.

En español:

> Manteniendo DRCNet constante, Hybrid RGS debería superar variantes tipo Patch2Self neural, Self2Self/MD-S2S neural y sequential hybrid, demostrando que la contribución viene del esquema de entrenamiento y no solo del modelo.

---

# 1. Experimento Principal: Objective-Controlled DRCNet Ablation

Este es el experimento más importante.

## Objetivo

Separar el efecto de:

- contexto angular,
- masking espacial,
- random subset sampling,
- uso del mismo backbone DRCNet.

La comparación debe responder:

> Si uso exactamente DRCNet, ¿qué esquema self-supervised funciona mejor?

---

## Condiciones a correr

### A. `DRCNet Angular-Only`, tipo Patch2Self neural

**Nombre sugerido en paper:**

```text
DRCNet Angular-Only
```

o

```text
DRCNet Patch2Self-style
```

**Objetivo específico**

Probar si basta con usar una red neural para aprender la relación entre gradientes, sin masking espacial del target.

**Diseño**

Para cada target volume `t`:

- Input: subset de gradientes contextuales `C_t`.
- El target volume `y_t` **no debe entrar como canal de entrada**.
- Output: predicción de `y_t`.
- Loss: contra el volumen noisy `y_t`.
- No hay voxel masking del target porque el target no está en input.
- Esto replica conceptualmente Patch2Self, pero reemplaza OLS/linear regression por DRCNet.

**Configuración recomendada**

```text
Backbone: DRCNet
Input channels: K-1 context gradients
Target channel included: no
Target masking: no
Context selection: random or leave-one-out subset
Loss: full-volume L1 or L2 against y_t
K: 16, para comparable con Hybrid RGS
```

**Importante**

Aquí hay dos posibles variantes:

1. **Angular-only random subset**  
   Usa `K-1` gradientes aleatorios como contexto.
2. **Angular-only all-available**  
   Usa todos los gradientes excepto `t`, si cabe en memoria.

Para fair comparison con Hybrid RGS, recomiendo empezar con:

```text
Angular-only random subset, K=16
```

porque iguala el presupuesto de canales.

**Qué demostraría**

Si este modelo funciona peor que Hybrid RGS, entonces:

> Angular context alone is insufficient; target-channel spatial masking adds useful local anatomical information.

**Interpretación esperada**

- Puede preservar bien estructura angular.
- Puede ser más débil en PSNR/SSIM porque no ve el target volume localmente.
- Puede ser competitivo en MD/FA si aprende relaciones entre gradientes.

---

### B. `DRCNet Spatial-Only`, tipo Self2Self / MD-S2S neural

**Nombre sugerido en paper:**

```text
DRCNet Spatial-Only
```

o

```text
DRCNet Self2Self-style
```

**Objetivo específico**

Probar si basta con masking espacial del target sin explotar contexto angular.

**Diseño**

Para cada target volume `t`:

- Input: solo el target volume parcialmente masked `\tilde{y}_t`.
- No usar otros gradientes como contexto.
- Output: predicción de `y_t`.
- Loss: solo en voxels masked.
- Es una versión volumétrica de Self2Self/MD-S2S usando DRCNet.

**Configuración recomendada**

```text
Backbone: DRCNet
Input channels: 1
Target channel included: yes, masked
Angular context: no
Target masking: yes
Mask probability p: same as Hybrid RGS
Loss: masked-only L1 or L2
K: not applicable
```

**Variante opcional**

Si el código exige `K` canales, puedes usar:

```text
Input = masked target + zero-filled dummy channels
```

pero es más limpio implementar un input de 1 canal.

**Qué demostraría**

Si este modelo funciona peor que Hybrid RGS, entonces:

> Spatial blind-spot masking alone is insufficient; angular diffusion context provides additional information.

**Interpretación esperada**

- Puede mejorar ruido local.
- Riesgo de oversmoothing.
- Puede tener peor preservación de dirección/tensor porque no ve otros gradientes.

---

### C. `DRCNet Sequential Hybrid`

**Nombre sugerido en paper:**

```text
DRCNet Sequential Hybrid
```

**Objetivo específico**

Aislar el valor de la aleatoriedad en el sampling angular.

**Diseño**

Igual que Hybrid RGS, pero el contexto no se samplea aleatoriamente.

Para cada target `t`:

- Input: masked target `\tilde{y}_t` + `K-1` context gradients.
- Context selection: ventana secuencial/determinística.
- Loss: masked-only.
- No shuffle, o shuffle controlado si quieres separar dos factores.

**Configuración recomendada**

```text
Backbone: DRCNet
Input channels: K
Target channel included: yes, masked
Angular context: yes
Target masking: yes
Context selection: sequential deterministic
K: 16
Mask probability p: same as Hybrid RGS
Loss: masked-only
```

**Qué demostraría**

Si Hybrid RGS random supera sequential hybrid:

> Random gradient subsets provide useful angular augmentation and avoid overfitting to fixed channel neighborhoods.

**Interpretación esperada**

- Sequential puede funcionar, pero menos robusto.
- Puede depender de orden de gradientes.
- Puede tener menor diversidad angular efectiva.

---

### D. `DRCNet Hybrid RGS`, método propuesto

**Nombre sugerido en paper:**

```text
DRCNet Hybrid RGS
```

**Objetivo específico**

Condición completa propuesta.

**Diseño**

Para cada target `t`:

- Input: masked target `\tilde{y}_t` + random subset `C_t` de otros gradientes.
- Context gradients excluyen el target.
- Loss: masked-only.
- Contexto random en training.
- Múltiples contextos opcionales en inference.

**Configuración recomendada**

```text
Backbone: DRCNet
Input channels: K
Target channel included: yes, masked
Angular context: yes
Target masking: yes
Context selection: random subset
K: 16
Mask probability p: same across all masking experiments
Loss: masked-only
```

**Qué demostraría**

Esta debería ser la condición principal.

Si supera A, B y C:

> The combination of angular context, spatial target masking, and random subset sampling is better than each component alone.

---

## Tabla principal recomendada

En el paper, esta tabla debería verse así:

```latex
\begin{table}[ht]
  \centering
  \caption{Objective-controlled DRCNet ablation on D-Brain with Rician noise $\sigma=0.1$. All rows use the same DRCNet backbone, training budget, normalization, and evaluation protocol. The ablation isolates angular context, target-channel masking, and random gradient subset sampling.}
  \label{tab:objective_controlled_ablation}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{lcccccccc}
    \toprule
    Training objective & Angular context & Target masking & Random subset & PSNR-ROI $\uparrow$ & SSIM-ROI $\uparrow$ & FA-MAE $\downarrow$ & MD-MAE $\downarrow$ & Time/vol. \\
    \midrule
    DRCNet Angular-Only & Yes & No & Yes & -- & -- & -- & -- & -- \\
    DRCNet Spatial-Only & No & Yes & No & -- & -- & -- & -- & -- \\
    DRCNet Sequential Hybrid & Yes & Yes & No & -- & -- & -- & -- & -- \\
    DRCNet Hybrid RGS & Yes & Yes & Yes & -- & -- & -- & -- & -- \\
    \bottomrule
  \end{tabular}
  }
\end{table}
```

---

# 2. Experimento de Randomization: Random vs Sequential vs Angular Neighbors

Este experimento robustece la idea de que **random gradient subset** no es solo una decisión computacional.

## Objetivo

Probar qué tipo de selección angular es mejor.

La pregunta:

> ¿Conviene seleccionar gradientes aleatoriamente, secuencialmente, vecinos angulares cercanos, o gradientes angularmente diversos?

---

## Condiciones

### A. Sequential subset

```text
Context = ventana fija alrededor del índice target o secuencia ordenada.
```

Sirve como baseline determinístico.

### B. Random uniform subset

```text
Context = K-1 gradientes aleatorios entre todos los no-target.
```

Es Hybrid RGS estándar.

### C. Nearest angular neighbors

```text
Context = K-1 gradientes con menor ángulo respecto al target.
```

Objetivo: ver si los gradientes más parecidos al target ayudan más.

### D. Farthest/diverse angular subset

```text
Context = gradientes distribuidos angularmente, maximizando diversidad.
```

Objetivo: ver si diversidad angular mejora generalización.

---

## Métricas clave

- PSNR-ROI
- SSIM-ROI
- FA-MAE
- MD-MAE
- runtime
- varianza entre seeds si puedes

---

## Interpretación

Si random uniform gana o empata:

> Random RGS is a strong practical default because it provides angular diversity without needing optimized sampling.

Si nearest neighbors gana:

> El framework puede mejorarse usando sampling informado por geometría angular.

Si diverse sampling gana:

> La diversidad angular es un factor clave y puede motivar una versión “Angular-Diverse RGS”.

---

# 3. Experimento de `K`: Context Size Sweep

Ya tienes parte de esto, pero conviene formularlo claramente.

## Objetivo

Determinar cuánto contexto angular necesita RGS y cuál es el tradeoff calidad/costo.

Pregunta:

> ¿Más gradientes siempre ayudan o existe un punto óptimo?

---

## Condiciones recomendadas

```text
K ∈ {1, 5, 10, 16, 24, 30}
```

Donde:

- `K=1`: spatial-only, solo target masked.
- `K=5`: contexto angular bajo.
- `K=10`: contexto moderado.
- `K=16`: default principal.
- `K=24`: contexto alto.
- `K=30`: contexto muy alto.

---

## Configuración fija

```text
Backbone: DRCNet
Training objective: Hybrid RGS
Mask probability p: fixed
Dataset: D-Brain sigma=0.1
Training steps: same or normalized by compute budget
Inference contexts N_c: fixed
```

---

## Qué buscar

- PSNR-ROI sube hasta cierto punto y luego se estabiliza o baja.
- FA-MAE puede no seguir PSNR.
- Runtime aumenta con `K`.
- Parámetros pueden aumentar si la primera capa depende de canales.

---

## Claim posible

> Moderate context sizes such as $K=16$ provide a favorable quality--cost tradeoff, while larger stacks do not guarantee better diffusion-derived metrics.

---

# 4. Experimento de Mask Probability `p`

Este falta y sería muy útil.

## Objetivo

Probar que Hybrid RGS no depende de un valor frágil de masking.

Pregunta:

> ¿El método es estable frente a distintas proporciones de voxels enmascarados?

---

## Condiciones

```text
p ∈ {0.05, 0.10, 0.20, 0.30, 0.50, 0.70}
```

Pero para reducir costo, puedes correr:

```text
p ∈ {0.10, 0.20, 0.30, 0.50}
```

---

## Configuración fija

```text
Backbone: DRCNet
K: 16
Context selection: random
Dataset: D-Brain sigma=0.1
Loss: masked-only
Inference N_c: fixed
```

---

## Interpretación esperada

- `p` muy bajo: poca señal self-supervised, entrenamiento débil.
- `p` moderado: mejor balance.
- `p` muy alto: se oculta demasiado target, puede degradar reconstrucción.
- Un rango estable fortalece el framework.

---

## Claim posible

> Hybrid RGS is not sensitive to a narrow masking probability; moderate masking rates preserve the blind-spot constraint while retaining sufficient spatial context.

---

# 5. Experimento de Inference Context Averaging `N_c`

Muy bueno para mostrar tradeoff práctico.

## Objetivo

Evaluar si promediar múltiples subsets random en inference mejora estabilidad/calidad.

Pregunta:

> ¿Cuántos contextos aleatorios conviene promediar al reconstruir cada volumen?

---

## Condiciones

```text
N_c ∈ {1, 3, 5, 12, 24}
```

Opcional:

```text
N_c = 48
```

si no es demasiado caro.

---

## Configuración fija

```text
Backbone: DRCNet
K: 16
p: fixed
Training: Hybrid RGS
Inference: average over N_c random contexts
```

---

## Qué medir

- PSNR-ROI
- SSIM-ROI
- FA-MAE
- MD-MAE
- Time/volume
- Standard deviation across stochastic predictions

---

## Interpretación esperada

- Mejora de `N_c=1` a `N_c=3/5`.
- Saturación después.
- Runtime aumenta casi linealmente.

---

## Claim posible

> A small number of inference contexts captures most of the benefit of stochastic RGS averaging, while larger $N_c$ mainly increases cost.

---

# 6. Experimento de Backbone Transfer

Este es útil, pero después del DRCNet-controlled ablation.

## Objetivo

Mostrar que RGS no depende exclusivamente de DRCNet.

Pregunta:

> ¿El mismo objetivo funciona con diferentes backbones?

---

## Condiciones mínimas

```text
DRCNet Hybrid RGS
Restormer Hybrid RGS
2D DRCNet Hybrid RGS
2D Restormer Hybrid RGS
```

Pero para no gastar mucho, puedes hacer:

```text
DRCNet Hybrid RGS
2D DRCNet Hybrid RGS
Restormer Hybrid RGS
```

---

## Importante

No necesitas que Restormer gane. De hecho, si no gana, igual sirve para el argumento:

> Hybrid RGS is architecture-agnostic, but architecture choice remains an empirical design variable.

---

## Qué reportar

- PSNR-ROI
- SSIM-ROI
- FA-MAE
- MD-MAE
- Params
- Time/volume

---

## Claim posible

> The Hybrid RGS objective transfers across compact convolutional, transformer-style, and two-dimensional variants, although the most practical backbone in the current experiments is DRCNet.

---

# 7. Experimento de Noise Robustness

Ya lo tienes parcialmente, pero conviene estructurarlo.

## Objetivo

Probar estabilidad del framework frente a distintos SNRs.

Pregunta:

> ¿Hybrid RGS degrada suavemente cuando aumenta el ruido?

---

## Condiciones

```text
sigma ∈ {0.05, 0.10, 0.15, 0.20}
```

---

## Configuración

Para cada sigma:

```text
Train from scratch at that sigma.
Evaluate at same sigma.
```

Esto prueba robustez del training protocol.

Opcional más fuerte:

```text
Train at sigma=0.10.
Evaluate at sigma={0.05,0.15,0.20}.
```

Esto prueba cross-noise generalization.

---

## Comparaciones

- Noisy
- MP-PCA
- Patch2Self
- MD-S2S
- DRCNet Hybrid RGS

---

## Claim posible

> Hybrid RGS shows graceful degradation across Rician noise levels, while MP-PCA remains a strong FA-preserving baseline under synthetic noise.

---

# 8. Experimento Real-Scanner: Stanford HARDI

Aquí hay que ser muy cuidadoso porque no hay ground truth.

## Objetivo

Demostrar factibilidad en datos reales, no accuracy cuantitativa.

Pregunta:

> ¿Se puede entrenar Hybrid RGS directamente en real scanner data y obtener mapas plausibles?

---

## Condiciones

```text
Noisy input
Patch2Self
MD-S2S
MP-PCA, si disponible
DRCNet Hybrid RGS
Restormer Hybrid RGS, opcional
```

---

## Qué NO decir

No decir:

```text
Hybrid RGS improves restoration accuracy on Stanford.
```

porque no hay ground truth.

---

## Qué sí decir

```text
Hybrid RGS can be trained directly on Stanford HARDI without clean references and produces qualitatively plausible FA/MD maps.
```

---

## Evaluaciones recomendadas

### A. Visual FA/MD maps

Mostrar:

- noisy,
- MP-PCA,
- Patch2Self,
- MD-S2S,
- Hybrid RGS.

### B. Residual maps

Mostrar:

```text
input - denoised
```

Objetivo: verificar que el residual parezca ruido, no anatomía.

### C. Histogramas

Histogramas de:

- FA en WM/GM/CSF si tienes máscaras,
- MD,
- intensidad residual.

### D. Tensor summary stability

Si puedes:

- media FA en corpus callosum,
- internal capsule,
- centrum semiovale,
- CSF,
- gray matter.

---

# 9. ROI-Based Tensor Evaluation

Este sería muy útil para que el paper no dependa solo de PSNR.

## Objetivo

Validar que el denoising no mejora imagen a costa de distorsionar biomarcadores.

Pregunta:

> ¿Los métodos preservan FA/MD en regiones anatómicas importantes?

---

## ROIs recomendadas

Si tienes atlas o máscaras:

```text
Corpus callosum
Internal capsule
Corona radiata
Corticospinal tract region
CSF / ventricles
Gray matter
Whole white matter
```

Si no tienes atlas, usa máscaras simples:

```text
High-FA white matter mask
Low-FA gray matter-like mask
High-MD CSF-like mask
Whole-brain mask
```

---

## Métricas

Para cada ROI:

- mean FA,
- std FA,
- mean MD,
- std MD,
- absolute error vs reference si hay ground truth,
- coefficient of variation si no hay ground truth.

---

## Claim posible

> ROI-level tensor summaries reveal whether image-domain improvements translate into anatomically plausible diffusion biomarkers.

---

# 10. Residual Anatomical Leakage Test

Este es simple y potente para real data.

## Objetivo

Verificar que el método no esté removiendo estructura anatómica.

Pregunta:

> ¿El residual contiene principalmente ruido o también bordes/anatomía?

---

## Procedimiento

Para cada método:

```text
residual = noisy - denoised
```

Evaluar:

- visual residual slices,
- correlation residual vs denoised image,
- edge energy in residual,
- residual variance inside homogeneous ROI.

---

## Interpretación

Buen denoising:

```text
Residual looks noise-like.
Low anatomical edges.
Low correlation with denoised image.
```

Mal denoising:

```text
Residual contains tissue boundaries or tract structure.
```

---

# 11. Test-Retest o Repeated Acquisition, si existe

Este sería el más fuerte para journal.

## Objetivo

Validar real-world quantitative stability sin clean ground truth.

Pregunta:

> ¿El denoising mejora la reproducibilidad entre dos adquisiciones del mismo sujeto?

---

## Diseño

Si tienes dos scans del mismo sujeto:

```text
scan A noisy
scan B noisy
denoise A
denoise B
compare tensor maps A vs B
```

---

## Métricas

- FA test-retest error,
- MD test-retest error,
- voxelwise correlation,
- ROI coefficient of variation,
- Bland–Altman plots.

---

## Claim posible

> Hybrid RGS improves test-retest stability of diffusion-derived measures without requiring clean references.

Este sería un argumento muy fuerte.

---

# 12. Prioridad Recomendada de Corridas

Si tienes tiempo limitado, yo haría esto en orden.

## Prioridad Alta

Estas son las más importantes para defender el framework.

```text
1. DRCNet Angular-Only
2. DRCNet Spatial-Only
3. DRCNet Sequential Hybrid
4. DRCNet Hybrid RGS
```

Con esto puedes armar la tabla central.

---

## Prioridad Media

```text
5. K sweep
6. Mask probability sweep
7. Inference context sweep
8. Noise-level robustness
```

Esto da robustez metodológica.

---

## Prioridad Alta si tienes real data

```text
9. Stanford qualitative FA/MD
10. Residual maps
11. ROI tensor summaries
```

Esto ayuda a que no parezca solo synthetic benchmark.

---

## Prioridad Opcional

```text
12. Restormer objective-controlled ablation
13. Angular-nearest vs angular-diverse sampling
14. Test-retest, si hay datos
```

---

# 13. Configuración Común Para Todas Las Corridas

Para que las comparaciones sean defendibles, todas las condiciones principales deben compartir:

```text
Dataset: D-Brain
Noise: Rician sigma=0.1
Train/val/test split: identical
Normalization: identical
Patch size: identical
Batch size: as close as possible
Optimizer: identical
Learning rate: identical
Training steps/epochs: identical
Random seed: fixed or repeated
Backbone: DRCNet for objective-controlled ablation
K: 16 unless being swept
Mask probability p: fixed unless being swept
Inference N_c: fixed unless being swept
Metrics: same scripts and same brain mask
```

Si alguna condición necesita batch size distinto por memoria, repórtalo.

---

# 14. Nombres de Experimentos Recomendados

Para mantener orden:

```text
dbrain_sigma01_drcnet_angular_only_k16
dbrain_sigma01_drcnet_spatial_only_p02
dbrain_sigma01_drcnet_sequential_hybrid_k16_p02
dbrain_sigma01_drcnet_hybrid_rgs_k16_p02
dbrain_sigma01_drcnet_hybrid_rgs_k05_p02
dbrain_sigma01_drcnet_hybrid_rgs_k10_p02
dbrain_sigma01_drcnet_hybrid_rgs_k24_p02
dbrain_sigma01_drcnet_hybrid_rgs_p01
dbrain_sigma01_drcnet_hybrid_rgs_p03
dbrain_sigma01_drcnet_hybrid_rgs_nc01
dbrain_sigma01_drcnet_hybrid_rgs_nc05
stanford_drcnet_hybrid_rgs_k16_p02
```

---

# 15. Qué Debe Guardar Cada Corrida

Cada experimento debería guardar:

```text
config.yaml / config.json
checkpoint best
training loss curve
validation self-supervised loss
denoised volumes
FA/MD/AD/RD maps
PSNR/SSIM/MSE full image
PSNR/SSIM/MSE ROI
FA-MAE
MD-MAE
AD-MAE
RD-MAE
runtime per volume
GPU memory peak
random seed
```

Para reproducibilidad, también:

```text
git commit hash, si aplica
dataset ID
subject IDs
noise sigma
mask probability
K
N_c
sampling strategy
```

---

# 16. Interpretación Final Esperada

La narrativa ideal quedaría así:

1. **Angular-only DRCNet** muestra que usar otros gradientes ayuda, pero no basta.
2. **Spatial-only DRCNet** muestra que masking local ayuda, pero pierde contexto angular.
3. **Sequential Hybrid** muestra que combinar ambas señales ayuda.
4. **Hybrid RGS** muestra que random subset sampling mejora la combinación.
5. **K/p/Nc sweeps** muestran que el método no depende de un hiperparámetro frágil.
6. **Backbone transfer** muestra que no es un truco de DRCNet.
7. **MP-PCA sigue fuerte en FA**, lo cual hace la evaluación honesta.
8. **Stanford HARDI** muestra factibilidad real-scanner, no accuracy absoluta.

---

# 17. Claim Que Podrías Escribir Después

Si los resultados salen como esperamos, el claim fuerte sería:

```text
Using the same DRCNet backbone, Hybrid RGS outperformed angular-only and spatial-only self-supervised objectives, indicating that the improvement is not simply attributable to network capacity. The results support Hybrid RGS as an architecture-agnostic training framework that couples random angular context with target-channel masking.
```

En español:

> Usando el mismo backbone DRCNet, Hybrid RGS supera variantes angular-only y spatial-only, lo que indica que la mejora no se debe simplemente a capacidad de red. Los resultados apoyan Hybrid RGS como un framework de entrenamiento architecture-agnostic que combina contexto angular aleatorio con target-channel masking.

---

Mi recomendación concreta: corre primero las **4 condiciones DRCNet-controlled**. Esa tabla sería la evidencia más importante para defender el paper como framework.
