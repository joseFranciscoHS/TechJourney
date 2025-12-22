Este reporte resume los fundamentos teóricos y las aplicaciones prácticas de la **$J$-invariancia** en los métodos de eliminación de ruido para imágenes de resonancia magnética de difusión (DWI), basándose en la literatura técnica proporcionada.

### 1. El Concepto de $J$-invariancia
La **$J$-invariancia** es un marco teórico general para la eliminación de ruido en mediciones de alta dimensión que no requiere datos de entrenamiento limpios ni un modelo explícito del ruido. Una función se define como $J$-invariante si la predicción realizada para un subconjunto de dimensiones ($J$) no depende de los valores ruidosos de entrada en esas mismas dimensiones. 

El principio fundamental es que si el ruido es estadísticamente independiente entre las dimensiones (como los píxeles o los volúmenes), pero la señal verdadera está correlacionada, minimizar el error entre la predicción y el dato ruidoso equivale a minimizar el error respecto a la señal limpia. Esto permite calibrar o entrenar algoritmos utilizando únicamente los datos ruidosos como su propia supervisión.

---

### 2. Self2Self (S2S) y la $J$-invariancia
El método **Self2Self** aplica la $J$-invariancia a nivel de píxel para permitir el entrenamiento de una red neuronal utilizando **una sola imagen ruidosa**.

*   **Relación con la $J$-invariancia:** S2S garantiza esta propiedad mediante el uso de **máscaras de Bernoulli**. Al ocultar píxeles aleatorios en la entrada y obligar a la red a predecirlos, se asegura que la predicción de un píxel dependa solo de sus vecinos y no de su propio valor ruidoso.
*   **Ideas Principales de Self2Self:**
    *   **Estimador de Bayes:** Interpreta la red como un estimador cuya precisión se mide por el Error Cuadrático Medio (MSE), donde el principal desafío es reducir la varianza causada por entrenar con una sola muestra.
    *   **Conjunto basado en Dropout (Dropout Ensemble):** La innovación clave es usar *dropout* tanto en el entrenamiento como en el test. Al promediar múltiples predicciones con diferentes configuraciones de *dropout*, se reduce drásticamente la varianza y se mejora la calidad de la imagen.
    *   **Uso de Convoluciones Parciales:** Implementa capas de convolución parcial para normalizar los píxeles muestreados, mejorando la eficiencia del entrenamiento.

---

### 3. Patch2Self y la $J$-invariancia
**Patch2Self** traslada el concepto de $J$-invariancia al dominio de los datos de DWI en 4D, aprovechando la redundancia entre diferentes adquisiciones.

*   **Relación con la $J$-invariancia:** En este caso, la partición $J$ se define por **volúmenes completos** (direcciones de gradiente). El modelo es $J$-invariante por construcción porque utiliza información de todos los volúmenes ($v_{-j}$) para predecir el contenido de un volumen objetivo ($v_j$), sin que este último participe en la entrada.
*   **Ideas Principales de Patch2Self:**
    *   **Independencia del Ruido:** Se asume que el ruido en una adquisición es independiente de las otras, lo que garantiza que la red aprenda solo la estructura anatómica coherente.
    *   **Regresión Lineal Local:** Originalmente propuesto como un regresor lineal que utiliza parches espaciales para capturar la relación entre volúmenes.
    *   **Análisis Multidimensional:** Aprovecha que las mismas estructuras biológicas están representadas en todos los volúmenes, permitiendo una separación efectiva entre señal y ruido sin un modelo de ruido explícito.

---

### 4. Esquemas de Entrenamiento Propuestos
Para el trabajo de investigación, se proponen dos esquemas que generalizan y fusionan estas ideas utilizando Redes Neuronales (NN) y Convoluciones 3D (Conv3D).

#### Esquema 1: Patch2Self Generalizado (Basado en Volúmenes)
Este esquema utiliza la arquitectura de una NN para mejorar la regresión local original de Patch2Self.
*   **Entrada:** Un conjunto de volúmenes de entrada $v_{-j}$ (por ejemplo, 59 volúmenes de un dataset de 60).
*   **Mecanismo:** La red utiliza Conv3D para extraer características espaciales y angulares de los 59 volúmenes para predecir el volumen $v_j$ completo.
*   **Objetivo:** Al ser $J$-invariante por volumen, la red suprime el ruido independiente de cada adquisición mientras preserva la señal estructural común.

#### Esquema 2: Estilo MD-S2S (Híbrido Espacio-Angular)
Este esquema combina la redundancia angular de Patch2Self con la redundancia espacial de MD-S2S (Multidimensional Self2Self).
*   **Entrada:** Se utilizan los volúmenes $v_{-j}$ completos más el volumen objetivo $v_j$ al cual se le aplica una **máscara de Bernoulli** a nivel de píxel.
*   **Mecanismo:** La red (con Conv3D) debe predecir los valores de los píxeles ocultos en $v_j$ utilizando tanto la información coherente de los otros 59 volúmenes como el contexto espacial de los píxeles visibles en el propio $v_j$.
*   **Objetivo:** Este enfoque híbrido es más potente porque permite a la red capturar detalles anatómicos finos que podrían variar ligeramente entre gradientes, manteniendo la $J$-invariancia estricta sobre los píxeles objetivo.

***

**Metáfora de cierre:**
El **Esquema 1** es como pedirle a un grupo de personas que describan una habitación que tú no puedes ver; el resultado es una visión general confiable. El **Esquema 2** es como si tú pudieras ver la habitación a través de una rejilla; utilizas lo que te dicen los demás **y** lo que ves por los huecos de la rejilla para reconstruir la imagen con una precisión mucho mayor.