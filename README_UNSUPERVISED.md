# Informe de Aprendizaje No Supervisado (EA3)

**Asignatura:** Machine Learning \- Duoc UC **Proyecto:** Scoring de Riesgo Crediticio

## 1\. Descripción de la Técnica y Justificación

Para la exploración no supervisada del dataset *Home Credit*, se implementó una estrategia combinada de **Reducción de Dimensionalidad (PCA)** seguida de **Clustering (K-Means)**.

### Justificación de la elección:

1. **Alta Dimensionalidad:** El dataset original (`application_.parquet`) cuenta con más de 120 variables. Aplicar clustering directamente sobre este volumen de datos genera la "maldición de la dimensionalidad", donde las distancias euclidianas pierden sentido y el ruido dificulta la segmentación.  
2. **Multicolinealidad:** Existen muchas variables altamente correlacionadas (ej. distintos tipos de ingresos o promedios de montos). **PCA (Principal Component Analysis)** permitió compactar esta información, reteniendo el 95% de la varianza explicada y eliminando redundancia.  
3. **Segmentación Latente:** Se eligió **K-Means** sobre los componentes principales para identificar grupos homogéneos de clientes. El objetivo era descubrir si existen perfiles "naturales" de riesgo sin utilizar explícitamente la etiqueta de `TARGET` durante el entrenamiento del algoritmo.

---

## 2\. Instrucciones de Ejecución

El código se encuentra en el archivo `EA3_PRACTICO.py`.

**Requisitos previos:**

* Python 3.8+  
* Librerías: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyarrow`.

**Pasos para ejecutar:**

1. Asegúrese de que el archivo de datos `application_.parquet` se encuentre en la misma carpeta que el script.  
2. Ejecute el script desde la terminal:  
     
   python EA3_PRACTICO.py  
     
3. **Nota durante la ejecución:** El script mostrará primero una ventana con las gráficas de *Elbow* y *Silhouette*. **Debe cerrar esta ventana** para que el programa seleccione el mejor **k** automáticamente y proceda a generar el clustering final.

---

## 3\. Análisis e Interpretación de Resultados

### Determinación del número de Clusters (**k)**

Se utilizó una búsqueda iterativa de **k=2** a **k=8** evaluando dos métricas:

* **Silhouette Score:** Mostró su valor máximo (aprox **0.361**) con **k=2**. Esto indica que la estructura más fuerte y natural en los datos es una división binaria.  
* **Método del Codo (Inertia):** La curva presentó una caída suave sin un quiebre brusco, lo que, combinado con el Silhouette, confirmó la elección de 2 segmentos principales.

### Caracterización de los Segmentos

Tras aplicar K-Means con **k=2** sobre todo el dataset, se obtuvieron los siguientes perfiles de riesgo (cruzando *a posteriori* con la variable `TARGET`):

| Cluster | Cantidad Clientes | Tasa de Incumplimiento (Default Rate) | Perfil Observado |
| :---- | :---- | :---- | :---- |
| **Cluster 0** | 272,458 | **8.42%** | Grupo Mayoritario (Riesgo Estándar/Alto) |
| **Cluster 1** | 35,053 | **5.41%** | Grupo Minoritario (Bajo Riesgo) |

**Interpretación del Negocio:** El algoritmo logró identificar un segmento específico de clientes (**Cluster 1**, aprox. 11% de la cartera) que presenta un comportamiento de pago significativamente mejor que el promedio. Mientras que el grupo masivo tiene una tasa de mora del 8.42%, este subgrupo baja al 5.41%. Esto sugiere que existen características latentes (posiblemente mayor edad, estabilidad laboral o tipo de producto) que hacen a este grupo más "seguro" para la institución.

---

## 4\. Discusión: Integración al Proyecto Final

¿Es útil este método para el modelo supervisado de Scoring? **SÍ.**

**Estrategia de incorporación:**

1. **Feature Engineering:** Se añadirá la columna `Cluster_ID` como una nueva variable categórica en el dataset de entrenamiento.  
2. **Valor Predictivo:** Dado que hay una diferencia de casi **3 puntos porcentuales** en la tasa de riesgo entre ambos clusters, esta variable ayudará al modelo supervisado (XGBoost/LightGBM) a premiar o castigar el score dependiendo del segmento al que pertenezca el solicitante.  
3. **Prevención de Data Leakage:** Todo el proceso (ajuste de PCA y K-Means) se realizó utilizando **estrictamente el set de entrenamiento** (filas con TARGET disponible), garantizando la integridad de la evaluación final.

