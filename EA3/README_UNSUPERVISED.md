# Informe de Aprendizaje No Supervisado (EA3)

**Asignatura:** Machine Learning - Duoc UC
**Proyecto:** Scoring de Riesgo Crediticio
**Metodología:** CRISP-DM

---

## 1. Introducción y Enfoque Metodológico (CRISP-DM)

Siguiendo la lógica del proceso **CRISP-DM**, se ha desarrollado una fase exploratoria de **Modelado No Supervisado** para complementar el modelo predictivo principal.

### Selección de la Técnica
Se optó por una estrategia híbrida que combina dos de las sugerencias del enunciado:
1.  **Reducción de Dimensionalidad (PCA):** Para mitigar la redundancia entre variables correlacionadas.
2.  **Clustering (K-Means):** Para segmentar clientes basándose en sus componentes principales.

### Justificación de la Elección
El dataset *Home Credit* presenta desafíos que justifican este enfoque:
* **Alta Dimensionalidad:** Con más de 120 variables, el cálculo de distancias para clustering se ve afectado por la "maldición de la dimensionalidad". PCA permite compactar la varianza en menos dimensiones, haciendo el clustering más robusto.
* **Detección de Patrones Latentes:** Se busca identificar si existen subgrupos de clientes ("Clusters") que compartan comportamientos financieros intrínsecos, invisibles al analizar variables por separado.

> **Nota sobre Data Leakage:** Cumpliendo estrictamente con las buenas prácticas, todo el entrenamiento de PCA y K-Means se realizó **exclusivamente sobre el set de entrenamiento**, sin "ver" datos de validación o prueba.

---

## 2. Instrucciones de Ejecución

El código fuente se encuentra en el archivo `EA3_PRACTICO.py`.

**Requisitos previos:**
* Python 3.8+
* Librerías: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyarrow`.

**Pasos para ejecutar:**
1. Se recomienda instalar las dependencias exactas para evitar errores con la lectura del formato Parquet:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn pyarrow
    ```
2.  Asegúrese de que el archivo de datos `application_.parquet` esté en la misma carpeta que el script.
3.  Ejecute el script desde la terminal:
    ```bash
    python EA3_PRACTICO.py
    ```
4.  **Importante:** El script generará visualizaciones interactivas (Curva de Codo y Silhouette). **Debe cerrar la ventana del gráfico** para que el programa continúe con el cálculo final de los clusters y la generación de estadísticas.

---

## 3. Análisis e Interpretación de Resultados

### Determinación del número óptimo de Clusters (k)
Se evaluó un rango de **k=2** a **k=8**.
* **Métrica Silhouette:** Alcanzó su máximo (**~0.361**) con **k=2**, sugiriendo una estructura binaria natural en los datos.
* **Inertia (Codo):** No mostró un quiebre brusco, lo que refuerza la decisión de guiarse por el Silhouette Score para evitar sobre-segmentar ruido.

### Caracterización de los Segmentos (Vinculación con el Negocio)
Tras aplicar la segmentación, se cruzaron los clusters con la variable objetivo `TARGET` (tasa de morosidad) para validar su utilidad:

| Cluster | Perfil Identificado | Cantidad Clientes | Tasa de Incumplimiento (Default Rate) | Diferencia vs Promedio |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | Riesgo Estándar/Alto | 272,458 | **8.42%** | +0.35% |
| **Cluster 1** | **Bajo Riesgo** | 35,053 | **5.41%** | **-2.66%** |

**Hallazgo:** El método no supervisado logró aislar exitosamente un segmento (Cluster 1, aprox. 11% de la muestra) cuyo riesgo es significativamente menor al del resto de la cartera. Esto valida que la combinación de variables comprimidas por PCA contiene información predictiva real sobre el comportamiento de pago.

---

## 4. Discusión e Integración al Proyecto Final

**¿Es recomendable incorporar este método al modelo supervisado?**
**SÍ.**

### Argumentación Técnica
1.  **Feature Engineering:** La variable `Cluster_ID` se incorporará como una nueva característica categórica (o *meta-feature*) en el modelo supervisado (XGBoost/LightGBM).
2.  **Captura de No-Linealidad:** Al usar PCA + K-Means, estamos capturando interacciones complejas y no lineales entre las variables originales. Entregarle esta información pre-procesada al modelo supervisado le ayuda a distinguir más fácilmente entre perfiles de riesgo.
3.  **Valor de Negocio:** La diferencia de casi **3 puntos porcentuales** en la tasa de default entre clusters demuestra que esta variable tiene un alto poder discriminante ("Information Value"), lo que debería mejorar métricas como el AUC-ROC en el modelo final.