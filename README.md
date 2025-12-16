# PROYECTO EXAMEN: SISTEMA DE SCORING CREDITICIO (HOME CREDIT)
Asignatura: Machine Learning
Duoc UC - 2025

DESCRIPCION DEL PROYECTO
Este proyecto implementa una solución "End-to-End" para predecir la probabilidad de 
impago de clientes financieros. El sistema realiza una integracion masiva de 6 fuentes 
de datos (vision 360 del cliente), aplica ingeniería de características avanzada 
(polinomica y de dominio) y despliega un modelo XGBoost optimizado a través de una API.

ESTRUCTURA DEL PROYECTO (CRISP-DM)
- 01_data_understanding: Scripts para analisis exploratorio (EDA) focalizado.
- 02_data_preparation: Pipeline ETL complejo que une 6 tablas (App, Bureau, Prev, POS, Installments, CC).
- 03_modeling: Entrenamiento del modelo con hiperparametros ajustados y manejo de desbalance.
- 04_evaluation: Generacion de reportes visuales (Confusion Matrix, ROC, Feature Importance).
- 05_deployment: Interfaz web interactiva (Streamlit) para evaluacion en tiempo real.
- artifacts: Almacenamiento de modelos serializados (.joblib) y lista de features.

INSTRUCCIONES DE INSTALACION Y EJECUCION
IMPORTANTE: Todos los comandos deben ejecutarse desde la carpeta raiz del proyecto "ML".

1. INSTALACION DE DEPENDENCIAS
Ejecute el siguiente comando para instalar las librerias necesarias:
pip install -r requirements.txt

2. COMPRENSION DE DATOS
Genere un reporte basico de los datos originales:
python 01_data_understanding/eda_focalizado.py

3. PREPARACION DE DATOS (ETL MASIVO)
Ejecute el proceso para integrar las 6 tablas, calcular ratios financieros y generar features polinomicas:
python 02_data_preparation/make_dataset.py

Output: Crea el archivo maestro '02_data_preparation/master_table.parquet' (+400 columnas).

4. ENTRENAMIENTO DEL MODELO
Entrene el modelo XGBoost (Configuracion Hardcore: 500 arboles + Regularizacion):
python 03_modeling/train_model.py

Output: Guarda el modelo optimizado en la carpeta artifacts.

5. EVALUACION DEL MODELO
Genere los graficos de rendimiento actualizados:
python 04_evaluation/evaluate_model.py

Output: Guarda las imagenes en la carpeta 04_evaluation.

6. DESPLIEGUE DE LA APLICACION
Inicie la interfaz web para realizar predicciones:
streamlit run 05_deployment/app.py

RESUMEN TECNICO
- Modelo: XGBoost Classifier (Tuned)
- Estrategia de Desbalance: Scale_pos_weight dinamico
- Validacion: Split Estratificado (80/20) y metrica AUC-ROC
- Datos: Integracion de +300.000 clientes con historiales de pago, tarjetas y buro externo.
- Ingenieria de Caracteristicas: +400 variables incluyendo Ratios Financieros, Interacciones Polinomicas (EXT_SOURCE) y Comportamiento de Pago (Installments).