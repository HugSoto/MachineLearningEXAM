# PROYECTO EXAMEN: MODELO DE RIESGO DE CREDITO (HOME CREDIT)
Asignatura: Machine Learning
Duoc UC - 2025

DESCRIPCION DEL PROYECTO
Este proyecto implementa una solución completa para predecir la probabilidad de 
impago de clientes financieros. El sistema integra múltiples fuentes de datos, 
aplica ingeniería de características avanzada y despliega un modelo XGBoost 
a través de una interfaz interactiva.

ESTRUCTURA DEL PROYECTO (CRISP-DM)
- 01_data_understanding: Scripts para analisis exploratorio de datos.
- 02_data_preparation: Scripts de limpieza e ingenieria de caracteristicas.
- 03_modeling: Scripts para entrenamiento del modelo.
- 04_evaluation: Generacion de graficos y metricas de desempeño.
- 05_deployment: Codigo de la aplicacion web (API).
- artifacts: Almacenamiento de modelos entrenados y metadatos.

INSTRUCCIONES DE INSTALACION Y EJECUCION
IMPORTANTE: Todos los comandos deben ejecutarse desde la carpeta raiz del proyecto "ML".

1. INSTALACION DE DEPENDENCIAS
Ejecute el siguiente comando para instalar las librerias necesarias:
pip install -r requirements.txt

2. COMPRENSION DE DATOS
Genere un reporte basico de los datos originales:
python 01_data_understanding/eda_focalizado.py

3. PREPARACION DE DATOS
Ejecute el proceso ETL para integrar tablas y calcular variables financieras:
python 02_data_preparation/make_dataset.py

Output: Crea el archivo 02_data_preparation/master_table.parquet

4. ENTRENAMIENTO DEL MODELO
Entrene el modelo XGBoost con manejo de desbalance de clases:
python 03_modeling/train_model.py

Output: Guarda el modelo y las features en la carpeta artifacts

5. EVALUACION DEL MODELO
Genere los graficos de rendimiento (Matriz de Confusion, Curva ROC):
python 04_evaluation/evaluate_model.py

Output: Guarda las imagenes en la carpeta 04_evaluation

6. DESPLIEGUE DE LA APLICACION
Inicie la interfaz web para realizar predicciones:
streamlit run 05_deployment/app.py

RESUMEN TECNICO
- Modelo: XGBoost Classifier
- Estrategia de Desbalance: Scale_pos_weight
- Validacion: Split Estratificado (80/20) y metrica AUC-ROC
- Ingenieria de Caracteristicas: Agregaciones historicas y ratios financieros (Credit/Income)