import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# CORRECCIÓN: Rutas ajustadas para ejecutar desde la carpeta ML
DATA_PATH = "02_data_preparation/master_table.parquet"
ARTIFACTS_DIR = "artifacts"

def train():
    print("Iniciando Entrenamiento del Modelo...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra {DATA_PATH}")
        print("Verifica que hayas ejecutado el paso anterior (make_dataset.py)")
        return
        
    df = pd.read_parquet(DATA_PATH)
    
    if 'TARGET' not in df.columns:
        print("Error: No existe la columna TARGET")
        return

    # Seleccionar solo numericas
    X = df.select_dtypes(include=[np.number]).drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    y = df['TARGET']
    
    print(f"Dimensiones de entrenamiento: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        random_state=42,
        n_jobs=-1
    )
    
    print("Entrenando XGBoost...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "xgboost_model.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(ARTIFACTS_DIR, "model_features.joblib"))
    
    print("Modelo y features guardados correctamente en artifacts.")

if __name__ == "__main__":
    train()