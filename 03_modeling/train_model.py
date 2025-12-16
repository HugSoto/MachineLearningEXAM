import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

DATA_PATH = "02_data_preparation/master_table.parquet"
ARTIFACTS_DIR = "artifacts"

def train():
    print("Iniciando Entrenamiento Avanzado...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra {DATA_PATH}")
        return
        
    df = pd.read_parquet(DATA_PATH)
    
    if 'TARGET' not in df.columns:
        print("Error: No existe la columna TARGET")
        return

    X = df.select_dtypes(include=[np.number]).drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    y = df['TARGET']
    
    print(f"Dimensiones de entrenamiento: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.02,
        scale_pos_weight=ratio,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("Entrenando XGBoost (Mode: Hardcore)...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC Score Final: {auc:.4f}")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "xgboost_model.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(ARTIFACTS_DIR, "model_features.joblib"))
    
    print("Modelo guardado.")

if __name__ == "__main__":
    train()