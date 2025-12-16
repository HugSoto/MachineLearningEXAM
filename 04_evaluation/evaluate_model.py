import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

DATA_PATH = "02_data_preparation/master_table.parquet"
MODEL_PATH = "artifacts/xgboost_model.joblib"
OUTPUT_DIR = "04_evaluation"

def evaluate():
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        print(f"Error: Faltan archivos de datos o modelo.")
        return

    df = pd.read_parquet(DATA_PATH)
    X = df.select_dtypes(include=[np.number]).drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Matriz de Confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusion')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa Falsos Positivos')
    plt.ylabel('Tasa Verdaderos Positivos')
    plt.title('Curva ROC - Riesgo Credito')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()

    # 3. Importancia de Variables
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-20:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
    plt.title('Top 20 Variables mas Importantes (XGBoost)')
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()

    print(f"Graficos guardados exitosamente en {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()