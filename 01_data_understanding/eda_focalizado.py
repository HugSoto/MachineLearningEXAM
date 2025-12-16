import pandas as pd
import os

# Ajusta la ruta según donde estés ejecutando
DATA_PATH = "../application_.parquet" if os.path.exists("../application_.parquet") else "application_.parquet"

def main():
    print("--- INFORME EXPLORATORIO BASICO (EDA) ---")
    if not os.path.exists(DATA_PATH):
        print("No se encuentran los datos crudos para analizar.")
        return

    df = pd.read_parquet(DATA_PATH)
    
    print(f"1. Dimensiones del Dataset: {df.shape}")
    print("\n2. Distribucion del Target (Desbalance):")
    print(df['TARGET'].value_counts(normalize=True))
    
    print("\n3. Tipos de Datos:")
    print(df.dtypes.value_counts())
    
    print("\n4. Valores Nulos (Top 10 columnas con mas nulos):")
    missing = df.isnull().mean().sort_values(ascending=False).head(10)
    print(missing)

    print("\nConclusión: Dataset desbalanceado con alta presencia de nulos.")
    print("Acción: Se requiere imputación y técnica de rebalanceo (Scale Pos Weight).")

if __name__ == "__main__":
    main()