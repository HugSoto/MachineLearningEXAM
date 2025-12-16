import pandas as pd
import numpy as np
import gc
import os

DATA_DIR = "."
OUTPUT_FOLDER = "02_data_preparation"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "master_table.parquet")

def agg_numeric(df, group_var, df_name):
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = df[group_var]
    agg = numeric_df.groupby(group_var).agg(['mean', 'max', 'min', 'sum'])
    columns = []
    for var in agg.columns.levels[0]:
        for stat in agg.columns.levels[1]:
            columns.append(f'{df_name}_{var}_{stat}')
    agg.columns = columns
    return agg.reset_index()

def main():
    print("Iniciando integracion de datos...")
    
    path_app = os.path.join(DATA_DIR, 'application_.parquet')
    if not os.path.exists(path_app):
        print(f"ERROR: No encuentro {path_app}")
        return

    df_main = pd.read_parquet(path_app, engine='pyarrow')
    print(f"Filas iniciales: {df_main.shape[0]}")

    path_bureau = os.path.join(DATA_DIR, 'bureau.parquet')
    if os.path.exists(path_bureau):
        print("Procesando bureau.parquet...")
        bureau = pd.read_parquet(path_bureau, engine='pyarrow')
        bureau_agg = agg_numeric(bureau.drop(columns=['SK_ID_BUREAU']), group_var='SK_ID_CURR', df_name='bureau')
        df_main = df_main.merge(bureau_agg, on='SK_ID_CURR', how='left')
        del bureau, bureau_agg
        gc.collect()

    path_prev = os.path.join(DATA_DIR, 'previous_application.parquet')
    if os.path.exists(path_prev):
        print("Procesando previous_application.parquet...")
        prev = pd.read_parquet(path_prev, engine='pyarrow')
        prev_agg = agg_numeric(prev.drop(columns=['SK_ID_PREV']), group_var='SK_ID_CURR', df_name='prev')
        df_main = df_main.merge(prev_agg, on='SK_ID_CURR', how='left')
        del prev, prev_agg
        gc.collect()

    print("Calculando Ratios Financieros...")
    df_main['PAYMENT_RATE'] = df_main['AMT_ANNUITY'] / df_main['AMT_CREDIT']
    df_main['INCOME_CREDIT_RATIO'] = df_main['AMT_INCOME_TOTAL'] / df_main['AMT_CREDIT']

    print("Limpiando y guardando...")
    df_main = df_main.replace([np.inf, -np.inf], np.nan)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df_main.to_parquet(OUTPUT_FILE, engine='pyarrow')
    print(f"Dataset maestro guardado en: {OUTPUT_FILE}")
    print(f"Dimensiones finales: {df_main.shape}")

if __name__ == "__main__":
    main()