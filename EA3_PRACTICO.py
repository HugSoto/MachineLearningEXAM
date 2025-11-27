import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set(style='whitegrid')

TRAIN_PATH = 'application_.parquet'
TARGET_COL = 'TARGET'
RANDOM_STATE = 42
MAX_CHECK_CLUSTERS = 8

print(f"--- INICIANDO PROCESO ---")
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"No se encontró {TRAIN_PATH}")

print(f"Cargando archivo: {TRAIN_PATH}...")

df = pd.read_parquet(TRAIN_PATH, engine='pyarrow')

if TARGET_COL in df.columns:
    df = df[df[TARGET_COL].notnull()]
    print(f"Filtrado set de entrenamiento. Dimensiones: {df.shape}")
else:
    print(f"Advertencia: No se encontró TARGET. Dimensiones: {df.shape}")

y = df[TARGET_COL] if TARGET_COL in df.columns else None
X = df.drop(columns=[TARGET_COL, 'SK_ID_CURR'], errors='ignore')

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nVariables iniciales: {len(numeric_cols)} numéricas, {len(cat_cols)} categóricas")

MAX_NUMERIC = 40
if len(numeric_cols) > MAX_NUMERIC and y is not None:
    print(f"Reduciendo variables numéricas a las top {MAX_NUMERIC} más correlacionadas...")
    correlations = df[numeric_cols].corrwith(df[TARGET_COL]).abs().sort_values(ascending=False)
    numeric_cols = correlations.index[:MAX_NUMERIC].tolist()
    print(f"Variables seleccionadas: {numeric_cols[:5]} ...")

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, cat_cols)
    ],
    remainder='drop'
)

print("\nPreprocesando datos (Imputación + Escalado + OneHot)...")
SAMPLE_SIZE = 50000
if len(X) > SAMPLE_SIZE:
    X_sample = X.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"Usando muestra de {SAMPLE_SIZE} filas para calibración...")
else:
    X_sample = X

preprocessor.fit(X_sample)
X_processed_sample = preprocessor.transform(X_sample)

print("Ejecutando PCA...")
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca_sample = pca.fit_transform(X_processed_sample)
print(f"PCA redujo los datos a {X_pca_sample.shape[1]} componentes principales.")

print("\nBuscando el número óptimo de clusters (k)...")
inertia = []
sil_scores = []
K_range = range(2, MAX_CHECK_CLUSTERS + 1)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca_sample)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca_sample, labels, sample_size=10000))
    print(f"K={k}: Inertia={int(km.inertia_)}, Silhouette={sil_scores[-1]:.3f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, 'bo-')
plt.title('Método del Codo (Inertia)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, 'ro-')
plt.title('Silhouette Score (Mayor es mejor)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Score')

plt.tight_layout()
plt.show()

best_k = K_range[np.argmax(sil_scores)]
print(f"\n---> El número de clusters sugerido es k={best_k}")

print(f"Aplicando K-Means final con k={best_k} a todo el dataset...")

X_processed_full = preprocessor.transform(X)
X_pca_full = pca.transform(X_processed_full)

kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans_final.fit_predict(X_pca_full)

df['Cluster'] = clusters

print("\n--- PERFIL DE LOS CLUSTERS ---")
if TARGET_COL in df.columns:
    stats = df.groupby('Cluster')[TARGET_COL].agg(['count', 'mean']).reset_index()
    stats.columns = ['Cluster', 'Cantidad_Clientes', 'Tasa_Incumplimiento']
    stats['Tasa_Incumplimiento'] = stats['Tasa_Incumplimiento'].map('{:.2%}'.format)
    print(stats)
else:
    print(df['Cluster'].value_counts())

plt.figure(figsize=(10, 6))
idx_plot = np.random.choice(len(X_pca_full), size=min(10000, len(X_pca_full)), replace=False)
plt.scatter(X_pca_full[idx_plot, 0], X_pca_full[idx_plot, 1], c=clusters[idx_plot], cmap='viridis', alpha=0.6, s=15)
plt.title(f'Segmentación Final (k={best_k}) - PCA Components 1 & 2')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()

print("\n¡Proceso finalizado!")