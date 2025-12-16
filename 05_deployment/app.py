import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "artifacts/xgboost_model.joblib"
FEATURES_PATH = "artifacts/model_features.joblib"
DATA_PATH = "02_data_preparation/master_table.parquet"

st.set_page_config(page_title="Sistema de Riesgo Crediticio")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        st.error("Error: No se encuentran los archivos del modelo en artifacts.")
        return None, None
    
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

@st.cache_data
def load_sample_data():
    if not os.path.exists(DATA_PATH):
        st.error("Error: No se encuentra master_table.parquet.")
        return pd.DataFrame()
    
    df = pd.read_parquet(DATA_PATH, columns=['SK_ID_CURR', 'TARGET'] + joblib.load(FEATURES_PATH))
    return df.sample(100, random_state=42)

st.title("Prediccion de Riesgo Crediticio")

model, model_features = load_assets()
df_sample = load_sample_data()

if model is not None and not df_sample.empty:
    
    st.subheader("Seleccionar Cliente")
    client_ids = df_sample['SK_ID_CURR'].astype(int).tolist()
    selected_id = st.selectbox("ID del Solicitante:", client_ids)
    
    client_data = df_sample[df_sample['SK_ID_CURR'] == selected_id].copy()
    
    st.dataframe(client_data.drop(columns=['TARGET'], errors='ignore'))

    if st.button("Evaluar Solicitud"):
        X_input = client_data.reindex(columns=model_features, fill_value=0)
        
        probability = model.predict_proba(X_input)[0][1]
        
        st.subheader("Resultado de la Evaluacion")
        st.metric("Probabilidad de Incumplimiento", f"{probability:.2%}")
        
        if probability > 0.50:
            st.error("RECOMENDACION: RECHAZAR")
        else:
            st.success("RECOMENDACION: APROBAR")

else:
    st.info("Cargando sistema...")