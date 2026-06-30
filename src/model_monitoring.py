# src/model_monitoring_v2.py

# librerías
import os
import pandas as pd
import numpy as np
# esta es la librería que usaremos para crear la aplicación en 'streamlit'
import streamlit as st
st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")
# librerías de visualización
import plotly.express as px
# librerías de Machine Learning 
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
# importamos el método para cargar datos
from cargar_datos import cargarDatos

#######################################
# 1. Cargar dataset y dividir los datos
#######################################
@st.cache_data
def load_data():
    # 1.1 Acá estamos llamando a la función 'cargarDatos()' para asignarlo a la variable 'df'
    df = cargarDatos()

    # 1.2 Acá vamos a crear los feautures y el target
    target = "Pago_atiempo"
    X = df.drop(columns=[target]) # estos son los features (estas son las variables que nos ayudarán a predecir el target)
    y = df[target]                # este es el target (es lo que queremos predecir)

    # 1.3 En este paso vamos a dividir la 'X' (features) y el 'y' (el target) en train/test
    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1.4 Que la función retorne esos dataframes/series
    return X_new, X_ref, y_new, y_ref

X_new, X_ref, y_new, y_ref = load_data()

print("Dataset cargado correctamente y dividido en referencia (ref) y nuevo (new):")
print("X_new son los features nuevos:")
print(X_new)
print("X_ref son los features viejos (ref):")
print(X_ref)
print("y_new es el target nuevo:")
print(y_new)
print("y_ref es el target viejo (ref):")
print(y_ref)


# 2. Crearnos la interfaz inicial de la aplicación
st.title("📊 Aplicación para el monitoreo de los datos")

# --- RESUMEN DE SALUD ---
st.subheader("Resumen de Monitoreo")
variables_con_drift = []
resultados_drift = []

cols_numericas = X_ref.select_dtypes(include=["float64", "int64"]).columns

# --- ALERTA GLOBAL ---
placeholder_alerta = st.empty()

# Iniciamos el bucle
for col in cols_numericas:
    # Creamos copias limpias temporalmente para el monitoreo
    ref_col = X_ref[col].dropna()
    new_col = X_new[col].dropna()

    # Si después de borrar los NaN no quedan datos, saltamos la variable
    if ref_col.empty or new_col.empty:
        st.warning(f"La variable {col} no tiene datos suficientes para analizar.")
        continue
    
    stat, p_value = ks_2samp(X_ref[col], X_new[col])
    
    # Detectamos drift
    if p_value < 0.05:
        variables_con_drift.append(col)

    
    resultados_drift.append({
        "Variable": col,
        "Estadístico KS": stat,
        "P-Value": p_value,
        "Estado": "⚠️ Drift" if p_value < 0.05 else "✅ Estable"
    })
    
    # Grafico
    with st.expander(f"Variable: {col} {'⚠️' if p_value < 0.05 else '✅'}"):
        col1, col2 = st.columns(2)
        col1.metric("Estadístico KS", f"{stat:.4f}")
        
        if p_value < 0.05:
            col2.error(f"⚠️ Drift detectado (p-value: {p_value:.4f})")
        else:
            col2.success(f"✅ Estable (p-value: {p_value:.4f})")

            
        # El gráfico ahora es específico para la variable actual
        fig = px.histogram(pd.DataFrame({"Referencia": X_ref[col], "Nuevo": X_new[col]}), 
                           barmode="overlay", opacity=0.7, 
                           title=f"Distribución: {col}")
        st.plotly_chart(fig)

# --- ALERTA GLOBAL (lógica actualizada) ---
if len(variables_con_drift) > 0:
    placeholder_alerta.error(f"🚨 ¡Atención! Se detectó Data Drift en {len(variables_con_drift)} variables.")
else:
    placeholder_alerta.success("✅ El modelo se encuentra estable.")
   

    # --- TABLA RESUMEN FINAL ---
st.subheader("📋 Resumen detallado de variables")
df_resultados = pd.DataFrame(resultados_drift)

# Usamos st.dataframe para una tabla interactiva y con colores
st.dataframe(
    df_resultados.style.map(
        lambda x: 'background-color: #ffcccc' if 'Drift' in x else 'background-color: #d4edda', 
        subset=['Estado']
    ),
    use_container_width=True
)


