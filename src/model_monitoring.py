# src/model_monitoring.py

# librerías
import os
import pandas as pd
import numpy as np
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
    # 1.1 Llamamos a la función para cargar datos
    df = cargarDatos()
    
    # 1.2 Creamos los features y el target
    target = "Pago_atiempo"
    X = df.drop(columns=[target]) 
    y = df[target]                

    # 1.3 Dividir respetando el orden estándar de scikit-learn
    # X_train (Referencia: 80%), X_test (Nuevo: 20%)
    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_ref, X_new, y_ref, y_new

# Asignamos manteniendo el orden del return
X_ref, X_new, y_ref, y_new = load_data()


# 2. Interfaz de la aplicación
st.title("📊 Aplicación para el monitoreo de los datos")

# --- RESUMEN DE SALUD ---
st.subheader("Resumen de Monitoreo")
variables_con_drift = []
resultados_drift = []

cols_numericas = X_ref.select_dtypes(include=["float64", "int64"]).columns

# --- ALERTA GLOBAL (Usamos el placeholder aquí) ---
placeholder_alerta = st.empty()

# Iniciamos el bucle
for col in cols_numericas:
    # Copias limpias para el monitoreo
    ref_col = X_ref[col].copy()
    new_col = X_new[col].copy()

    # Si la variable está vacía en ambos, saltamos
    if ref_col.isnull().all() or new_col.isnull().all():
        continue

    mediana_ref = ref_col.median()
    ref_col = ref_col.fillna(mediana_ref)
    new_col = new_col.fillna(mediana_ref)
    
    stat, p_value = ks_2samp(ref_col, new_col)
    
    # Detectamos drift
    if p_value < 0.05:
        variables_con_drift.append(col)
    
    resultados_drift.append({
        "Variable": col,
        "Estadístico KS": stat,
        "P-Value": p_value,
        "Estado": "⚠️ Drift" if p_value < 0.05 else "✅ Estable"
    })
    
    # Gráfico por variable
    with st.expander(f"Variable: {col} {'⚠️' if p_value < 0.05 else '✅'}"):
        col1, col2 = st.columns(2)
        col1.metric("Estadístico KS", f"{stat:.4f}")
        
        if p_value < 0.05:
            col2.error(f"⚠️ Drift detectado (p-value: {p_value:.4f})")
        else:
            col2.success(f"✅ Estable (p-value: {p_value:.4f})")

        # Reestructuramos los datos en formato largo ('long-form') para Plotly Express
        df_plot_ref = pd.DataFrame({col: ref_col, "Distribución": "Referencia"})
        df_plot_new = pd.DataFrame({col: new_col, "Distribución": "Nuevo"})
        df_total = pd.concat([df_plot_ref, df_plot_new])
            
        fig = px.histogram(
            df_total, 
            x=col, 
            color="Distribución",
            barmode="overlay", 
            opacity=0.7, 
            title=f"Distribución: {col}"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- ACTUALIZACIÓN DE ALERTA GLOBAL ---
if len(variables_con_drift) > 0:
    placeholder_alerta.error(f"🚨 ¡Atención! Se detectó Data Drift en {len(variables_con_drift)} variables.")
else:
    placeholder_alerta.success("✅ El modelo se encuentra estable.")
    

# --- TABLA RESUMEN FINAL ---
st.subheader("📋 Resumen detallado de variables")
df_resultados = pd.DataFrame(resultados_drift)

# Estilo interactivo para la tabla de resultados
st.dataframe(
    df_resultados.style.map(
        lambda x: 'background-color: #ffcccc' if 'Drift' in x else 'background-color: #d4edda', 
        subset=['Estado']
    ),
    use_container_width=True
)