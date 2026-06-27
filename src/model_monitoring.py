# src/model_monitoring_v2.py

# librerías
import os
import pandas as pd
# esta es la librería que usaremos para crear la aplicación en 'streamlit'
import streamlit as st
st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")
# librerías de visualización
import plotly.express as px
# librerías de Machine Learning 
from sklearn.model_selection import train_test_split
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

####################################
# 2. Crearnos la interfaz inicial de 
# la aplicación de Streamlit
####################################
st.title("📊 Aplicación para el monitoreo de los datos")