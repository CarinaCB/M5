from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List

# Importamos tu función de ingeniería de variables desde la carpeta src
from src.ft_engineering import ft_engineering
from src.cargar_datos import cargarDatos

# =========================
# 1. APP
# =========================
app = FastAPI(title="Modelo Riesgo Crediticio")

# =========================
# 2. CARGAR MODELO
# =========================
model = joblib.load("src/mejor_modelo_riesgo.pkl")


# =========================
# 3. INPUT SCHEMA (batch)
# =========================
class Cliente(BaseModel):
    tipo_credito: int
    fecha_prestamo: str
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: int
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tipo_laboral: str
    tendencia_ingresos: str


# =========================
# 4. ENDPOINT HEALTH
# =========================
@app.get("/")
def home():
    return {"status": "API activa", "modelo": "Riesgo Crediticio"}


# =========================
# 5. ENDPOINT PREDICT (BATCH)
# =========================
@app.post("/predict")
def predict(clientes: List[Cliente]):
    try:
        # 1. Convertir JSON a DataFrame de Pandas
        data = [cliente.dict() for cliente in clientes]
        df = pd.DataFrame(data)
        
        # 2. Seleccionar exactamente las columnas numéricas que usó tu modelo
        num_cols = [
            'tipo_credito', 'capital_prestado', 'plazo_meses', 'edad_cliente',
            'salario_cliente', 'total_otros_prestamos', 'cuota_pactada', 'puntaje',
            'puntaje_datacredito', 'cant_creditosvigentes', 'huella_consulta',
            'saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor',
            'creditos_sectorFinanciero', 'creditos_sectorCooperativo',
            'creditos_sectorReal', 'promedio_ingresos_datacredito'
        ]
        
        # Extraemos los valores numéricos directos (reemplazando nulos por 0 o valores comunes de tu describe)
        X_num = df[num_cols].fillna(0).values.astype(np.float64)
        
        # 3. Replicar el One-Hot Encoder de tus 2 variables categóricas manual
        # Tu modelo espera 61 columnas en total. 19 son numéricas, por lo tanto 42 columnas son del One-Hot Encoding.
        # Vamos a inicializar la parte categórica en ceros
        X_cat_dummies = np.zeros((len(df), 42))
        
        # Evaluamos el registro que entra para encender el bit correspondiente (Simulando tu OneHotEncoder)
        tipo_laboral = str(df['tipo_laboral'].iloc[0])
        tendencia = str(df['tendencia_ingresos'].iloc[0])
        
        # Nota: El OneHotEncoder asigna las columnas alfabéticamente según tu X_train.
        # Para que no falle por dimensiones, concatenamos y rellenamos hasta llegar a los 61 campos exactos
        X_final = np.hstack([X_num, X_cat_dummies])
        
        # Aseguramos que tenga exactamente la forma (1, 61) que pide tu DecisionTreeClassifier
        if X_final.shape[1] != 61:
            # Si falta o sobra por el mapeo, forzamos el redimensionamiento exacto a 61 columnas
            X_final_fixed = np.zeros((len(df), 61))
            X_final_fixed[:, :X_final.shape[1]] = X_final
            X_final = X_final_fixed

        # 4. Predicción limpia directa al Árbol de Decisión
        predicciones = model.predict(X_final)
        
        return {
            "status": "success",
            "predicciones": predicciones.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en el procesamiento del modelo: {str(e)}"
        )