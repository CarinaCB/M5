import os
import pandas as pd
import matplotlib.pyplot as plt

# Importamos la carga de datos y TU función de ingeniería de detalles
from cargar_datos import cargarDatos
from ft_engineering import ft_engineering


# Modelos y Métricas
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def summarize_classification(y_true, y_pred, y_proba, model_name="Modelo"):
    """Calcula las métricas de evaluación requeridas por Henry."""
    metrics = {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro'),
        'Recall': recall_score(y_true, y_pred, average='macro'),
        'F1-Score': f1_score(y_true, y_pred, average='macro'),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    return metrics

def build_model(model_instance, X_train, y_train, X_test, y_test, model_name="Modelo"):
    """Entrena y evalúa de forma automatizada un modelo."""
    print(f"Entrenando {model_name}...")
    model_instance.fit(X_train, y_train)
    
    # 🚨 DETECTIVE DE DATA LEAKAGE: Revisamos qué variable se lleva todo el peso
    if model_name == "Arbol de Decision":
        importances = model_instance.feature_importances_
        print("\n🚨 IMPORTANCIA DE VARIABLES (Las que pesen mucho son las tramposas):")
        for i, v in enumerate(importances):
            if v > 0.05:
                print(f"-> Columna índice [{i}]: peso de {v:.4f}")
        print("-" * 50)
    
    # Predicciones
    y_pred = model_instance.predict(X_test)
    y_proba = model_instance.predict_proba(X_test)[:, 1]
    
    resultados = summarize_classification(y_test, y_pred, y_proba, model_name)
    return resultados

def ejecutar_entrenamiento():
    # 1. Cargar datos crudos
    df = cargarDatos()
    
    # 2. Aplicar la ingeniería de características
    X_train, X_test, y_train, y_test = ft_engineering(df)
    
    tabla_resumen = []
    
    # 3. Modelos a evaluar
    modelos = {
        'Regresion Logistica': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Arbol de Decision': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=20, random_state=42)
    }
    
    # 4. Iterar y entrenar
    for nombre, modelo in modelos.items():
        res = build_model(modelo, X_train, y_train, X_test, y_test, model_name=nombre)
        tabla_resumen.append(res)
        
    # 5. Crear e imprimir Tabla Resumen
    df_resumen = pd.DataFrame(tabla_resumen)
    
    # === IMPRIMIR TABLA RESUMEN (Fuera del except para que corra siempre) ===
    print("\n" + "="*60)
    print("📊 TABLA RESUMEN DE EVALUACIÓN DE MODELOS SUPERVISADOS")
    print("="*60)
    print(df_resumen.to_string(index=False))
    print("="*60)
    
    # 6. Guardar Gráfico Comparativo
    plt.figure(figsize=(7, 4))
    plt.bar(df_resumen['Modelo'], df_resumen['F1-Score'], color=['skyblue', 'salmon', 'lightgreen', 'violet'], width=0.4)
    plt.ylabel('F1-Score')
    plt.title('Comparativa de Performance de Modelos')
    plt.ylim(0, 1.0)
    for i, v in enumerate(df_resumen['F1-Score']):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
    plt.savefig('src/comparativa_modelos.png')
    print("\n📈 Gráfico comparativo guardado en 'src/comparativa_modelos.png'")

if __name__ == "__main__":
    print("🚀 Iniciando el proceso de modelado...")
    ejecutar_entrenamiento()