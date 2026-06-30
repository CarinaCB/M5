import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


# Importamos la carga de datos y función de ingeniería de detalles
from cargar_datos import cargarDatos
from ft_engineering import ft_engineering


# Modelos y Métricas
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

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
        print("\n IMPORTANCIA DE VARIABLES (Las que pesen mucho son las tramposas):")
        for i, v in enumerate(importances):
            if v > 0.05:
                print(f"-> Columna índice [{i}]: peso de {v:.4f}")
        print("-" * 50)
    
    # Predicciones
    y_pred = model_instance.predict(X_test)
    y_proba = model_instance.predict_proba(X_test)[:, 1]

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nMatriz de Confusión - {model_name}")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {model_name}")

    nombre_archivo = model_name.lower().replace(" ", "_")
    plt.savefig(f"src/matriz_confusion_{nombre_archivo}.png")
    plt.close()


    resultados = summarize_classification(y_test, y_pred, y_proba, model_name)
    return resultados

def optimizar_random_forest(X_train, y_train):
    print("Iniciando búsqueda de mejores hiperparámetros para Random Forest...")
    
    # Definimos el rango de "perillas" a ajustar
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced_subsample'
    )
    
    # El GridSearchCV busca la mejor combinación usando validación cruzada
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='f1', verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    return grid_search.best_estimator_

def ejecutar_entrenamiento():
    # 1. Cargar datos crudos
    df = cargarDatos()
    
    # 2. Separar target y eliminar variables de fuga (Leakage)
    # Hacemos esto ANTES de pasar los datos a la función de ingeniería
    y = df['Pago_atiempo'] 
    df = df.drop(columns=['Pago_atiempo', 'fecha_prestamo']) 
    
    # 3. Aplicar la ingeniería de características (Ahora recibe X e y)
    X_train, X_test, y_train, y_test = ft_engineering(df, y)

    # Verificación de seguridad
    import numpy as np
    if np.isnan(X_train).any():
        print("¡ALERTA! Aún hay nulos.")
    else:
        print("Datos listos y limpios. Iniciando entrenamiento...")
    
    # 4. Definir los modelos
    modelos = {
        'Regresion Logistica': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Arbol de Decision': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=10, random_state=42)
    }
    
    # 5. Iterar, entrenar y evaluar
    tabla_resumen = []
    for nombre, modelo in modelos.items():
        # build_model entrena (fit) y evalúa (predict)
        res = build_model(modelo, X_train, y_train, X_test, y_test, model_name=nombre)
        tabla_resumen.append(res)


    ## ENTRENAR RANDOM FOREST OPTIMIZADO (Lo hacemos por fuera)
    rf_best = optimizar_random_forest(X_train, y_train)
    res_rf = build_model(rf_best, X_train, y_train, X_test, y_test, model_name="Random Forest")
    tabla_resumen.append(res_rf)
        
    # 6. Crear e imprimir Tabla Resumen
    df_resumen = pd.DataFrame(tabla_resumen)
    print("\n" + "="*60)
    print(" TABLA RESUMEN FINAL")
    print(df_resumen.to_string(index=False))
    print("="*60)

    modelo_a_guardar = modelos['Regresion Logistica']
    
    # Aseguramos que la carpeta src exista (por seguridad)
    if not os.path.exists('src'):
        os.makedirs('src')
        
    # Guardamos el modelo
    ruta_guardado = 'src/mejor_modelo_riesgo.pkl'
    joblib.dump(modelo_a_guardar, ruta_guardado)
    
    print(f"\n✅ Modelo seleccionado: Regresión Logística")
    print(f"🚀 Modelo guardado correctamente en: {ruta_guardado}")
    print("¡Ya tienes tu modelo listo para el Avance 4!")


    
    # 6. Guardar Gráfico Comparativo
    plt.figure(figsize=(7, 4))
    plt.bar(df_resumen['Modelo'], df_resumen['F1-Score'], color=['skyblue', 'salmon', 'lightgreen', 'violet'], width=0.4)
    plt.ylabel('F1-Score')
    plt.title('Comparativa de Performance de Modelos')
    plt.ylim(0, 1.0)
    for i, v in enumerate(df_resumen['F1-Score']):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        
    plt.savefig('src/comparativa_modelos.png')
    print("\n Gráfico comparativo guardado en 'src/comparativa_modelos.png'")




if __name__ == "__main__":
    print(" Iniciando el proceso de modelado...")
    ejecutar_entrenamiento()