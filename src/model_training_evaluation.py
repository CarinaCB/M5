import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from cargar_datos import cargarDatos
from ft_engineering import ft_engineering

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)



def summarize_classification(y_true, y_pred, y_proba, model_name="Modelo"):
    return {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='binary', pos_label=1),
        "Recall": recall_score(y_true, y_pred, average='binary', pos_label=1),
        "F1-Score": f1_score(y_true, y_pred, average='binary', pos_label=1),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }


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

# =========================
# 🔥 FEATURE ENGINEERING SIMPLE
# =========================
def crear_features(df):

    df = df.copy()

    # 🔧 FUSIONES IMPORTANTES
    df["deuda_total"] = df["saldo_total"].fillna(0) + df["saldo_mora"].fillna(0)

    df["capacidad_pago"] = df["salario_cliente"] / (df["cuota_pactada"] + 1)

    df["riesgo_credito"] = df["puntaje"] / (df["saldo_total"] + 1)

    # ❌ eliminar fecha (riesgo leakage si no se trata bien)
    if "fecha_prestamo" in df.columns:
        df = df.drop(columns=["fecha_prestamo"])

    return df

# =========================
# 🔥 ELIMINAR FEATURES DOMINANTES
# =========================
def eliminar_features_importantes(model, X, threshold=0.15):

    importances = model.feature_importances_
    cols = X.columns

    importantes = [cols[i] for i, v in enumerate(importances) if v > threshold]

    print("\n🚨 Features eliminadas por dominancia:")
    print(importantes)

    X = X.drop(columns=importantes, errors="ignore")

    return X


def optimizar_random_forest(X_train, y_train):
    print("Optimización Random Forest...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced_subsample"
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    print("Mejores parámetros:", grid.best_params_)
    return grid.best_estimator_


def ejecutar_entrenamiento():

    df = cargarDatos()

    y = df["Pago_atiempo"]
    X = df.drop(columns=["Pago_atiempo"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = ft_engineering(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    

    print("Datos listos y procesados")

    print(list(X_train.columns)[7])

    modelos = {
        "Regresion Logistica": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Arbol de Decision": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=10,
            random_state=42
        )
    }

    resultados = []
    modelos_entrenados = {}

    for nombre, model in modelos.items():

        res = build_model(
            model,
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
            model_name=nombre
        )

        resultados.append(res)
        modelos_entrenados[nombre] = model

    rf_best = optimizar_random_forest(X_train_processed, y_train)

    res_rf = build_model(
        rf_best,
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        model_name="Random Forest"
    )

    resultados.append(res_rf)
    modelos_entrenados["Random Forest"] = rf_best

    df_resumen = pd.DataFrame(resultados)

    print("\nTABLA RESUMEN")
    print(df_resumen)

    mejor_nombre = "Regresion Logistica"
    mejor_modelo = modelos_entrenados[mejor_nombre]

    os.makedirs("src", exist_ok=True)
    joblib.dump(mejor_modelo, "src/mejor_modelo_riesgo.pkl")
    joblib.dump(preprocessor, "src/preprocessor.pkl")

    print("Modelo guardado:", mejor_nombre)

    plt.figure(figsize=(7, 4))
    plt.bar(df_resumen["Modelo"], df_resumen["F1-Score"])
    plt.ylim(0, 1)

    for i, v in enumerate(df_resumen["F1-Score"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.savefig("src/comparativa_modelos.png")


if __name__ == "__main__":
    print("Iniciando entrenamiento...")
    ejecutar_entrenamiento()