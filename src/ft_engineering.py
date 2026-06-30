# librerías
import pandas as pd
import numpy as np
from cargar_datos import cargarDatos
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def ft_engineering(df, y):
    df = df.copy()
    
    # 1. Ingeniería
    df['ratio_endeudamiento'] = df['saldo_total'] / (df['salario_cliente'] + 1)
    df['carga_financiera'] = df['cuota_pactada'] / (df['salario_cliente'] + 1)
    
    # 2. Selección de columnas
    columnas_tramposas = ['saldo_mora', 'saldo_total', 'saldo_principal', 
                          'saldo_mora_codeudor', 'fecha_prestamo', 'puntaje']
    df_clean = df.drop(columns=columnas_tramposas, errors='ignore')
    
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

    # 3. Pipeline Corregido
    # Forzamos a que todo sea string en las categóricas
    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda x: x.astype(str))), # <-- ESTA ES LA CLAVE
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_transformer = SimpleImputer(strategy='median')

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(df_clean, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Transformación
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    nulos_train = np.isnan(X_train_proc).sum()
    print(f"¿Quedan nulos en X_train procesado?: {nulos_train}")
    
    return X_train_proc, X_test_proc, y_train, y_test


