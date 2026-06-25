# librerías
import pandas as pd
from cargar_datos import cargarDatos
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# cargar los datos
df = cargarDatos()

# vamos a tener una vista previa de los datos
print(df.head())
print(df.info())
print(df.describe())

# Paso 1: features/target split
# Armamos la lista de columnas tramposas
columnas_tramposas = ['saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor', 'fecha_prestamo','puntaje' ]

# Las dropeamos junto con el target para armar X
X = df.drop(columns=['Pago_atiempo'] + columnas_tramposas, errors='ignore') 

#  Dejamos solo el target en y
y = df['Pago_atiempo']


# Paso 2: definir variables por tipo
num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes('object').columns

print("Numeric features")
print(num_features)
print("Categorical features")
print(cat_features)

# Paso 3: Crear pipelines para cada ruta
## Ruta 1: numéricas
num_transformer =  Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='mean'))
]
)

## Ruta 2: categóricas
cat_transformer = Pipeline(steps=[
    ('to_str', FunctionTransformer(lambda x: x.astype(str))),
    ('inputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
]
)

# Paso 4: Combinar las 2 rutas en ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Paso 5: dividir el dataset en train/test (antes de preprocesar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Paso 6: Aplicamos el preprocesamiento

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Paso 7: resultados del preprocesamiento 

print("X_train preprocesados:")
print(X_train_processed)
print(X_train_processed.shape)
print("X_test preprocesados:")
print(X_test_processed)
print(X_test_processed.shape)

# Paso 8: construimos una función para "exportar": ft_engineering()
def ft_engineering(dataframe):
    """
    Recibe el dataframe original completo, remueve variables de fuga,
    separa features/target y aplica el ColumnTransformer correctamente.
    """
    # 1. Lista de columnas sospechosas que delatan el comportamiento futuro
    columnas_a_excluir = [
        'fecha_prestamo', 
        'saldo_mora', 
        'saldo_mora_codeudor', 
        'saldo_total', 
        'saldo_principal'
    ]
    
    # Nos aseguramos de eliminar solo las que existan en el dataframe real
    columnas_a_dropear = [col for col in columnas_a_excluir if col in dataframe.columns]
    
    # 2. Separar Features y Target
    X_raw = dataframe.drop(columns=['Pago_atiempo'] + columnas_a_dropear, errors='ignore')
    y_raw = dataframe['Pago_atiempo']
    
    # 3. Detectar tipos de variables dinámicamente sobre X_raw
    num_features = X.select_dtypes(include=['number']).columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns

    # 4. Definir Pipelines de transformación
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    # 5. Dividir en Train y Test usando el stratify correcto
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    # 6. Ajustar y transformar de manera aislada
    X_train_proc = preprocessor.fit_transform(X_train_r)
    X_test_proc = preprocessor.transform(X_test_r)
    
    return X_train_proc, X_test_proc, y_train_r, y_test_r




