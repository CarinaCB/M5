import pandas as pd
import os

# 1. Detectar automáticamente la ubicación de este script
dir_actual = os.path.dirname(os.path.abspath(__file__))

# 2. Construir la ruta correcta hacia la carpeta 'data' subiendo un nivel
# Esto busca en: M5INTEGRADOR/M5/data/Base_de_datos.xlsx
ruta_data = os.path.join(dir_actual, '..', 'data', 'Base_de_datos.xlsx')

print(f"Buscando el archivo en: {ruta_data}")

# 3. LEER EL EXCEL (Usando read_excel en lugar de read_csv)
try:
    df = pd.read_excel(ruta_data)
    print("¡Lectura exitosa!")
    print(f"Dimensiones del dataset: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print("\n❌ Error: El archivo no se encuentra en la carpeta 'data'.")
    print("Asegúrate de que 'Base_de_datos.xlsx' esté guardado dentro de M5/data/")
except Exception as e:
    print(f"\n❌ Ocurrió otro error: {e}")
    print("Tip: Si te pide 'openpyxl', ejecuta en la terminal: pip install openpyxl")