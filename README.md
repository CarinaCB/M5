# Módulo 5: Sistema de Monitoreo de Data Drift

## 🎯 Objetivo del Avance
Implementar un sistema de auditoría automatizada para el modelo de riesgo crediticio. El objetivo es identificar desviaciones en la distribución de los datos (Data Drift) comparando la información histórica de entrenamiento contra la información actual mediante pruebas estadísticas.

## 🛠️ Tecnologías y Métodos
* **Framework:** Streamlit para la creación del dashboard interactivo.
* **Análisis Estadístico:** Test de Kolmogorov-Smirnov (`scipy.stats.ks_2samp`) para comparar las distribuciones de variables numéricas.
* **Visualización:** Histogramas superpuestos (Plotly) para contraste visual de frecuencias.
* **Tratamiento de Datos:** Pandas para la limpieza y preparación de series (manejo de valores nulos estructurales).

## 📊 Funcionalidades del Módulo
1.  **Auditoría Automática:** Bucle iterativo que analiza automáticamente todas las columnas numéricas del dataset.
2.  **Sistema de Semáforos:** * ✅ **Estable:** p-value >= 0.05.
    * ⚠️ **Alerta:** p-value < 0.05 (Indica cambio significativo).
3.  **Visualización:** Gráficos de distribución (Ref vs New) disponibles para cada variable, permitiendo inspeccionar el cambio visualmente.
4.  **Gestión de Calidad:** Exclusión automática de nulos en el test estadístico para evitar sesgos, con tratamiento técnico para variables de buró.

## 📉 Consideraciones Técnicas
* **Manejo de Nulos:** Se detectaron nulos en variables de buró (`puntaje_datacredito`, `saldos`, etc.). Se determinó que corresponden a clientes sin historial crediticio. Se implementó un filtrado de limpieza (`dropna`) para el cálculo del estadístico, garantizando la validez del Test KS.
* **Interpretación:** Una alerta roja (drift) no implica necesariamente un error en el modelo, sino la necesidad de revisar si el perfil de los clientes actuales ha cambiado respecto al perfil original.