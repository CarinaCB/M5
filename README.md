Proyecto M5 – Sistema de Predicción de Pago a Tiempo
Objetivo del proyecto
El objetivo del proyecto es construir un modelo de Machine Learning capaz de predecir si un cliente pagará o no a tiempo un crédito, utilizando información financiera y demográfica.
1. Exploración de Datos (EDA)
Archivo: compresión_eda.ipynb / análisis inicial
En esta etapa se realizó el análisis exploratorio del dataset.
 Dataset
•	10.763 registros 
•	23 variables 
•	Variable objetivo: Pago_atiempo 
Hallazgos principales
•	Dataset con alto desbalance de clases 
o	Clase 1 (paga a tiempo): ~95% 
o	Clase 0 (no paga): ~5% 
•	Variables con valores nulos: 
o	puntaje_datacredito 
o	saldo_mora 
o	saldo_total 
o	saldo_principal 
o	saldo_mora_codeudor 
o	promedio_ingresos_datacredito 
o	tendencia_ingresos 
•	Variables mixtas: 
o	Numéricas 
o	Categóricas (tipo_laboral, tendencia_ingresos) 
o	Fecha (fecha_prestamo) 
Conclusión EDA
El dataset requiere:
•	Limpieza de nulos 
•	Codificación de variables categóricas 
•	Transformación de fechas 
•	Pipeline de preprocesamiento 
2. Ingeniería de características
Archivo: ft_engineering.py
Se implementó un pipeline de preprocesamiento:
Transformaciones realizadas
•	Imputación de valores faltantes 
•	Codificación One-Hot para variables categóricas 
•	Escalado implícito en pipeline 
•	Separación de variables numéricas y categóricas 
Output
•	Matriz final de entrenamiento: (8610, 61) 
•	Matriz de test: (2153, 61) 
3. Entrenamiento de modelos
Archivo: model_training_evaluation.py
Se entrenaron múltiples modelos de clasificación:
Modelos utilizados
•	Regresión Logística 
•	Árbol de Decisión 
•	XGBoost 
•	Random Forest (con GridSearchCV) 
 Configuración
•	Split: 80/20 estratificado 
•	Métricas: 
o	Accuracy 
o	Precision 
o	Recall 
o	F1-Score 
o	ROC-AUC 
4. Resultados del modelo
 Regresión Logística
•	Accuracy: 0.81 
•	F1-score: 0.89 
•	Recall clase minoritaria: 0.83 
Modelo más estable y realista
 Árbol de Decisión
•	Accuracy: 1.00 
•	F1-score: 1.00 
Indicio de sobreajuste / fuga de información
 XGBoost
•	Accuracy: 1.00 
•	F1-score: 1.00 
 Posible overfitting severo
 Random Forest
•	Accuracy: 1.00 
•	F1-score: 1.00 
 Igual comportamiento de sobreajuste
 Problema detectado

Se identificó que varios modelos complejos presentan:
•	Predicción perfecta en train/test 
•	Matrices de confusión sin errores 
 Esto indica:
•	Posible data leakage 
•	O variables altamente determinísticas 
Variable crítica detectada
Se identificó que una variable del pipeline:
total_otros_prestamos
tiene un peso excesivo en el Árbol de Decisión, dominando completamente la predicción.
 Modelo final seleccionado
 Regresión Logística
 Motivo
•	No presenta sobreajuste extremo 
•	Generaliza mejor 
•	Es el único modelo confiable en condiciones actuales 
•	Maneja mejor el desbalance 
 Artefacto generado
Se guarda el modelo final en:
src/mejor_modelo_riesgo.pkl
 Visualizaciones generadas
•	Matriz de confusión por modelo 
•	Comparación de F1-score entre modelos 
 Conclusión final
El sistema demuestra que:
•	El dataset está fuertemente desbalanceado 
•	Los modelos complejos tienden a sobreajustar 
•	La regresión logística es la opción más estable en este contexto 
 Mejoras futuras
•	Aplicar SMOTE o undersampling 
•	Revisión profunda de data leakage 
•	Feature selection más estricta 
•	Validación cruzada estratificada 
•	Ajuste de umbral de clasificación 

 Sistema de Monitoreo de Data Drift

Objetivo del Avance
Implementar un sistema de auditoría automatizada para el modelo de riesgo crediticio. El objetivo es identificar desviaciones en la distribución de los datos (Data Drift) comparando la información histórica de entrenamiento contra la información actual mediante pruebas estadísticas.

 Tecnologías y Métodos
* **Framework:** Streamlit para la creación del dashboard interactivo.
* **Análisis Estadístico:** Test de Kolmogorov-Smirnov (`scipy.stats.ks_2samp`) para comparar las distribuciones de variables numéricas.
* **Visualización:** Histogramas superpuestos (Plotly) para contraste visual de frecuencias.
* **Tratamiento de Datos:** Pandas para la limpieza y preparación de series (manejo de valores nulos estructurales).

 Funcionalidades del Módulo
1.  **Auditoría Automática:** Bucle iterativo que analiza automáticamente todas las columnas numéricas del dataset.
2.  **Sistema de Semáforos:** * **Estable:** p-value >= 0.05.
    ***Alerta:** p-value < 0.05 (Indica cambio significativo).
3.  **Visualización:** Gráficos de distribución (Ref vs New) disponibles para cada variable, permitiendo inspeccionar el cambio visualmente.
4.  **Gestión de Calidad:** Exclusión automática de nulos en el test estadístico para evitar sesgos, con tratamiento técnico para variables de buró.

Consideraciones Técnicas
* **Manejo de Nulos:** Se detectaron nulos en variables de buró (`puntaje_datacredito`, `saldos`, etc.). Se determinó que corresponden a clientes sin historial crediticio. Se implementó un filtrado de limpieza (`dropna`) para el cálculo del estadístico, garantizando la validez del Test KS.
* **Interpretación:** Una alerta roja (drift) no implica necesariamente un error en el modelo, sino la necesidad de revisar si el perfil de los clientes actuales ha cambiado respecto al perfil original.

Informe de Avance 4: Despliegue con Docker
1. Objetivo del Avance
Implementar un entorno de ejecución aislado mediante Docker para garantizar la portabilidad y escalabilidad del modelo de predicción de riesgo crediticio, eliminando problemas de compatibilidad entre diferentes sistemas operativos.
2. Arquitectura de Despliegue
Entorno de Ejecución: Se utilizó una imagen basada en python:3.11-slim para optimizar el tamaño y la seguridad del despliegue.

Gestión de Dependencias: Se realizó una selección estratégica de librerías esenciales (FastAPI, Scikit-learn, XGBoost, etc.) para asegurar que el contenedor sea liviano y funcional.
3. Procedimiento de Implementación
Configuración del Dockerfile: Se definieron las instrucciones para el entorno:

WORKDIR /app: Definición del directorio de trabajo.

ENV PYTHONPATH=/app: Configuración del path para la correcta resolución de módulos (gestión de src/).

CMD: Configuración del punto de entrada para ejecutar uvicorn.

Proceso de Build: Se construyó la imagen con el comando docker build -t api-riesgo-credito:latest ., lo cual permite "congelar" todas las dependencias necesarias en una única unidad.

Ejecución: El contenedor se levanta mediante el mapeo de puertos (-p 8080:8080), permitiendo el acceso local a la interfaz de la API.


Resultado del Avance:
Se logró la contenedorización exitosa del modelo de predicción mediante Docker. Se resolvió el despliegue corrigiendo la gestión de dependencias (openpyxl) y la resolución de módulos internos mediante la configuración del PYTHONPATH y la estructura del Dockerfile, garantizando que el servicio sea totalmente independiente del entorno local de desarrollo.

Estado del servicio: Up (Operativo).

Acceso: Interfaz API documentada y accesible en el puerto 8080.

Estrategia: Se mantuvo la integridad del código fuente original, delegando la configuración del entorno exclusivamente a la capa de contenedorización.