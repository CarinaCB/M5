FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas numpy xgboost joblib python-multipart openpyxl

COPY . .

# ESTO ES LO QUE SOLUCIONA TU PROBLEMA SIN TOCAR TU CÓDIGO:
# Al añadir 'src' al PYTHONPATH, le decimos a Python que busque módulos 
# en esa carpeta automáticamente.
ENV PYTHONPATH=/app/src

# Ahora ejecutamos uvicorn apuntando al archivo, 
# pero como la ruta ya está en PYTHONPATH, no fallará.
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8080"]