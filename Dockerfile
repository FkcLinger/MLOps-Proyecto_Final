# 1. Imagen base exigida
FROM python:3.10-slim

# 2. Variables de entorno para optimizar Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Directorio de trabajo
WORKDIR /app

# 4. Dependencias del sistema (Instalar gcc y limpiar caché de apt)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# 5. Gestión de dependencias aprovechando la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copia de código fuente y modelos entrenados
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# 7. Exponer el puerto interno de FastAPI
EXPOSE 8000

# 8. El "Smoke Test" exigido como comando por defecto
# Adaptado para usar joblib y tu archivo model_pipeline.pkl
CMD ["python", "-c", "import joblib; m=joblib.load('artifacts/model_pipeline.pkl'); print('Modelo listo:', type(m).__name__)"]