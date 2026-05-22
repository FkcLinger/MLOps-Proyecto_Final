# ARCHIVO: src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# Importamos la lógica de transformación que ya validamos
from src.preprocess import apply_feature_engineering

app = FastAPI(
    title="API de Renovación de Préstamos",
    description="Endpoint para evaluar riesgo y probabilidad de renovación de clientes.",
    version="1.0.0"
)

# Cargar modelos en la memoria al iniciar el servidor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    model_pipeline = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'model_pipeline.pkl'))
    kmeans_model = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'kmeans_model.pkl'))
except Exception as e:
    print(f"Error cargando artefactos. ¿Ejecutaste train.py primero? Detalle: {e}")

# Definimos la estructura exacta que la API esperará recibir (Data cruda)
class ClienteData(BaseModel):
    Linea_Renovado: float = 3770.0
    Plazo_Renovado: int = 12
    Uso_Linea: float = 0.0
    Uso_TrimLinea: float = 0.0
    Nro_Entidades: int = 1
    Dif_Entidades: int = -1
    Saldo_Consumo: float = 271.12
    Meses_oferta: float = 21.0
    Ahorro: float = 1850.0
    Prestamo_vigente: float = 457.0
    Promed_6Mdeuda: float = 3754.33
    Flag_LimProv: int = 1
    Deuda_Cubierta_porc: float = Field(0.07, alias="Deuda_Cubierta%")
    EDAD: int = 25
    SEXO: str = "M"
    EST_CIVIL: str = "S"
    ANTIGUEDAD_MES: int = 23
    REGION: str = "LIMA NORTE"
    SUELDO_ESTIMADO: float = 3531.0

@app.post("/predict")
def predict_renovacion(cliente: ClienteData):
    try:
        # 1. Convertir el JSON entrante a un DataFrame de Pandas
        # Usamos by_alias para que respete el nombre "Deuda_Cubierta%"
        df_input = pd.DataFrame([cliente.dict(by_alias=True)])
        
        # 2. Aplicar el Feature Engineering (Capping y Logaritmos)
        df_processed = apply_feature_engineering(df_input)
        
        # 3. Asignar el Cluster utilizando el modelo KMeans guardado
        clustering_features = ['Uso_TrimLinea_LOG', 'Prestamo_vigente_LOG', 'Uso_Linea_LOG']
        
        # Llenar nulos temporalmente por si un logaritmo generó NaN
        df_cluster = df_processed[clustering_features].fillna(0)
        df_processed['Cluster'] = kmeans_model.predict(df_cluster)
        
        # 4. Predicción final con el RandomForest
        prediccion = model_pipeline.predict(df_processed)
        probabilidad = model_pipeline.predict_proba(df_processed)[0][1] # Probabilidad de clase 1
        
        return {
            "renueva_prestamo": bool(prediccion[0]),
            "probabilidad_renovacion": round(float(probabilidad), 4),
            "cluster_asignado": int(df_processed['Cluster'].iloc[0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API de Renovación de Préstamos operativa"}