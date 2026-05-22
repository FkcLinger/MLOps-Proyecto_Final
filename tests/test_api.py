# ARCHIVO: tests/test_api.py
from fastapi.testclient import TestClient
from src.app import app

# Creamos un cliente de pruebas para simular peticiones web
client = TestClient(app)

def test_health_check():
    """Valida que el servidor levante correctamente"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API de Renovación de Préstamos operativa"}

def test_predict_endpoint_success():
    """Valida que el modelo responda a un JSON válido"""
    payload = {
        "Linea_Renovado": 3770, "Plazo_Renovado": 12, "Uso_Linea": 0, "Uso_TrimLinea": 0,
        "Nro_Entidades": 1, "Dif_Entidades": -1, "Saldo_Consumo": 271.12, "Meses_oferta": 21,
        "Ahorro": 1850, "Prestamo_vigente": 457, "Promed_6Mdeuda": 3754.33, "Flag_LimProv": 1,
        "Deuda_Cubierta%": 0.07, "EDAD": 25, "SEXO": "M", "EST_CIVIL": "S",
        "ANTIGUEDAD_MES": 23, "REGION": "LIMA NORTE", "SUELDO_ESTIMADO": 3531
    }
    
    response = client.post("/predict", json=payload)
    
    # Verificamos que no haya error de servidor ni de validación Pydantic
    assert response.status_code == 200
    
    # Verificamos que la estructura de respuesta contenga nuestras llaves clave
    data = response.json()
    assert "renueva_prestamo" in data
    assert "probabilidad_renovacion" in data
    assert "cluster_asignado" in data