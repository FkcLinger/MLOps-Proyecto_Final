# ARCHIVO: tests/test_data.py
import os
import pandas as pd

def test_csv_exists_and_not_empty():
    # Toma exactamente la ruta desde donde lanzas el comando en tu terminal de Codespaces
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "data", "Dataset Renovacion_prestamo.csv")
    
    # Assert con mensaje descriptivo para debug
    assert os.path.exists(file_path), f"Fallo Crítico: Python buscó en esta ruta exacta y no lo halló: {file_path}"
    
    # Validaciones de datos
    df = pd.read_csv(file_path, sep=';')
    assert not df.empty, "Error: El dataset cargó pero está vacío."
    assert 'FLAG_VENTA' in df.columns, "Error: La variable objetivo FLAG_VENTA desapareció."