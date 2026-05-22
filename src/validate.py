# ARCHIVO: src/validate.py
import json
import sys
import os

def validate_metrics(metrics_path='artifacts/metrics.json', threshold=0.55):
    print(f"Iniciando validación de calidad del modelo (Umbral mínimo de Recall: {threshold})...")
    
    # Verificamos que el entrenamiento se haya ejecutado y exista el JSON
    if not os.path.exists(metrics_path):
        print(f"ERROR: No se encontró el archivo de métricas en {metrics_path}. ¿Ejecutaste train.py primero?")
        sys.exit(1)
        
    try:
        # 1. Leer las métricas generadas por train.py
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        current_recall = metrics.get("recall", 0.0)
        print(f"Recall del modelo entrenado: {current_recall:.4f}")
        
        # 2. Validar contra el umbral del negocio
        if current_recall < threshold:
            print(f"CRÍTICO: El modelo no cumple con el estándar de calidad. {current_recall:.4f} < {threshold}")
            # sys.exit(1) le dice a GitHub Actions que el Job falló y debe detenerse
            sys.exit(1)
        else:
            print("ÉXITO: El modelo cumple con los estándares de producción. Puerta de calidad aprobada.")
            # sys.exit(0) le dice a GitHub Actions que todo está perfecto y siga al Docker
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR crítico leyendo las métricas: {e}")
        sys.exit(1)

if __name__ == '__main__':
    validate_metrics()