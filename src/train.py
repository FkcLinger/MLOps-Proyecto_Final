# ARCHIVO: src/train.py
import json
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from preprocess import load_data, apply_feature_engineering, build_preprocessor

def main():
    print("Iniciando pipeline integral (Replicando experimento del Colab)...")
    
    # 1. Carga y Feature Engineering base
    df = load_data('data/Dataset Renovacion_prestamo.csv') 
    df = apply_feature_engineering(df)
    
    # 2. Separar X e Y 
    target_col = 'FLAG_VENTA'
    df = df.drop(columns=['CLIENTE', 'MES'], errors='ignore')
    df[target_col] = df[target_col].astype(int)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split con estratificación (stratify=y) como en tu línea 141
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # =========================================================
    # 3. UNDERSAMPLING MANUAL (Clave para rescatar el Recall)
    # =========================================================
    print("Aplicando Undersampling a la clase mayoritaria...")
    df_train = pd.concat([X_train, y_train], axis=1)
    count_class_0, count_class_1 = df_train[target_col].value_counts()
    
    df_class_0 = df_train[df_train[target_col] == 0]
    df_class_1 = df_train[df_train[target_col] == 1]
    
    # sample(count_class_1) iguala la cantidad de ceros y unos
    df_class_0_under = df_class_0.sample(count_class_1, random_state=42)
    df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
    
    X_train_u = df_train_under.drop(columns=[target_col])
    y_train_u = df_train_under[target_col]

    # =========================================================
    # 4. CLUSTERING COMO VARIABLE (Entrenar KMeans y añadir Cluster)
    # =========================================================
    print("Entrenando modelo K-Means y asignando Clusters...")
    clustering_features = ['Uso_TrimLinea_LOG', 'Prestamo_vigente_LOG', 'Uso_Linea_LOG']
    
    # Preparamos datos limpios para KMeans temporalmente
    X_cluster_train = X_train_u[clustering_features].fillna(X_train_u[clustering_features].median())
    X_cluster_test = X_test[clustering_features].fillna(X_train_u[clustering_features].median())
    
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    
    # Asignamos la nueva columna a los datasets de entrenamiento y prueba
    X_train_u['Cluster'] = kmeans.fit_predict(X_cluster_train)
    X_test['Cluster'] = kmeans.predict(X_cluster_test)

    # =========================================================
    # 5. PREPROCESADOR FINAL Y ENTRENAMIENTO DEL MODELO
    # =========================================================
    numeric_features = X_train_u.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    categorical_features = X_train_u.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # RandomForest con parámetros robustos basados en tu GridSearchCV
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, class_weight='balanced', random_state=42))
    ])

    print("Entrenando RandomForest Final...")
    model_pipeline.fit(X_train_u, y_train_u)

    # 6. Evaluación
    y_pred = model_pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred)
    }
    
    print(f"\n--- MÉTRICAS OBTENIDAS EN TEST ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}  <-- ¡Esto debe subir radicalmente!")
    print(f"Precision: {metrics['precision']:.4f}")
    
    # 7. Exportación de múltiples artefactos para la API
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    joblib.dump(model_pipeline, 'artifacts/model_pipeline.pkl')
    
    # ¡Guardamos el KMeans! La API lo necesitará para calcular el clúster de clientes nuevos
    joblib.dump(kmeans, 'artifacts/kmeans_model.pkl')
    
    print("\nÉxito: Artefactos (Pipeline, KMeans y Métricas) guardados en 'artifacts/'")

if __name__ == '__main__':
    main()