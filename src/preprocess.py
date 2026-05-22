# ARCHIVO: src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    # Tu renombramiento exacto del Colab
    df.rename(columns={
        'LINEA_RENOVADO':'Linea_Renovado', 'PLAZO_RENOVADO':'Plazo_Renovado', 
        'USO_LINEA_TOTAL_TC_T2':'Uso_Linea', 'USO_TRIM_LINEA_BBVA':'Uso_TrimLinea',
        'NR_ENTIDADES_TOTAL_T2':'Nro_Entidades', 'DIFF_NRO_ENTIDA_TOTALES_T2_T12':'Dif_Entidades',
        'SDO_CONSUMO_T2':'Saldo_Consumo', 'RESENCIA_OFERTA_PLD_RENOVADO':'Meses_oferta',
        'Ahorro_Sldo_Bco_T1':'Ahorro', 'PConsumo_Sldo_Bco_T1':'Prestamo_vigente',
        'SDO_BCO_tot_sm_pasivo_Bco_6M':'Promed_6Mdeuda', 'FLAG_LIMA_PROVINCIA':'Flag_LimProv',
        'CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD':'Deuda_Cubierta%'
    }, inplace=True)
    return df

def apply_feature_engineering(df):
    """Aplica el Capping y Transformaciones Logarítmicas de tu Colab"""
    df = df.copy()
    
    # 1. Capping (Límites a 0)
    for col in ['Ahorro', 'Prestamo_vigente', 'Promed_6Mdeuda']:
        if col in df.columns:
            df[col] = np.maximum(0, df[col])

    # 2. Transformaciones Logarítmicas
    variables_to_transform = [
        'Uso_Linea', 'Uso_TrimLinea', 'Saldo_Consumo', 'SUELDO_ESTIMADO', 'ANTIGUEDAD_MES',
        'Linea_Renovado', 'Ahorro', 'Prestamo_vigente', 'Promed_6Mdeuda', 'Deuda_Cubierta%'
    ]
    for col in variables_to_transform:
        new_col = f'{col}_LOG'
        if col in df.columns:
            df[new_col] = np.log1p(df[col])
            df = df.drop(columns=[col]) # Eliminamos la original como en tu script
            
    return df

def build_preprocessor(numeric_features, categorical_features):
    # En MLOps usamos SimpleImputer(mediana/moda) para garantizar que la API 
    # asigne los mismos valores en producción sin depender de aleatoriedad.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])