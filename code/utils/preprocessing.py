import pandas as pd
import json

from joblib import load

pca = load("models\\pca_transformer.joblib")
scaler = load('models\\standard_scaler.joblib')

def load_label_encodings(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def encode_categorical(df: pd.DataFrame, column: str, file_path: str = "utils\\label_encodings.json"):
    encodings = load_label_encodings(file_path)
    
    if column not in encodings:
        raise ValueError(f"Column {column} not found in label encodings")
    
    df[column] = df[column].map(encodings[column])
    
    if df[column].isnull().any():
        raise ValueError(f"Encontrada categoría desconocida en la columna {column}")
    
    return df

def preprocess_input(data):
    df = pd.DataFrame([data])

    df["SeniorCity"] = df["SeniorCity"].astype(int)
    df["Charges"] = df["Charges"].astype(float)
    
    required_columns = ['SeniorCity', 'Partner', 'Dependents', 'Service1', 'Service2', 'Security', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod', 'Charges']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no está presente en los datos de entrada")

    df = df.drop(["autoID", "SeniorCity"], axis=1)
    
    categorical_columns = [
        'Partner',
        'Dependents',
        'Service1',
        'Service2',
        'Security',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod'
    ]

    numeric_columns = [
        'Charges'
    ]
    
    for col in categorical_columns:
        df = encode_categorical(df, col)
    
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    pca_array = pca.transform(df)
    
    return pca_array