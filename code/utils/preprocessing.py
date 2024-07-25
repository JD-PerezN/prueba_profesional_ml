import pandas as pd

from joblib import load

pca = load("code\\models\\pca_transformer.joblib")
scaler = load('code\\models\\standard_scaler.joblib')

label_encodings = {
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'Service1': {'No': 0, 'Yes': 1},
    'Service2': {'No': 0, 'Yes': 1, 'No phone service': 2},
    'Security': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
}

def encode_categorical(df: pd.DataFrame, column: str, encodings: dict):
    df[column] = df[column].map(encodings)
    if df[column].isnull().any():
        raise ValueError(f"Encontrada categoría desconocida en la columna {column}")
    return df

def preprocess_input(data):
    # df = pd.DataFrame([data])
    df = pd.read_csv(data)
    
    required_columns = ['SeniorCity', 'Partner', 'Dependents', 'Service1', 'Service2', 'Security', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod', 'Charges']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no está presente en los datos de entrada")

    df = df.drop(["autoID", "SeniorCity", "Demand", "Class"], axis=1)
    
    # Codificar variables categóricas
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
        df = encode_categorical(df, col, label_encodings[col])
    
    # Escalar las características numéricas
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    pca_array = pca.transform(df)
    
    return pca_array