# prueba_profesional_ml

Repositorio

## Tabla de Contenidos

- [Instalación](#instalación)
- [Uso](#uso)
- [Características](#características)
- [Licencia](#licencia)
- [Contacto](#contacto)

## Instalación

### Requisitos Previos

- Python >3.9
- Git 

### Clonar el Repositorio
```bash
git clone https://github.com/JD-PerezN/prueba_profesional_ml
```

### Creación y activación del ambiente virtual
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Instalación de las librerías
```bash
pip install -r requirements.txt
```

### Navegación al directorio del proyecto
```bash
cd code
```

## Uso

### Estructura de carpetas
```markdown
PRUEBA_PROFESIONAL_ML/
├── .venv/
├── code/
│   ├── api/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   ├── data/
│   │   ├── dataset_alpha_betha.csv
│   │   └── dataset_demand_acumulate.csv
│   ├── models/
│   │   ├── pca_transformer.joblib
│   │   ├── standard_scaler.joblib
│   │   ├── xgb_classifier.joblib
│   │   └── xgb_regressor.joblib
│   ├── utils/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── classification_demand.ipynb
│   │   ├── preprocessing.py
│   │   └── time_series_demand.ipynb
│   └── main.py
├── devops/
│   └── dockerfile
├── .env
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── teoria_.pdf
```
### Carpeta models
Contiene los modelos de regresión (xgb_classifier.joblib), clasificación (xgb_classifier.joblib), PCA con los datos entrenados (pca_transformer.joblib) y la normalización de los datos entrenados (standard_scaler.joblib)

### Carpeta api
Contiene el archivo endpoints.py en el cual se realiza la predicción de los valores a partir de un archivo json

### Carpeta utils
Contiene los 2 notebooks de ML con el archivo preprocessing.py el cual realiza un preprocesamiento de los datos ingresados por json para generar un array de salida el cual llegará al endpoint para la predicción

### Modelo de series temporales
El notebook time_series_demand.ipynb contiene el modelo para predecir la demanda mensual del archivo dataset_demand_acumulate.csv

### Modelo de clasificación y regresión
El notebook classification_demand.ipynb contiene el modelo para predecir el valor de la clase clasificación y el valor de la demanda mediante regresión. Estos modelos se guardan en la carpeta El archivo con el que se entrenó los modelos es dataset_alpha_betha.csv 

## Características
* Modelo de series temporales para predecir la demanda mensual.
* Modelo de clasificación y regresión para predecir tanto la demanda como la clase.
* API para consumir los modelos guardados de clasificación y regresión.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT - mira el archivo LICENSE para más detalles.

## Contacto
Julián David Pérez Navarro - jd.perez998@gmail.com

Enlace del Proyecto: https://github.com/JD-PerezN/prueba_profesional_ml
