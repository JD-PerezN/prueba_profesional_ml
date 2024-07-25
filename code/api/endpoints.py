from utils.preprocessing import preprocess_input
from joblib import load

xgb_classifier = load('code\\models\\xgb_classifier.joblib')
svr_regressor = load('code\\models\\svr_regressor.joblib')

path = "code\\api\\to_predict.csv"

def prueba():
    X_pca = preprocess_input(path)

    class_prediction = xgb_classifier.predict(X_pca)[0]
    demand_prediction = svr_regressor.predict(X_pca)[0]

    response = {
            'class': class_prediction,
            'demand': float(demand_prediction)
        }

    return (response)