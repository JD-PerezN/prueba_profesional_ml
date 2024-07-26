from utils.preprocessing import preprocess_input
from joblib import load
from flask import Blueprint, request, jsonify

xgb_classifier = load('models\\xgb_classifier.joblib')
xgb_regressor = load('models\\xgb_regressor.joblib')

api = Blueprint("api", __name__)

@api.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X_pca = preprocess_input(data)

    class_prediction = xgb_classifier.predict(X_pca)[0]
    demand_prediction = xgb_regressor.predict(X_pca)[0]

    mapping = {
        0: "Alpha",
        1: "Betha"
    }

    real_class = mapping.get(class_prediction)

    response = {
            'class': real_class,
            'demand': str(demand_prediction)
        }

    return jsonify(response)