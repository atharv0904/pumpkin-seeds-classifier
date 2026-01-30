from flask import Blueprint, render_template, request, flash
import numpy as np
import joblib
import os

routes = Blueprint('routes', __name__, template_folder='../templates')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'model', 'RandomForestClassifier_model')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'static', 'model', 'scaler')

FEATURES = [
    'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
    'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity',
    'Extent', 'Roundness', 'Aspect_Ration', 'Compactness'
]

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# 🔹 Landing Page
@routes.route('/')
def index():
    return render_template('index.html')


# 🔹 Prediction Form Page
@routes.route('/predict')
def predict_page():
    return render_template('predict.html')


# 🔹 Prediction Logic
@routes.route('/predict/result', methods=['POST'])
def predict():
    try:
        input_data = []

        for feature in FEATURES:
            input_data.append(float(request.form.get(feature)))

        input_array = np.array([input_data])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        input_summary = dict(zip(FEATURES, input_data))

        return render_template(
            'result.html',
            prediction=prediction,
            input_summary=input_summary
        )

    except Exception as e:
        flash(str(e))
        return render_template('predict.html')
