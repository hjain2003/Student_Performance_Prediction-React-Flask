from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load your trained machine learning model
model = joblib.load('linear_regression_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to Student Performance API"

@app.route('/predict', methods=['POST'])
def predict_score():
    try:
        data = request.get_json()

        input_data = [[
            data['hours_studied'],
            data['previous_score'], 
            data['extracurricular_activities'],
            data['sleep_hours'], 
            data['sample_papers_solved']
        ]]

        print("Input data:", input_data)

        # Scale the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)
        print("Scaled data:", input_data_scaled)

        # Make predictions using the model
        predicted_score = model.predict(input_data_scaled)[0]
        rounded_score = round(predicted_score, 2)
        print("Predicted score:", predicted_score)

        return jsonify({'predicted_score': rounded_score})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
