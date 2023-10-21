import joblib
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

model = joblib.load('hybrid_predictions.pkl')

# Mapping of string values to encoded values
gender_mapping = {'Male': 0, 'Female': 1}
education_level_mapping = {"Bachelor's": 1, "Master's": 2, "PhD": 3}

# Load your dataset to extract unique job titles
dataset = pd.read_csv('SalaryData.csv')
unique_job_titles = dataset['Job_Title'].unique()

# Generate the job_title_mapping based on the unique job titles
job_title_mapping = {job_title: [0] * len(unique_job_titles) for job_title in unique_job_titles}

@app.route('/')
def home():
    return "Welcome to Salary Prediction API"

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    try:
        input_data = request.get_json()

        # Map string values to encoded values
        gender = gender_mapping.get(input_data['Gender'], None)
        education_level = education_level_mapping.get(input_data['Education_Level'], None)

        if gender is None or education_level is None:
            return jsonify({'error': 'Invalid input for Gender or Education_Level'})

        # Retrieve the user's job title and convert it to the encoded format
        user_job_title = input_data['Job_Title']
        job_title = job_title_mapping.get(user_job_title, [0] * len(unique_job_titles))

        if not job_title:
            return jsonify({'error': 'Invalid input for Job_Title'})

        # Continue with the prediction
        prediction = model.predict([[
            input_data['Age'],
            gender,
            education_level,
            *job_title,
            input_data['Years_of_Experience']
        ]])

        response = {'predicted_salary': prediction[0]}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
