import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("heart_disease_model.joblib")

# Initialize the FastAPI app
app = FastAPI()

# Define the input data schema using Pydantic
class HeartDiseaseData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Root endpoint
@app.get('/')
def index():
    return {'message': 'Heart Disease Prediction API'}

# Prediction endpoint
@app.post('/predict')
def predict_heart_disease(data: HeartDiseaseData):
    data = data.dict()
    input_data = [
        data['age'],
        data['sex'],
        data['cp'],
        data['trestbps'],
        data['chol'],
        data['fbs'],
        data['restecg'],
        data['thalach'],
        data['exang'],
        data['oldpeak'],
        data['slope'],
        data['ca'],
        data['thal']
    ]

    # Convert input data to numpy array and reshape for prediction
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data_as_numpy_array)

    # Interpret the prediction
    if prediction[0] == 0:
        result = 'The Person does not have Heart Disease'
    else:
        result = 'The Person has Heart Disease'

    return {'prediction': result}

# Run the FastAPI app
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
