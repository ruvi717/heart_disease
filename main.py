import numpy as np
import pickle
import streamlit as st

# Load the trained model from pickle file
with open('trained_heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_heart_disease(input_data):
    # Convert input data to numpy array and reshape for prediction
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    
    return prediction[0]

# Streamlit app UI
st.title("Heart Disease Prediction App")

# Collecting user input
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type (0-3)', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', options=[0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results (0-2)', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', options=[0, 1])
oldpeak = st.number_input('ST depression induced by exercise relative to rest (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the peak exercise ST segment (0-2)', options=[0, 1, 2])
ca = st.number_input('Number of major vessels (0-3) colored by fluoroscopy (ca)', min_value=0, max_value=3, value=0)
thal = st.selectbox('Thal (3 = normal; 6 = fixed defect; 7 = reversible defect)', options=[3, 6, 7])

# Store input data in a tuple
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

# Prediction button
if st.button("Predict"):
    prediction = predict_heart_disease(input_data)
    
    if prediction == 0:
        st.success("The Person does not have Heart Disease")
    else:
        st.warning("The Person has Heart Disease")

