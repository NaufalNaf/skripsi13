import joblib
import streamlit as st

# Load the saved model and scaler
model = joblib.load('klasifikasi_obesitas_svm.pkl')
scaler = joblib.load('scaler.pkl')

# Define the application layout
st.title('BMI Classification App')

# Get user input
gender = st.radio('Gender', ['Male', 'Female'])
height = st.number_input('Height (cm)', min_value=0.0)
weight = st.number_input('Weight (kg)', min_value=0.0)

# Preprocess the user input
input_data = [[gender, height, weight]]
input_scaled = scaler.transform(input_data)

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_scaled)[0]
    index_labels = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obese',
        5: 'Extremely Obese'
    }
    st.write(f'Predicted BMI Category: {index_labels[prediction]}')

