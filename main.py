import streamlit as st
import pandas as pd
from joblib import load

# Load the pre-trained model
model = load('random_forest_classifier.joblib')

# Streamlit app
st.title('Wine Quality Prediction')
st.write('This app predicts the quality of wine as good (1) or not good (0) based on physicochemical tests.')

# Input features
fixed_acidity = st.slider('Fixed Acidity', 4.0, 15.0, 7.4)
volatile_acidity = st.slider('Volatile Acidity', 0.1, 1.5, 0.7)
citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.0)
residual_sugar = st.slider('Residual Sugar', 0.9, 15.0, 1.9)
chlorides = st.slider('Chlorides', 0.012, 0.611, 0.076)
free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1.0, 72.0, 11.0)
total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6.0, 289.0, 34.0)
density = st.slider('Density', 0.990, 1.004, 0.9978)
pH = st.slider('pH', 2.7, 4.01, 3.51)
sulphates = st.slider('Sulphates', 0.33, 2.0, 0.56)
alcohol = st.slider('Alcohol', 8.0, 14.9, 9.4)

# Create a dataframe from user input
input_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# Display input data
st.write('Input data:')
st.write(input_data)

# Make prediction
prediction = model.predict(input_data)

# Set arbitrary cutoff for wine quality
if prediction[0] >= 7:
    st.write('The wine is predicted to be: Good (1)')
else:
    st.write('The wine is predicted to be: Not Good (0)')
