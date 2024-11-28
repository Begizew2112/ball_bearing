import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("optimized_rul_model.pkl")

# App title
st.title("Bearing Remaining Useful Life (RUL) Prediction")

# Input fields
st.header("Enter Bearing Data:")
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)
# Add more input fields as per your dataset

# Predict button
if st.button("Predict RUL"):
    # Prepare input for the model
    features = np.array([[feature_1, feature_2, feature_3, feature_4]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Remaining Useful Life: {prediction:.2f} hours")
