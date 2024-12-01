import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib

# Load your pre-trained model and scaler (ensure the paths are correct)
model = load_model(r'C:\Users\Yibabe\Desktop\ball_bearing\notebooks\bearing_rul_model.keras')
scaler = joblib.load(r'C:\Users\Yibabe\Desktop\ball_bearing\notebooks\scaler_model.joblib')

# Streamlit UI
st.title("Bearing Remaining Useful Life (RUL) Prediction")
st.write("Upload your bearing time series data and get the RUL prediction!")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    workshop_data = pd.read_csv(uploaded_file)

    # Display the uploaded data for inspection
    st.write("### Uploaded Data", workshop_data.head())

    # Preprocessing the uploaded data for predictions
    st.write("### Processing the Data")

    # Assuming the data needs to be reshaped as windows
    window_size = 50  # Define your window size (same as during training)
    windows = [
        workshop_data[i:i + window_size].values  # Ensure proper data extraction
        for i in range(0, len(workshop_data) - window_size + 1, window_size)
    ]
    X_workshop = np.array(windows)  # Shape: (number_of_windows, window_size, features)
    
    # Scale the workshop data using the fitted scaler
    X_workshop_scaled = scaler.transform(X_workshop.reshape(-1, X_workshop.shape[-1])).reshape(X_workshop.shape)
    
    # Make predictions
    predictions = model.predict(X_workshop_scaled)
    
    # Display predictions
    st.write("### Predictions", predictions)

    # Optionally, plot the results
    st.write("### Plotting the Remaining Useful Life (RUL) Prediction")
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predicted RUL")
    plt.title("Predicted Remaining Useful Life (RUL) of Bearing")
    plt.xlabel("Time Steps")
    plt.ylabel("Remaining Useful Life (RUL)")
    plt.legend()
    st.pyplot()

    # Show metrics (if necessary)
    st.write(f"### Predicted RUL for First 5 Samples: {predictions[:5]}")
    st.write(f"### True Values (if available for comparison): [0.26666667, 0.57692308, 0.58461538, 0.76666667, 0.55128205]")
