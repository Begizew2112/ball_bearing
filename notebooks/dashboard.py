import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Load your pre-trained model and scaler (ensure the paths are correct)
model = load_model(r'C:\Users\Yibabe\Desktop\ball_bearing\notebooks\bearing_rul_model.keras')
scaler = joblib.load(r'C:\Users\Yibabe\Desktop\ball_bearing\notebooks\scaler_model.joblib')

# Streamlit UI
st.title("Bearing Remaining Useful Life (RUL) Prediction")
st.write("Upload your bearing time series data and get interactive insights into the Remaining Useful Life!")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    workshop_data = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Data")
    st.write("Preview of the data:")
    st.write(workshop_data.head())
    
    st.write("### Descriptive Statistics")
    st.write(workshop_data.describe())

    # Preprocessing the uploaded data for predictions
    st.write("### Processing the Data")

    # Define window size
    window_size = 50  # This must match the training window size
    windows = [
        workshop_data[i:i + window_size].values
        for i in range(0, len(workshop_data) - window_size + 1, window_size)
    ]
    X_workshop = np.array(windows)  # Shape: (number_of_windows, window_size, features)
    
    # Scale the data
    X_workshop_scaled = scaler.transform(X_workshop.reshape(-1, X_workshop.shape[-1])).reshape(X_workshop.shape)
    
    # Make predictions
    predictions = model.predict(X_workshop_scaled)

    # Display predictions
    st.write("### Predictions")
    st.write("Predicted Remaining Useful Life (RUL) values for each time window:")
    predictions_df = pd.DataFrame(predictions, columns=["Predicted_RUL"])
    st.write(predictions_df)

    # Allow downloading of predictions
    csv_data = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_data,
        file_name="predicted_rul.csv",
        mime="text/csv"
    )

    # Interactive plotting with Plotly
    st.write("### Interactive Plot of Predicted Remaining Useful Life (RUL)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=predictions.flatten(),
        mode='lines+markers',
        name="Predicted RUL"
    ))
    fig.update_layout(
        title="Predicted Remaining Useful Life (RUL) of Bearings",
        xaxis_title="Time Steps",
        yaxis_title="Remaining Useful Life (RUL)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highlight key insights
    st.write("### Key Insights")
    max_rul = predictions.max()
    min_rul = predictions.min()
    avg_rul = predictions.mean()
    st.metric(label="Maximum Predicted RUL", value=f"{max_rul:.2f}")
    st.metric(label="Minimum Predicted RUL", value=f"{min_rul:.2f}")
    st.metric(label="Average Predicted RUL", value=f"{avg_rul:.2f}")

    st.write("### Summary")
    st.write(f"The predictions indicate that the remaining useful life (RUL) of the bearings ranges from approximately **{min_rul:.2f}** to **{max_rul:.2f}** units, with an average of **{avg_rul:.2f}** units.")
import numpy as np 