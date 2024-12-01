import streamlit as st
import pickle
import numpy as np

# Load the trained model
MODEL_PATH = r'C:\Users\Yibabe\Desktop\ball_bearing\notebooks\optimized_rul_model.pkl'

# Load the model with error handling
def load_model(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please check the provided path.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        st.stop()

model = load_model(MODEL_PATH)

# Set up the Streamlit app
st.title("Ball Bearing Remaining Useful Life (RUL) Prediction")
st.markdown(
    """This interactive dashboard predicts the Remaining Useful Life (RUL) of a ball bearing based on vibration and temperature data. 
    Please provide the required input values below."""
)

# Collect user inputs
st.sidebar.header("Input Features")
vibration_x = st.sidebar.number_input("Vibration (X-direction) [g]", value=0.0, format="%.6f")
vibration_y = st.sidebar.number_input("Vibration (Y-direction) [g]", value=0.0, format="%.6f")
temperature_bearing = st.sidebar.number_input("Temperature (Bearing) [°C]", value=40.0, format="%.2f")

# Prepare the input data
input_data = np.array([[vibration_x, vibration_y, temperature_bearing]])

# Validate the input and make a prediction
if st.sidebar.button("Predict RUL"):
    try:
        # Ensure the input matches the model's expectations
        if input_data.shape[1] != 3:
            st.error("Error: The model expects exactly 3 input features.")
        else:
            prediction = model.predict(input_data)
            st.success(f"Predicted Remaining Useful Life (RUL): {prediction[0]:.2f} hours")
    except AttributeError as e:
        st.error(f"Model is not correctly loaded or does not support predictions: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "Developed by **Yibabe**, this tool utilizes machine learning to assist in predictive maintenance."
)
