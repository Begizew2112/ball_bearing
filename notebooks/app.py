from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("optimized_rul_model.pkl")

# Define the app
app = FastAPI()

# Input schema
class BearingData(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    # Add all necessary features used in training

@app.post("/predict")
def predict(data: BearingData):
    # Convert input to model format
    features = np.array([[data.feature_1, data.feature_2, data.feature_3, data.feature_4]])
    prediction = model.predict(features)[0]
    return {"remaining_useful_life": prediction}
