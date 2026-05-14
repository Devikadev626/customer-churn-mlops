# Import libraries
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Home route
@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running successfully"
    }

# Prediction route
@app.post("/predict")
def predict(data: dict):

    try:
        # Convert input data into DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Prediction probability
        probability = model.predict_proba(df)[0][1]

        # Convert prediction into readable format
        result = "Churn" if prediction == 1 else "No Churn"

        # Logging
        logging.info("Prediction completed successfully")

        # Return response
        return {
            "status": "success",
            "prediction": result,
            "prediction_code": int(prediction),
            "churn_probability": round(float(probability), 2)
        }

    except Exception as e:

        logging.error(f"Prediction error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )