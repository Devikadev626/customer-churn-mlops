from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("models/churn_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API Running"
    }


@app.post("/predict")
def predict(data: dict):

    try:
        print("\nReceived Data:")
        print(data)

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        print("\nInput DataFrame:")
        print(df)

        print("\nColumns:")
        print(df.columns.tolist())

        # Prediction
        prediction = model.predict(df)

        result = "Churn" if prediction[0] == 1 else "No Churn"

        return {
            "prediction": int(prediction[0]),
            "result": result
        }

    except Exception as e:

        print("\nERROR:")
        print(str(e))

        return {
            "status": "error",
            "message": str(e)
        }