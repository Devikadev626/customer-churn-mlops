# Import libraries
import streamlit as st
import requests

# Streamlit page title
st.title("Customer Churn Prediction System")

# Subtitle
st.write("Enter customer details to predict churn")

# User input fields
gender = st.selectbox("Gender", [0, 1])

senior_citizen = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", [0, 1])

dependents = st.selectbox("Dependents", [0, 1])

tenure = st.slider("Tenure", 0, 72)

phone_service = st.selectbox("Phone Service", [0, 1])

multiple_lines = st.selectbox("Multiple Lines", [0, 1])

internet_service = st.selectbox("Internet Service", [0, 1, 2])

online_security = st.selectbox("Online Security", [0, 1])

online_backup = st.selectbox("Online Backup", [0, 1])

device_protection = st.selectbox("Device Protection", [0, 1])

tech_support = st.selectbox("Tech Support", [0, 1])

streaming_tv = st.selectbox("Streaming TV", [0, 1])

streaming_movies = st.selectbox("Streaming Movies", [0, 1])

contract = st.selectbox("Contract", [0, 1, 2])

paperless_billing = st.selectbox("Paperless Billing", [0, 1])

payment_method = st.selectbox("Payment Method", [0, 1, 2, 3])

monthly_charges = st.number_input("Monthly Charges")

total_charges = st.number_input("Total Charges")

# Predict button
if st.button("Predict Churn"):

    # Create input dictionary
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Send POST request to FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )

    # Get prediction result
    result = response.json()

    # Display prediction
    st.success(f"Prediction: {result['prediction']}")