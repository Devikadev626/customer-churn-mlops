# Import libraries
import streamlit as st
import requests

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ==========================
# Title
# ==========================

st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer details below to predict whether the customer is likely to churn."
)

# ==========================
# Customer Information
# ==========================

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)
gender = 1 if gender == "Male" else 0

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)
senior_citizen = 1 if senior_citizen == "Yes" else 0

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)
partner = 1 if partner == "Yes" else 0

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)
dependents = 1 if dependents == "Yes" else 0

tenure = st.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

# ==========================
# Services
# ==========================

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)
phone_service = 1 if phone_service == "Yes" else 0

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes"]
)
multiple_lines = 1 if multiple_lines == "Yes" else 0

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber Optic",
        "No Internet"
    ]
)

internet_mapping = {
    "DSL": 0,
    "Fiber Optic": 1,
    "No Internet": 2
}

internet_service = internet_mapping[internet_service]

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes"]
)
online_security = 1 if online_security == "Yes" else 0

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes"]
)
online_backup = 1 if online_backup == "Yes" else 0

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes"]
)
device_protection = 1 if device_protection == "Yes" else 0

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes"]
)
tech_support = 1 if tech_support == "Yes" else 0

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes"]
)
streaming_tv = 1 if streaming_tv == "Yes" else 0

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes"]
)
streaming_movies = 1 if streaming_movies == "Yes" else 0

# ==========================
# Contract Information
# ==========================

contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-Month",
        "One Year",
        "Two Year"
    ]
)

contract_mapping = {
    "Month-to-Month": 0,
    "One Year": 1,
    "Two Year": 2
}

contract = contract_mapping[contract]

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)
paperless_billing = 1 if paperless_billing == "Yes" else 0

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic Check",
        "Mailed Check",
        "Bank Transfer",
        "Credit Card"
    ]
)

payment_mapping = {
    "Electronic Check": 0,
    "Mailed Check": 1,
    "Bank Transfer": 2,
    "Credit Card": 3
}

payment_method = payment_mapping[payment_method]

# ==========================
# Billing Information
# ==========================

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# ==========================
# Prediction Button
# ==========================

if st.button("Predict Churn"):

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

    try:

        response = requests.post(
            "http://localhost:8000/predict",
            json=input_data
        )

        if response.status_code == 200:

            result = response.json()
            prediction = result["prediction"]

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠ Customer is likely to Churn"
                )

                st.warning(
                    "Recommended Action: Offer discounts, loyalty benefits, or customer support intervention."
                )

            else:

                st.success(
                    "✅ Customer is not likely to Churn"
                )

                st.info(
                    "Customer appears satisfied and likely to remain with the service."
                )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except Exception as e:

        st.error(
            f"Connection Error: {str(e)}"
        )