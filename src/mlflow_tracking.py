# Import libraries
import mlflow
import mlflow.sklearn

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score

# Data
from preprocessing import load_and_preprocess_data

# Load dataset
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Create Experiment
mlflow.set_experiment("Customer_Churn_Experiment")

# ==================================================
# Logistic Regression
# ==================================================

with mlflow.start_run(run_name="Logistic_Regression"):

    print("\nTraining Logistic Regression...")

    lr_model = LogisticRegression(max_iter=1000)

    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    # Parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    # Metrics
    mlflow.log_metric("accuracy", accuracy)

    # Model
    mlflow.sklearn.log_model(
        sk_model=lr_model,
        name="logistic_regression_model"
    )

    print("Logistic Regression Logged Successfully")


# ==================================================
# Random Forest
# ==================================================

with mlflow.start_run(run_name="Random_Forest"):

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    # Parameters
    mlflow.log_param("model_type", "RandomForest")

    mlflow.log_param("n_estimators", 100)

    mlflow.log_param("random_state", 42)

    # Metrics
    mlflow.log_metric("accuracy", accuracy)

    # Save Model
    mlflow.sklearn.log_model(
        sk_model=rf_model,
        name="random_forest_model"
    )

    # Log SHAP Summary Plot
    mlflow.log_artifact(
        "models/shap_summary.png"
    )

    print("Random Forest Logged Successfully")


print("\nMLflow Tracking Completed Successfully")