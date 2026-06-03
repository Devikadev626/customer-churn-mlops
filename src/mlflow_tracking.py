# Import libraries
import mlflow
import mlflow.sklearn

# Sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score

# Import preprocessing function
from preprocessing import load_and_preprocess_data

# Load processed data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Create MLflow experiment
mlflow.set_experiment("Customer_Churn_Experiment")

# ==========================================
# Logistic Regression Experiment
# ==========================================

with mlflow.start_run(run_name="Logistic_Regression"):

    # Create model
    lr_model = LogisticRegression(max_iter=1000)

    # Train model
    lr_model.fit(X_train, y_train)

    # Predictions
    y_pred = lr_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print result
    print("\nLogistic Regression Accuracy:")
    print(accuracy)

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    mlflow.sklearn.log_model(
    sk_model=lr_model,
    name="logistic_regression_model"
)

# ==========================================
# Random Forest Experiment
# ==========================================

with mlflow.start_run(run_name="Random_Forest"):

    # Create model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print result
    print("\nRandom Forest Accuracy:")
    print(accuracy)

    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    mlflow.sklearn.log_model(
    sk_model=rf_model,
    name="random_forest_model"
)

print("\nMLflow tracking completed successfully.")

mlflow.log_artifact("models/shap_summary.png")