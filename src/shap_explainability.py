# Import libraries
import shap
import joblib
import matplotlib.pyplot as plt

# Import preprocessing function
from preprocessing import load_and_preprocess_data

# Load processed data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Generate SHAP values
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

# Save plot
plt.savefig("models/shap_summary.png")

print("\nSHAP summary plot saved successfully.")