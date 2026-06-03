import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/customers.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# Remove customerID if exists
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges if present
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Remove null values
df.dropna(inplace=True)

print("\nShape After Cleaning:")
print(df.shape)

# ==========================
# TARGET VARIABLE HANDLING
# ==========================

print("\nUnique Churn Values Before Processing:")
print(df["Churn"].unique())

print("\nChurn Data Type:")
print(df["Churn"].dtype)

# If Churn is Boolean
if df["Churn"].dtype == bool:
    df["Churn"] = df["Churn"].astype(int)

# If Churn is Yes/No
elif df["Churn"].dtype == object:
    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .map({
            "Yes": 1,
            "No": 0
        })
    )

print("\nChurn Distribution After Conversion:")
print(df["Churn"].value_counts())

print("\nNull Values After Conversion:")
print(df["Churn"].isnull().sum())

# Remove rows with invalid target values
df = df.dropna(subset=["Churn"])

print("\nFinal Dataset Shape:")
print(df.shape)

# ==========================
# FEATURE ENGINEERING
# ==========================

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# Safety Check
if len(df) == 0:
    raise ValueError(
        "Dataset became empty after preprocessing."
    )

if y.isnull().sum() > 0:
    raise ValueError(
        "Target variable still contains NULL values."
    )

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================
# MODEL TRAINING
# ==========================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully")

# ==========================
# EVIDENTLY REPORT
# ==========================

print("\nRunning Evidently Report...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=X_train,
    current_data=X_test
)
os.makedirs("reports", exist_ok=True)



# ==========================
# SAVE REPORT
# ==========================

report.save_html(
    "reports/drift_report.html"
)

print("\nDrift Report Generated Successfully")
print("Location: reports/drift_report.html")