import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/customers.csv")

# Dataset info
print(df.info())

# Convert boolean columns into integers
bool_cols = df.select_dtypes(include=["bool"]).columns

for col in bool_cols:
    df[col] = df[col].astype(int)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns

print(categorical_cols)

# Label encoding
encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Verify transformed dataset
print(df.info())

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Shapes
print(X_train.shape)
print(X_test.shape)

# Save processed dataset
df.to_csv("data/processed_customers.csv", index=False)

print("Processed dataset saved successfully")