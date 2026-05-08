import pandas as pd

# Load dataset
df = pd.read_csv("data/customers.csv")

# Display first 5 rows
print(df.head())

# Shape
print(df.shape)

# Columns
print(df.columns)


print(df.info())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())    

# Target variable distribution
print(df["Churn"].value_counts())

# save cleaned data
df.to_csv("data/cleaned_customers.csv", index=False)