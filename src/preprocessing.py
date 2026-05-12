# Import libraries
import pandas as pd

# Sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function for preprocessing
def load_and_preprocess_data():

    # Load dataset
    df = pd.read_csv("data/customers.csv")

    print(df.info())

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert boolean columns into integers
    bool_columns = df.select_dtypes(include='bool').columns

    for col in bool_columns:
        df[col] = df[col].astype(int)

    # Find categorical columns
    categorical_columns = df.select_dtypes(include='object').columns

    print(categorical_columns)

    # Label Encoding
    encoder = LabelEncoder()

    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])

    print(df.info())

    # Features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(X_train.shape)
    print(X_test.shape)

    print("Processed dataset saved successfully")

    # Return processed data
    return X_train, X_test, y_train, y_test


# Run file directly
if __name__ == "__main__":

    load_and_preprocess_data()