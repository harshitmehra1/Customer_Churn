# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath="data/Customer-Churn.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)

    # Drop customer ID if present
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric and drop rows with missing TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    df = df.dropna().reset_index(drop=True)

    # Encode object (categorical) columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # Separate features and target
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in the dataset.")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test



print("✅ preprocessing done ")