# src/train_model.py
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from termcolor import colored


def train_model(model_type="random_forest"):
    df = pd.read_csv("data/Customer-Churn.csv").dropna()
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])

    selected_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "Contract", "InternetService", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    X = df[selected_features]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
        experiment_name = "Telco Churn Prediction (XGBoost)"
        model_filename = "models/xgb_model.pkl"
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        experiment_name = "Telco Churn Prediction (RandomForest)"
        model_filename = "models/rf_model.pkl"

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(model, model_filename)
        joblib.dump(scaler, "models/scaler.pkl")
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("\n" + "=" * 60)
        print(colored(f"✅ TRAINING SUMMARY ({model_type.upper()} MODEL)", "green", attrs=["bold"]))
        print("-" * 60)
        print(f"📊 Accuracy       : {acc * 100:.2f}%")
        print(f"🎯 F1-score       : {f1 * 100:.2f}%")
        print(f"💾 Saved To       : {model_filename}")
        print("=" * 60 + "\n")

        print(colored("\n📋 Classification Report:\n", "yellow"))
        print(classification_report(y_test, preds))

    print(colored("✅ Training Complete!\n", "green", attrs=["bold"]))


if __name__ == "__main__":
    # Train both for comparison
    train_model("random_forest")
    train_model("xgboost")
