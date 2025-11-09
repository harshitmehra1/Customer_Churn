# src/train_model.py
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.data_preprocessing import load_and_preprocess_data

from termcolor import colored                               # 🟢 For colored console output


def train_model():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/Customer-Churn.csv")

    # Set MLflow experiment
    mlflow.set_experiment("Telco Churn Prediction")

    with mlflow.start_run():
        # Train baseline Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # 🧾 Pretty output summary
        print("\n" + "=" * 60)
        print(colored("✅ TRAINING SUMMARY", "green", attrs=["bold"]))
        print("-" * 60)
        print(f"📊 Accuracy       : {acc * 100:.2f}%")
        print(f"🎯 F1-score       : {f1 * 100:.2f}%")
        print(f"🧮 Train Samples  : {len(X_train):,}")
        print(f"🧪 Test Samples   : {len(X_test):,}")
        print(f"💾 Model Saved To : models/baseline_model.pkl")
        print("-" * 60)
        print(colored("Logged to MLflow experiment: Telco Churn Prediction", "cyan"))
        print("=" * 60 + "\n")

        # Log to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save model and log to MLflow
        joblib.dump(model, "models/baseline_model.pkl")
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Optional detailed report
        print(colored("\n📋 Classification Report:\n", "yellow"))
        print(classification_report(y_test, preds))

    print(colored("✅ Training Complete!\n", "green", attrs=["bold"]))


if __name__ == "__main__":
    train_model()
