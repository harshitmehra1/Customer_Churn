# src/pipeline_flow.py
from termcolor import colored
from src.train_model import train_model


def churn_pipeline():
    print(colored("🚀 Starting Telco Churn ML pipeline...", "cyan"))

    try:
        train_model()
        print(colored("🏁 Pipeline execution completed successfully.", "green"))
    except Exception as e:
        print(colored(f"❌ Pipeline failed: {e}", "red"))


if __name__ == "__main__":
    churn_pipeline()
