import os
import sys
import socket
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --- Config ---
MODEL_NAME = os.getenv("MODEL_NAME", "TitanicClassifier")

def get_mlflow_uri():
    try:
        socket.gethostbyname("mlflow-server")
        return "http://mlflow-server:5000"
    except socket.gaierror:
        return "http://localhost:5001"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", get_mlflow_uri())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(f"{MODEL_NAME}_Experiment")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# --- Model Loader ---
def load_model():
    """Load the latest Production model from MLflow Model Registry (alias 'prod')"""
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=None)
    if not latest_versions:
        raise RuntimeError(f"No versions found for model {MODEL_NAME}")

    model_version = latest_versions[0].version
    print(f">>> Loading model {MODEL_NAME} version {model_version}")

    # Optional: set alias 'prod'
    try:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            version=model_version,
            alias="prod"
        )
        print(f">>> Model {MODEL_NAME} v{model_version} set as 'prod'")
    except Exception as e:
        print(f"Warning: Could not set alias 'prod': {e}")

    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    return mlflow.sklearn.load_model(model_uri)


# --- Evaluation + Promote ---
def evaluate(test_file="processed_data.csv", threshold=0.8):
    df = pd.read_csv(test_file)
    X = df[[c for c in df.columns if c != "target"]]
    y = df["target"]

    model = load_model()
    acc = (model.predict(X) == y).mean()
    print(f"Accuracy: {acc:.4f}")

    if acc >= threshold:
        latest = client.get_latest_versions(MODEL_NAME, stages=None)[0]
        client.set_registered_model_alias(
            name=MODEL_NAME,
            version=latest.version,
            alias="prod"
        )
        print(f">>> Model {MODEL_NAME} v{latest.version} promoted to Production")
    else:
        print(f">>> Accuracy below threshold ({threshold}). Not promoting.")

    return acc


# --- Main ---
if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "processed_data.csv"
    evaluate(test_file)
