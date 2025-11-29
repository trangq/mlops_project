import os
import mlflow
import mlflow.sklearn
import socket
from mlflow.tracking import MlflowClient

import os
MODEL_NAME = os.getenv("MODEL_NAME", "TitanicClassifier")
def get_mlflow_uri():
    try:
        socket.gethostbyname("mlflow-server")
        return "http://mlflow-server:5000"
    except socket.gaierror:
        # fallback sang local folder mlruns
        return f"file:{os.path.abspath('./mlruns')}"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", get_mlflow_uri())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(f"{MODEL_NAME}_Experiment")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def load_model():
    """Load the latest Production model from MLflow Model Registry (alias 'prod')"""
    # Lấy tất cả version mới nhất
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=None)
    if not latest_versions:
        raise RuntimeError(f"No versions found for model {MODEL_NAME}")

    # Lấy version mới nhất
    model_version = latest_versions[0].version
    print(f">>> Loading model {MODEL_NAME} version {model_version}")

    try:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            version=model_version,
            alias="prod"
        )
        print(f">>> Model {MODEL_NAME} v{model_version} set as 'prod'")
    except Exception as e:
        print(f"Warning: Could not set alias 'prod': {e}")

    # Load model từ URI chuẩn
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model
