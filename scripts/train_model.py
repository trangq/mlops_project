import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow import MlflowClient
import socket

MODEL_NAME = "TitanicClassifier"

# Luôn dùng host của container MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Tạo experiment nếu chưa có
mlflow.set_experiment(f"{MODEL_NAME}_Experiment")



def train(input_file: str):
    print(f"Loading processed data from: {input_file}")
    df = pd.read_csv(input_file)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        params = {"C": 1.0, "kernel": "rbf", "random_state": 42}

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(**params))
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # ---- Log + Register ----
        print("\n=== Logging model to MLflow ===")

      
            # Log model
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(1),
        )

        # Lấy client 1 lần
        client = MlflowClient()

        # Lấy version mới nhất đúng
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=None)
        if len(latest_versions) == 0:
            raise RuntimeError(f"No versions found for model {MODEL_NAME}")

        model_version = latest_versions[0].version
        print(f">>> Registered model version = {model_version}")

        # Gán alias 'prod' (tương đương Production)
        client.set_registered_model_alias(
            name=MODEL_NAME,
            version=model_version,
            alias="prod"
        )

        print(f">>> Model {MODEL_NAME} v{model_version} moved to Production.")
        return acc
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: Missing input path for training data.")
        print("Usage: python train_model.py <cleaned_csv_path>")
        sys.exit(1)

    train(sys.argv[1])
