# MLOps Project

This repo contains a minimal MLOps project skeleton:

- `docker-compose.yaml` - orchestrates core services (MinIO, MLflow, Airflow, FastAPI, Prometheus, Grafana)
- `.env` - environment variables and credentials placeholder
- `scripts/` - contains ML model training and preprocessing steps (MLflow logging)
- `airflow/` - Dockerfile and DAGs to schedule the pipeline
- `mlflow-server/` - placeholder for MLflow server setup
- `fastapi/` - API container for serving model / metrics
- `monitoring/` - Prometheus and Grafana configurations

This is a scaffold to help you run a lightweight MLOps stack for experimentation.

Quick start (high-level):

1. Populate `.env` with credentials (MinIO, MLflow config).
2. Start the stack: `docker compose up --build` using the provided `docker-compose.yaml`.
3. Run the Airflow DAG or trigger the training manually.
4. Use the FastAPI endpoint for predictions and the metrics endpoint for monitoring.

See the individual folders for more details.
