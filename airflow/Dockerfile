# Cố định Airflow + Python version (RẤT QUAN TRỌNG)
FROM apache/airflow:2.9.2-python3.10

# ======================
# System dependencies
# ======================
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ======================
# Python dependencies
# ======================
USER airflow

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    boto3 \
    dvc[s3] \
    -r /requirements.txt

# ======================
# Verify (optional nhưng nên có)
# ======================
RUN airflow version && \
    dvc version && \
    python - <<EOF
import mlflow
print("MLflow version:", mlflow.__version__)
EOF
