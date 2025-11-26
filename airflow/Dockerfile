# Sử dụng base image của Airflow
FROM apache/airflow:2.9.2-python3.10

RUN pip install --no-cache-dir mlflow pandas scikit-learn boto3


# Cài đặt các dependencies cần thiết
USER root
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt DVC với hỗ trợ S3 và các dependencies Python khác
USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install dvc[s3]