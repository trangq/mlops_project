# Cố định Airflow + Python version
FROM apache/airflow:2.9.2-python3.10

# ======================
# System dependencies (Chạy quyền root)
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
# Python dependencies (Chạy quyền airflow)
# ======================
USER airflow

# SỬA TẠI ĐÂY: Thêm --chown để user airflow có quyền đọc file requirements
COPY --chown=airflow:root requirements.txt /requirements.txt

# Cài đặt các thư viện cần thiết
# Lưu ý: Thêm apache-airflow-providers-amazon để Airflow kết nối MinIO/S3 mượt hơn
RUN pip install --no-cache-dir \
    mlflow==2.11.3 \
    pandas \
    scikit-learn \
    boto3 \
    dvc[s3] \
    apache-airflow-providers-amazon \
    -r /requirements.txt

# ======================
# Verify
# ======================
# SỬA TẠI ĐÂY: Dùng lệnh đơn giản để kiểm tra, tránh dùng EOF phức tạp trong RUN
RUN airflow version && \
    dvc version && \
    python -c "import mlflow; print('MLflow version:', mlflow.__version__)"