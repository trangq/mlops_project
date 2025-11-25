# Sử dụng image MLflow chính thức làm base
FROM ghcr.io/mlflow/mlflow:v2.13.0

# Cài đặt psycopg2-binary
# Chúng ta cài đặt nó dưới tài khoản người dùng root trước, sau đó chuyển lại sang mlflow
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt psycopg2-binary dưới quyền ROOT để tránh lỗi user ID
# Lỗi 'unable to find user mlflow' đã bị loại bỏ
RUN pip install --no-cache-dir psycopg2-binary


# End of Dockerfile - run as the default user from the base image

