FROM python:3.11-slim

WORKDIR /app

# 1. Cài đặt dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy toàn bộ thư mục app vào /app/app
# Điều này tạo ra cấu trúc: /app/app/api/main.py
COPY app/ /app/app/

# 3. Biến môi trường để Python tìm thấy module 'app'
ENV PYTHONPATH=/app

EXPOSE 8000

# 4. Chạy app bằng module path
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]