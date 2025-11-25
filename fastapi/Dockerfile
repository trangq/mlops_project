FROM python:3.11-slim

WORKDIR /app

# Copy và cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. COPY SOURCE CODE VÀ TESTS

COPY main.py /app/
COPY model_loader.py /app/
COPY tests /app/tests
COPY schemas.py /app/

# 3. EXPOSE PORTS
# 8000: Uvicorn port (API), 8080: Prometheus metrics port
EXPOSE 8000 8080

# 4. RUN SERVER


# Export PYTHONPATH để Python thấy main.py
ENV PYTHONPATH=/app/fastapi

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]