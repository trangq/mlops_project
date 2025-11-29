FROM python:3.11-slim

WORKDIR /app

# Copy và cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. COPY SOURCE CODE VÀ TESTS

COPY api/main.py /app/
COPY api/model_loader.py /app/
COPY api/tests /app/tests
COPY api/schemas.py /app/ 



# 3. EXPOSE PORTS
# 8000: Uvicorn port (API), 8080: Prometheus metrics port
EXPOSE 8000 8080

# 4. RUN SERVER


# Export PYTHONPATH để Python thấy main.py
# ENV PYTHONPATH=/app/api

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]