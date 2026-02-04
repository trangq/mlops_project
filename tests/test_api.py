import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
# KHÔNG dùng sys.path.append nữa
from app.api.main import app  # Import trực tiếp từ gốc


# Giả lập hàm load_model để không kết nối tới MLflow/S3
with patch("app.api.model_loader.load_model") as mock_load:
    # Giả lập một object có hàm predict (giống model thật)
    class MockModel:
        def predict(self, data):
            import numpy as np
            return np.array([1]) # Luôn trả về kết quả dự đoán là 1
            
    mock_load.return_value = MockModel()
    
    # Import app SAU KHI đã patch hàm load_model
    from app.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    payload = {
        "Age": 30, "Pclass": 3, "Fare": 15.5, "FamilySize": 2, "Sex_male": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == [1] # Kiểm tra khớp với giá trị mock
