import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# 1. Giả lập (Mock) một Model có phương thức .predict() trả về list [1]
mock_model = MagicMock()
mock_model.predict.return_value = np.array([1]) 

# 2. Patch hàm load_model TRƯỚC khi import app để tránh gọi lên MLflow thật
with patch("app.api.model_loader.load_model", return_value=mock_model):
    from app.api.main import app
    client = TestClient(app)

def test_health():
    """Kiểm tra xem API có sống không"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    """Kiểm tra logic dự đoán của API"""
    # Dữ liệu mẫu gửi lên
    payload = {
        "Age": 30,
        "Pclass": 3,
        "Fare": 15.5,
        "FamilySize": 2,
        "Sex_male": 1
    }
    
    response = client.post("/predict", json=payload)
    
    # Kiểm tra status code
    assert response.status_code == 200
    
    # Kiểm tra dữ liệu trả về
    result = response.json()
    assert "prediction" in result
    
    # QUAN TRỌNG: So sánh list với list [1] == [1] để không bị lỗi assert 1 == [1]
    assert result["prediction"] == [1]

def test_predict_invalid_data():
    """Kiểm tra xem API có bắt lỗi dữ liệu sai không"""
    payload = {"Age": "không phải số"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Lỗi Validation của FastAPI