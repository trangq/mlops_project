import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# 1. Tạo Mock Model trả về array [1]
mock_model = MagicMock()
mock_model.predict.return_value = np.array([1])

# 2. Patch load_model TRƯỚC khi import app
with patch("app.api.model_loader.load_model", return_value=mock_model):
    from app.api.main import app
    client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    payload = {
        "Age": 30.0,
        "Pclass": 3,
        "Fare": 15.5,
        "FamilySize": 2,
        "Sex_male": 1
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    result = response.json()
    
    # KIỂM TRA: FastAPI trả về list, nên ta so sánh với list [1]
    # Lỗi trước đó của bạn là so sánh 1 (số) == [1] (danh sách)
    assert result["prediction"] == [1]

def test_predict_invalid_data():
    # Kiểm tra xem API có bắt lỗi khi truyền chuỗi vào trường số không
    payload = {"Age": "invalid"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422