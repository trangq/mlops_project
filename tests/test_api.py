import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Bước 1: Tạo một Mock object trước
mock_model = MagicMock()
mock_model.predict.return_value = [1] # Khi gọi model.predict() sẽ trả về [1]

# Bước 2: Patch hàm load_model TRƯỚC khi import app
# Chúng ta dùng context manager hoặc patch thủ công ở cấp module
with patch("app.api.model_loader.load_model", return_value=mock_model):
    from app.api.main import app
    client = TestClient(app)

# Bước 3: Viết các hàm test như bình thường
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    payload = {
        "Age": 30, 
        "Pclass": 3, 
        "Fare": 15.5, 
        "FamilySize": 2, 
        "Sex_male": 1
    }
    response = client.post("/predict", json=payload)
    
    # Debug nếu lỗi
    if response.status_code != 200:
        print(response.json())
        
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == [1]