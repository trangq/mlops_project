from fastapi import FastAPI
import pandas as pd
from app.api.model_loader import load_model
from pydantic import BaseModel

app = FastAPI()

# Load model khi khởi chạy
model = load_model()

class TitanicRequest(BaseModel):
    Age: float
    Pclass: int
    Fare: float
    FamilySize: int
    Sex_male: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: TitanicRequest):
    # SỬA TẠI ĐÂY: Dùng model_dump() thay cho dict() để đúng chuẩn Pydantic V2
    data_dict = request.model_dump()
    df = pd.DataFrame([data_dict])
    
    prediction = model.predict(df)
    
    # prediction thường là numpy array, chuyển về list để FastAPI trả về JSON
    return {"prediction": prediction.tolist()}
