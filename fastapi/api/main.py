from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd

from schemas import PredictRequest, PredictResponse
from fastapi.api.model_loader import load_model

app = FastAPI(title="Titanic Model Serving")

# Load model once at startup
model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert to DataFrame
    df = pd.DataFrame([request.dict()])
    # Predict
    pred = model.predict(df)
    return PredictResponse(prediction=int(pred[0]))
