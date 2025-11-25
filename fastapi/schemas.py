from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    Age: float = Field(..., example=22)
    Pclass: int = Field(..., example=3)
    Fare: float = Field(..., example=7.25)
    FamilySize: int = Field(..., example=1)
    Sex_male: int = Field(..., example=1)

class PredictResponse(BaseModel):
    prediction: int
