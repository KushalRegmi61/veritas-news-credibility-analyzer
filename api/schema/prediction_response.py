# Pydantic schema for prediction response
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    probability: float
