# FastAPI entry point

from fastapi import FastAPI
from .dependencies import get_model, get_vectorizer, get_encoder
from .schema.user_input import UserInput
from .schema.prediction_response import PredictionResponse

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Misinformation Detector API is running."}

# Add prediction endpoint here
