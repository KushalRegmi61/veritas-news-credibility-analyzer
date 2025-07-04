# FastAPI entry point
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from .schema.user_input import UserInput
from .schema.prediction_response import PredictionResponse
from utils.predict_output import predict_output
from utils.nltk_setup import ensure_nltk_resources
from contextlib import asynccontextmanager



# Setting up startup and shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure required NLTK resources are downloaded at startup
    ensure_nltk_resources()
    yield

#fast api app 
app = FastAPI(lifespan= lifespan)



# home root
@app.get("/")
def root():
    return {"message": "Misinformation Detector API is running."}

# # machine readable
# @app.get('/health')
# def health_check():
#     return {
#         'status': 'OK',
#         'version': MODEL_VERSION,
#         'model_loaded': model is not None
#     }


# prediction root
@app.post("/predict", response_model=PredictionResponse)
def predict_news(text: UserInput):
    user_input = {
        'text': text.text,
        'text_length': text.text_length,
        'exclamations_mark_count': text.exclamations_mark_count,
        'questions_mark_count': text.questions_mark_count,
        'uppercase_words_count': text.uppercase_words_count
    }
    try: 
        prediction = predict_output(user_input)
        return JSONResponse(status_code=200, content={"response": prediction})
    
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
