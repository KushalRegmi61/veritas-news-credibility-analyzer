# FastAPI entry point
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from .schema.user_input import UserInput
from .schema.prediction_response import PredictionResponse
from utils.predict_output import predict_output, MODEL_VERSION, model_pipeline
from utils.nltk_config import configure_nltk_path
from contextlib import asynccontextmanager




# Setting up startup and shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure required NLTK resources path at startup
    configure_nltk_path()
    yield

#fast api app 
app = FastAPI(lifespan= lifespan)



# home root
@app.get("/")
def root():
    return {"message": "Misinformation Detector API is running."}

# machine readable
@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaded': model_pipeline is not None
    }


# prediction root
@app.post("/predict", response_model=PredictionResponse)
def predict_news(text: UserInput):
    user_input = {
        'text': text.text,
        # 'text_length': text.text_length,
        'text_length_bin': text.text_length_bin,
        'has_uppercase_emphasis': text.has_uppercase_emphasis,
        'long_text_flag': text.long_text_flag,
        'readability_score': text.readability_score,
        'punctuation_alert': text.punctuation_alert,
        'first_sentence_length': text.first_sentence_length, 
        'uppercase_words_count': text.uppercase_words_count
    }
    try: 
        prediction = predict_output(user_input)
        return JSONResponse(status_code=200, content={"response": prediction})
    
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))


