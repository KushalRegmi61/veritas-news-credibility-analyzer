import pandas as pd

# from utils.nltk_resources import ensure_nltk_resources
from api.schema.user_input import UserInput
from utils.model_utils import load_preprocessor_and_model
from dotenv import load_dotenv
import os

load_dotenv()

# Ensure required NLTK resources are available
# ensure_nltk_resources()
model_pipeline_path = os.getenv('MODEL_PATH')

# Load model components
model_pipeline, text_preprocessor = load_preprocessor_and_model(model_pipeline_path=model_pipeline_path)

# Mapping from numeric label to human-readable class name
CLASS_LABELS = {
    0: "Real News",
    1: "Fake News"
}


def predict_output(user_input):
    """
    Predicts the category of a news article and returns detailed prediction info.

    Args:
        user_input (dict): Dictionary containing input features (e.g., {"text": "some news text"})

    Returns:
        dict: {
            "predicted_category": str,
            "confidence": float,
            "class_probabilities": Dict[str, float]
        }
    """
    # Convert input to DataFrame and preprocess text
    df = pd.DataFrame([user_input])
    df["text"] = text_preprocessor.transform(df["text"])

    # Predict class and get probability distribution
    predicted_label = int(model_pipeline.predict(df)[0])
    class_proba = model_pipeline.predict_proba(df)[0]
    class_indices = model_pipeline.classes_

    # Map numeric class labels to readable names
    class_names = [CLASS_LABELS[int(cls)] for cls in class_indices]
    class_probabilities =  {name: round(float(prob), 6) for name, prob in zip(class_names, class_proba)}

    predicted_class_name = CLASS_LABELS[predicted_label]
    confidence = round(class_probabilities[predicted_class_name], 6)

    return {
        "predicted_category": predicted_class_name,
        "confidence": confidence,
        "class_probabilities": class_probabilities
    }


# Optional CLI test runner
if __name__ == "__main__":
    sample_input = UserInput(text="Corona is dangerous.")
    prediction = predict_output(sample_input.model_dump(mode="json"))
    print(prediction)
