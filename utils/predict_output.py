import pandas as pd

# from utils.nltk_setup import ensure_nltk_resources
from api.schema.user_input import UserInput
from utils.model_utils import load_preprocessor_and_model
from dotenv import load_dotenv
import os

load_dotenv()

# Ensure required NLTK resources are available
# ensure_nltk_resources()

model_pipeline_path = os.getenv('MODEL_PATH')


# MLFlow
MODEL_VERSION = '1.0.0'

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
    # df["text"] = text_preprocessor.transform(df["text"])

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
    sample_input = UserInput(
            **{
        "text": "On July 4, 2025, the United States experienced a complex Independence Day marked by both traditional celebrations and widespread political unrest. Millions participated in the 'Free America Weekend' protests organized by groups like 50501, Indivisible, and the Women’s March, opposing President Trump’s recently signed \"One Big Beautiful Bill,\" which extends 2017 tax cuts, cuts $1.2 trillion from Medicaid and food assistance, and increases immigration enforcement. Critics argue the bill disproportionately benefits the wealthy while risking vital benefits for 12 million Americans and adding $3.3 trillion to the national deficit over ten years. These demonstrations, called \"No Kings 2.0,\" echoed June's massive \"No Kings Day\" protests, spreading across all 50 states and internationally to cities such as Berlin and Tokyo, highlighting deep national divisions. Despite the tensions, traditional Fourth of July festivities, including the 45th annual \"A Capitol Fourth\" concert and a grand fireworks display, continued, uniting many Americans in celebration. However, concerns about a possible surge in coronavirus cases and California’s largest wildfire this year, which has grown to over 52,000 acres threatening communities and wildlife, tempered the mood. The day underscored a nation at a crossroads—balancing patriotic traditions with active political engagement and urgent social challenges—reflecting the complex and sometimes contentious state of American democracy in 2025."
    }
        )
    prediction = predict_output(sample_input.model_dump(mode="json"))
    print(prediction)
