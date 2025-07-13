import pandas as pd
import numpy as np
import shap
from utils.model_utils import load_model, load_preprocessor
from dotenv import load_dotenv
import os
from api.schema.user_input import UserInput
# from utils.nltk_config import configure_nltk_path



load_dotenv()


# Model Version 
MODEL_VERSION='1.0.0'


model_pipeline_path = os.getenv('MODEL_PATH')
model_pipeline = load_model(model_pipeline_path=model_pipeline_path)
text_preprocessor = load_preprocessor()

CLASS_LABELS = {
    0: "Real News",
    1: "Fake News"
}

# SHAP explainer should be reused across requests
explainer = shap.Explainer(model_pipeline.named_steps['classifier'])

def predict_output(user_input: dict, top_k: int = 20):
    """
    Predict news category and return SHAP-based local explanation.
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    input_df['text'] = text_preprocessor.fit_transform(input_df['text'])

    # 1. Transform input
    X_transformed = model_pipeline.named_steps['trf1'].transform(input_df)

    # 2. Generate SHAP values
    shap_values = explainer(X_transformed)
    instance_index = 0  # only one input
    values = shap_values.values[instance_index]
    base_value = float(shap_values.base_values[instance_index])

    # SHAP data matrix (sparse or dense)
    data_values = shap_values.data
    if hasattr(data_values, "toarray"):  # convert sparse to dense
        data_row = data_values.toarray()[instance_index]
    else:
        data_row = data_values[instance_index]

    # 3. Extract feature names
    vectorizer = model_pipeline.named_steps['trf1'].named_transformers_['vectorizer']
    text_features = vectorizer.get_feature_names_out()
    metadata_features = ['text_length', 'exclamations_mark_count', 'questions_mark_count', 'uppercase_words_count']
    feature_names = list(text_features) + metadata_features

    # Ensure feature name and data row match in length
    n_features = len(feature_names)
    if len(data_row) != n_features:
        raise ValueError(f"Feature length mismatch: {len(data_row)} values vs {n_features} feature names")

    # 4. Select top-k important features
    top_indices = np.argsort(np.abs(values))[-top_k:][::-1]

    explanation = {
        "shap_values": [float(values[i]) for i in top_indices],
        "feature_values": [float(data_row[i]) for i in top_indices],
        "feature_names": [feature_names[i] for i in top_indices],
        "base_value": base_value
    }

    # 5. Model prediction
    proba = model_pipeline.predict_proba(input_df)[0]
    threshold = 0.55
    if proba[1] >= threshold:
        predicted_label = 1
    else:
        predicted_label = 0
    proba = model_pipeline.predict_proba(input_df)[0]
    class_names = [CLASS_LABELS[int(c)] for c in model_pipeline.classes_]
    class_probs = {name: round(float(prob), 6) for name, prob in zip(class_names, proba)}
    predicted_class_name = CLASS_LABELS[predicted_label]
    confidence = class_probs[predicted_class_name]

    return {
        "predicted_category": predicted_class_name,  # "Fake News" or "Real News"
        "confidence": confidence,
        "class_probabilities": class_probs,
        "prediction_threshold": threshold,
        "explanation": explanation
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
