import pandas as pd
import numpy as np
import shap
from utils.model_utils import load_model, load_preprocessor
from dotenv import load_dotenv
import os
from api.schema.user_input import UserInput


load_dotenv()

# Model Version 
MODEL_VERSION = '1.0.0'

model_pipeline_path = os.getenv('MODEL_PATH')
model_pipeline = load_model(model_pipeline_path='./model/model_pipeline_98-48.pkl')
text_preprocessor = load_preprocessor()

CLASS_LABELS = {
    0: "Real News",
    1: "Fake News"
}

# SHAP explainer (trained on full pipeline)
explainer = shap.Explainer(model_pipeline.named_steps['classifier'])

def predict_output(user_input: dict, top_k: int = 20):
    """
    Predict news category and return SHAP-based local explanation.
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    input_df['text'] = text_preprocessor.fit_transform(input_df['text'])

    # === Step 1: Transform input ===
    transformer = model_pipeline.named_steps['trf1']
    X_transformed = transformer.transform(input_df)

    # === Step 2: Get SHAP values ===
    shap_values = explainer(X_transformed)
    instance_index = 0
    values = shap_values.values[instance_index]
    base_value = float(shap_values.base_values[instance_index])

    # Convert SHAP input to dense
    data_values = shap_values.data
    if hasattr(data_values, "toarray"):
        data_row = data_values.toarray()[instance_index]
    else:
        data_row = data_values[instance_index]

    # === Step 3: Get feature names ===
    vectorizer = transformer.named_transformers_['vectorizer']
    text_features = vectorizer.get_feature_names_out()

    # Remainder features (passthrough metadata)
    metadata_features = [col for col in input_df.columns if col not in  ['text', 'text_length']]

    feature_names = list(text_features) + metadata_features

    # === Step 4: Validate shape ===
    if len(data_row) != len(feature_names):
        raise ValueError(f"Feature length mismatch: {len(data_row)} values vs {len(feature_names)} feature names")

    # === Step 5: Top-k SHAP features ===
    top_indices = np.argsort(np.abs(values))[-top_k:][::-1]

    feature_values = []
    n_text_feats = len(text_features)

    for i in top_indices:
        if i < n_text_feats:
            # Text feature value from transformed vector
            val = float(data_row[i])
        else:
            # Metadata feature value from original user_input dict
            meta_feature = feature_names[i]
            # Defensive: get from input_df or user_input dict, convert to float if possible
            raw_val = user_input.get(meta_feature, None)
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                val = raw_val  # fallback to raw if can't convert to float
        feature_values.append(val)


    explanation = {
        "shap_values": [float(values[i]) for i in top_indices],
        "feature_values": feature_values,
        "feature_names": [feature_names[i] for i in top_indices],
        "base_value": base_value
    }

    # === Step 6: Prediction ===
    proba = model_pipeline.predict_proba(input_df)[0]
    threshold = 0.40
    predicted_label = int(proba[1] >= threshold)

    class_names = [CLASS_LABELS[int(c)] for c in model_pipeline.classes_]
    class_probs = {name: round(float(prob), 6) for name, prob in zip(class_names, proba)}
    predicted_class_name = CLASS_LABELS[predicted_label]
    confidence = class_probs[predicted_class_name]

    return {
        "predicted_category": predicted_class_name,
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
            # 'text': 'kushal is my name.'
    
    }
        )
    
    
    prediction = predict_output(sample_input.model_dump(mode="json", exclude=['exclamations_mark_count', 'questions_mark_count']))
    print(prediction)

