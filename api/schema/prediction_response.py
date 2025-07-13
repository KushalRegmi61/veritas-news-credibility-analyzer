from pydantic import BaseModel, Field
from typing import Dict, List

class ExplanationModel(BaseModel):
    shap_values: List[float] = Field(
        ...,
        description="SHAP values indicating feature importance for the prediction",
        json_schema_extra={"examples": [[-3.82, 1.68, -1.35, 0.92, 0.89]]}
    )
    
    feature_values: List[float] = Field(
        ...,
        description="Actual values of the features (word counts for text features, raw counts for metadata)",
        json_schema_extra={"examples": [[1.0, 0.0, 1.0, 3.0, 1.0]]}
    )
    
    feature_names: List[str] = Field(
        ...,
        description="Names of the features corresponding to SHAP values",
        json_schema_extra={"examples": [["trump", "said", "berlin", "american", "america"]]}
    )
    
    base_value: float = Field(
        ...,
        description="Base value for SHAP explanation (model's average prediction)",
        json_schema_extra={"examples": [1.938552737236023]}
    )

class PredictionResponse(BaseModel):
    predicted_category: str = Field(
        ...,
        description="The predicted news category",
        json_schema_extra={"examples": ["Real News", "Fake News"]}
    )
    
    confidence: float = Field(
        ...,
        description="Model's confidence score for the predicted class (range: 0 to 1)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"examples": [0.871346]}
    )
    
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all possible classes",
        json_schema_extra={"examples": [{"Real News": 0.871346, "Fake News": 0.128654}]}
    )
    
    explanation: ExplanationModel = Field(
        ...,
        description="SHAP-based explanation for the prediction"
    )
    
    prediction_threshold : float = Field(
        ...,
        description='Model prediction treshold (range: 0 to 1)',
        ge=0.0,
        le=1.0,
        
        json_schema_extra={"examples": [0.55]}
    ) 

# Example usage and validation
if __name__ == "__main__":
    # Sample data based on your output
    sample_response = {
        "predicted_category": "Real News",
        "confidence": 0.871346,
        "class_probabilities": {
            "Real News": 0.871346,
            "Fake News": 0.128654
        },
        "explanation": {
            "shap_values": [-3.8233001232147217, 1.6845903396606445, -1.3543044328689575, 0.9201043844223022, 0.8887825012207031],
            "feature_values": [1.0, 0.0, 1.0, 3.0, 1.0],
            "feature_names": ["trump", "said", "berlin", "american", "america"],
            "base_value": 1.938552737236023
        }
    }
    
    # Validate the response
    response = PredictionResponse(**sample_response)
    print("Validation successful!")
    print(response.model_dump_json(indent=2))