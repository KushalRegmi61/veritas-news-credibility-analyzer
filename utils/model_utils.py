# Utility functions for model handling
import joblib
from utils.preprocessing import TextPreprocessor
    
def load_preprocessor_and_model(model_pipeline_path):
    """
    Loads a machine learning model pipeline and initializes a text preprocessor.
    Args:
        model_pipeline_path (str): The file path to the saved model pipeline (e.g., a .joblib file).
    Returns:
        tuple: A tuple containing:
            - model_pipeline: The loaded machine learning model pipeline.
            - text_preprocessor: An instance of the TextPreprocessor class for preprocessing text data.
    Raises:
        RuntimeError: If there is an error loading the model pipeline or initializing the preprocessor.
    Example:
        model, preprocessor = load_preprocessor_and_model("models/my_pipeline.joblib")
    """
    try:
        model_pipeline = joblib.load(model_pipeline_path)
        text_preprocessor = TextPreprocessor()
    except Exception as e:
        raise RuntimeError(f"Error loading model components: {e}")
    
    return model_pipeline, text_preprocessor


    