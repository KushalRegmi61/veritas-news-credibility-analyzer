# Utility functions for model handling
import joblib
from utils.preprocessing import TextPreprocessor

def load_model(model_pipeline_path):
    """
    Loads a machine learning model pipeline.
    Args:
        model_pipeline_path (str): The file path to the saved model pipeline (e.g., a .joblib file).
    Returns:
        model_pipeline: The loaded machine learning model pipeline.
    Raises:
        RuntimeError: If there is an error loading the model pipeline.
    """
    try:
        model_pipeline = joblib.load(model_pipeline_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model pipeline: {e}")
    return model_pipeline

def load_preprocessor():
    """
    Initializes a text preprocessor.
    Returns:
        text_preprocessor: An instance of the TextPreprocessor class.
    Raises:
        RuntimeError: If there is an error initializing the preprocessor.
    """
    try:
        text_preprocessor = TextPreprocessor()
    except Exception as e:
        raise RuntimeError(f"Error initializing text preprocessor: {e}")
    return text_preprocessor
