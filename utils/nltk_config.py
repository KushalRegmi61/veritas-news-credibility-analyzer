# utils/nltk_config.py
import os
import nltk
from dotenv import load_dotenv
load_dotenv()

def configure_nltk_path():
    nltk_dir = os.getenv("NLTK_DATA", "./nltk_data")
    nltk_dir = os.path.abspath(nltk_dir)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)
