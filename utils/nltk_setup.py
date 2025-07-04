import os
import nltk


# Define your project-specific nltk_data directory
NLTK_DIR = os.getenv("NLTK_DATA", "./nltk_data")
NLTK_DIR = os.path.abspath(NLTK_DIR)

# Ensure nltk uses this path
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)

# Resource map: {nltk_name: relative_path}
REQUIRED_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "stopwords": "corpora/stopwords",
}

def ensure_nltk_resources():
    os.makedirs(NLTK_DIR, exist_ok=True)

    for resource, path in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(path)
            print(f"[nltk] '{resource}' already exists.")
        except LookupError:
            print(f"[nltk] Downloading '{resource}'...")
            nltk.download(resource, download_dir=NLTK_DIR)


if __name__ == "__main__":
    ensure_nltk_resources()