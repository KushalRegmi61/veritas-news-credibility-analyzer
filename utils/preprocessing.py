import os
import re
import string
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Load environment variables
load_dotenv()

# Get NLTK data directory from .env or fallback to ./nltk_data
NLTK_DIR = os.getenv("NLTK_DATA", "./nltk_data")
NLTK_DIR = os.path.abspath(NLTK_DIR)

# Add to NLTK data search path
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.freqwords = set([
            'trump', 'president', 'reuters', 'state', 'donald',
            'states', 'house', 'government', 'republican', 'united'
        ])
        self.stopwords_set = set(self.stop_words).union(self.freqwords)
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)

        # Remove newlines and special unicode punctuation
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[’“”…]', '', text)

        # Remove emojis
        emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F"
                                        u"\U0001F300-\U0001F5FF"
                                        u"\U0001F680-\U0001F6FF"
                                        u"\U0001F1E0-\U0001F1FF"
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)

        # Expand contractions
        contractions = {
            "isn't": "is not", "he's": "he is", "wasn't": "was not", "there's": "there is",
            "couldn't": "could not", "won't": "will not", "they're": "they are", "she's": "she is",
            "wouldn't": "would not", "haven't": "have not", "that's": "that is", "you've": "you have",
            "what's": "what is", "weren't": "were not", "we're": "we are", "hasn't": "has not",
            "you'd": "you would", "shouldn't": "should not", "let's": "let us", "they've": "they have",
            "you'll": "you will", "i'm": "i am", "we've": "we have", "it's": "it is", "don't": "do not",
            "that´s": "that is", "i´m": "i am", "it’s": "it is", "she´s": "she is", "i’m": "i am",
            "i’d": "i did", "there’s": "there is"
        }
        for contraction, expanded in contractions.items():
            text = re.sub(rf"\b{re.escape(contraction)}\b", expanded, text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stopwords_set]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]
