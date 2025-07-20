import re
import string
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Base stopwords
        self.stop_words = stopwords.words('english')

        # Domain-specific terms (fake news frequent terms)
        self.freq_words = [
            'said', 'trump', 'president', 'people', 'state', 'reuters',
            'new', 'donald', 'house', 'government'
        ]

        # Common aliases or obfuscations
        self.aliases = {
            'gov': 'government',
            'gvt': 'government',
            'govt': 'government',
            'pres': 'president',
            'potus': 'president',
            'reut3rs': 'reuters',
            'trump2024': 'trump',
            'drumpf': 'trump'
        }

        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

        # Lemmatize all stopwords and freq words and aliases
        lemmatized_stopwords = set(self.lemmatizer.lemmatize(w) for w in self.stop_words)
        lemmatized_freqwords = set(self.lemmatizer.lemmatize(w) for w in self.freq_words)
        lemmatized_aliases = set(self.lemmatizer.lemmatize(w) for w in self.aliases.values())

        self.stopwords_set = lemmatized_stopwords.union(lemmatized_freqwords).union(lemmatized_aliases)

    def remove_obfuscated_keywords(self, text):
        """
        Removes obfuscated or stylized versions of known fake-news terms.
        """
        patterns = [
            r'\btr[._-]*u[._-]*m[._-]*p\b',
            r'\bdonald[._-]*trump\b',
            r'\breut[3e]*rs\b',
            r'\bpotus\b',
            r'\bgov[t]*\b',
            r'#(trump|reuters|president|government|house|people|state|said|new)',
            r'@(trump|reuters|president|government|house|people|state|said|new)',
            r'\btrump\d{2,4}\b',
        ]
        for pattern in patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        return text

    def preprocess_text(self, text):
        # Normalize text
        text = unidecode(text)  # Remove accents
        text = text.lower()

        # Remove URLs, HTML, emojis, numbers
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'\b\d+\b', '', text)

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

        # Remove punctuation, newlines
        text = re.sub(r"'s\b", "", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r'\s+', ' ', text)

        # Remove stylized keywords
        text = self.remove_obfuscated_keywords(text)

        # Tokenize
        tokens = self.tokenizer.tokenize(text)

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens]

        # Replace known aliases with their canonical form
        tokens = [self.aliases.get(tok, tok) for tok in tokens]

        # Filter out all stopwords and fake-indicative tokens
        tokens = [tok for tok in tokens if tok not in self.stopwords_set]

        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]



