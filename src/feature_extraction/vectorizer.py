"""
src/feature_extraction/vectorizer.py
Abordagens de vetorização: TF-IDF e Bag of Words.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
import os


VECTORIZER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def build_tfidf(max_features: int = 30_000, ngram_range=(1, 2)):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )


def build_bow(max_features: int = 30_000):
    return CountVectorizer(max_features=max_features, min_df=2)


def save_vectorizer(vectorizer, name: str):
    os.makedirs(VECTORIZER_DIR, exist_ok=True)
    path = os.path.join(VECTORIZER_DIR, f"{name}.pkl")
    joblib.dump(vectorizer, path)
    print(f"  Vetorizador salvo: {path}")
    return path


def load_vectorizer(name: str):
    path = os.path.join(VECTORIZER_DIR, f"{name}.pkl")
    return joblib.load(path)
