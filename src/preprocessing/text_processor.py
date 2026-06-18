"""
src/preprocessing/text_processor.py
Pipeline de pré-processamento textual para o SIATD de Fake News.
Reproduz o padrão do Fake.Br Corpus (preprocessed).
"""

import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixa recursos necessários do NLTK (silencioso se já existirem)
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

STOPWORDS_PT = set(stopwords.words("portuguese"))


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\b\d+\b", " ", text)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_accents(text: str) -> str:
    """Remove acentos e diacríticos, como no dataset pré-processado."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize_and_remove_stopwords(text: str) -> str:
    tokens = word_tokenize(text, language="portuguese")
    filtered = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]
    return " ".join(filtered)


def preprocess(text: str) -> str:
    """
    Pipeline completo que reproduz o pré-processamento do Fake.Br Corpus.
    Etapas:
      1. Minúsculas
      2. Remove URLs
      3. Remove números isolados
      4. Remove pontuação
      5. Remove acentos
      6. Remove espaços extras
      7. Remove stopwords + tokeniza
    """
    text = text.lower()
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_accents(text)
    text = remove_extra_spaces(text)
    text = tokenize_and_remove_stopwords(text)
    return text


if __name__ == "__main__":
    sample = "O presidente anunciou hoje, 12/06/2026, novas medidas. Acesse: https://gov.br para saber mais!"
    print("Original :", sample)
    print("Processado:", preprocess(sample))
