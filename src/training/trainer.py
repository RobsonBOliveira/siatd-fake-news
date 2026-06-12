"""
src/training/trainer.py
Treinamento dos três modelos: Naive Bayes, SVM e Random Forest.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.text_processor import preprocess
from src.feature_extraction.vectorizer import build_tfidf, save_vectorizer
from src.evaluation.metrics import evaluate_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Carrega o CSV do Fake.Br Corpus (preprocessed).
    Aceita variações de nome de coluna: 'label'/'Label', 'text'/'preprocessed_news'.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normaliza nome da coluna de texto
    for col in ["preprocessed_news", "text", "body", "news"]:
        if col in df.columns:
            df.rename(columns={col: "text"}, inplace=True)
            break

    # Normaliza nome da coluna de rótulo
    for col in ["label", "class", "target"]:
        if col in df.columns:
            df.rename(columns={col: "label"}, inplace=True)
            break

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    # Mapeia rótulos para 0/1
    label_map = {"fake": 0, "false": 0, "0": 0,
                 "true": 1, "real": 1, "1": 1}
    df["label"] = df["label"].str.lower().map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"  Dataset carregado: {len(df)} amostras")
    print(f"  Distribuição: {df['label'].value_counts().to_dict()}")
    return df


def apply_preprocessing(df: pd.DataFrame, already_preprocessed: bool = True) -> pd.DataFrame:
    """
    Se o CSV já veio pré-processado (pasta preprocessed do Fake.Br),
    mantém o texto. Caso contrário, aplica o pipeline completo.
    """
    if not already_preprocessed:
        print("  Aplicando pré-processamento...")
        df["text"] = df["text"].apply(preprocess)
    else:
        print("  Texto já pré-processado – etapa de limpeza ignorada.")
    return df


def get_models():
    """Retorna dicionário com os três modelos configurados."""
    # SVM não suporta predict_proba nativamente → envolvemos com CalibratedClassifierCV
    svm_base = LinearSVC(max_iter=2000, C=1.0)
    svm_calibrated = CalibratedClassifierCV(svm_base, cv=3)

    return {
        "naive_bayes":   MultinomialNB(alpha=0.1),
        "svm":           svm_calibrated,
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=42
        ),
    }


def train_all(csv_path: str, vectorizer_type: str = "tfidf",
              already_preprocessed: bool = True, test_size: float = 0.2):
    """
    Pipeline completo de treinamento.
    Persiste modelos + vetorizador em models/.
    Retorna dicionário de métricas por modelo.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n=== CARREGANDO DATASET ===")
    df = load_dataset(csv_path)
    df = apply_preprocessing(df, already_preprocessed)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")

    # Vetorização
    print("\n=== VETORIZAÇÃO ===")
    vectorizer = build_tfidf() if vectorizer_type == "tfidf" else build_bow()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    save_vectorizer(vectorizer, vectorizer_type)

    # Treinamento + Avaliação
    all_metrics = {}
    models = get_models()

    for name, model in models.items():
        print(f"\n=== TREINANDO: {name.upper()} ===")
        model.fit(X_train_vec, y_train)

        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"  Modelo salvo: {model_path}")

        metrics = evaluate_model(model, X_test_vec, y_test, name)
        all_metrics[name] = metrics

    # Salva metadados
    meta_path = os.path.join(MODELS_DIR, "meta.pkl")
    joblib.dump({"vectorizer_type": vectorizer_type, "classes": [0, 1],
                 "label_names": {0: "Fake", 1: "Verdadeira"}}, meta_path)

    return all_metrics


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "preprocessed", "fake_news.csv")
    train_all(csv)
