"""
src/prediction/predictor.py
Sistema de Apoio à Tomada de Decisão (SATD).
Gera: probabilidades, nível de confiança, palavras relevantes, similaridade.
"""

import os
import json
import joblib
import numpy as np
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.text_processor import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")

# Thresholds para nível de confiança
CONF_HIGH   = 0.80
CONF_MEDIUM = 0.65


class FakeNewsPredictor:
    """
    Carrega o modelo persistido e o vetorizador e executa predição
    com saída de apoio à decisão (SATD).
    """

    def __init__(self, model_name: str = "svm", vectorizer_type: str = "tfidf"):
        self.model_name      = model_name
        self.vectorizer_type = vectorizer_type
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(MODELS_DIR, f"{self.model_name}.pkl")
        vec_path   = os.path.join(MODELS_DIR, f"{self.vectorizer_type}.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}\n"
                "Execute o treinamento antes de usar o preditor."
            )
        if not os.path.exists(vec_path):
            raise FileNotFoundError(
                f"Vetorizador não encontrado: {vec_path}\n"
                "Execute o treinamento antes de usar o preditor."
            )

        self.model      = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)
        print(f"  Modelo '{self.model_name}' e vetorizador '{self.vectorizer_type}' carregados.")

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """
        Recebe texto bruto, aplica pré-processamento e retorna
        dicionário com todos os indicadores do SATD.
        """
        processed = preprocess(text)
        vec       = self.vectorizer.transform([processed])

        # Probabilidades
        proba       = self.model.predict_proba(vec)[0]  # [P(fake), P(verdadeira)]
        pred_class  = int(np.argmax(proba))
        label_names = {0: "Fake", 1: "Verdadeira"}
        label       = label_names[pred_class]

        prob_fake  = round(float(proba[0]) * 100, 1)
        prob_true  = round(float(proba[1]) * 100, 1)
        confidence = float(max(proba))

        # Nível de confiança
        if confidence >= CONF_HIGH:
            conf_level = "Alto"
        elif confidence >= CONF_MEDIUM:
            conf_level = "Médio"
        else:
            conf_level = "Baixo"

        # Palavras mais relevantes (top-15)
        top_words = self._get_top_words(vec, pred_class, n=15)

        result = {
            "classificacao": label,
            "probabilidade": {
                "Fake":       f"{prob_fake}%",
                "Verdadeira": f"{prob_true}%",
            },
            "confianca": {
                "nivel": conf_level,
                "score": round(confidence, 4),
            },
            "palavras_relevantes": top_words,
            "texto_processado":    processed[:300] + ("..." if len(processed) > 300 else ""),
            "modelo_utilizado":    self.model_name,
            "vetorizador":         self.vectorizer_type,
        }
        return result

    def _get_top_words(self, vec_matrix, pred_class: int, n: int = 15) -> list:
        """
        Extrai os termos que mais contribuíram para a classe predita
        com base nos pesos do vetorizador × features do texto.
        """
        try:
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            # scores = produto elemento a elemento dos pesos do documento
            doc_array = vec_matrix.toarray()[0]
            top_idx   = np.argsort(doc_array)[::-1][:n]
            words     = feature_names[top_idx].tolist()
            scores    = doc_array[top_idx].tolist()
            return [{"termo": w, "peso": round(float(s), 4)}
                    for w, s in zip(words, scores) if s > 0]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Similaridade com base de treino (opcional)
    # ------------------------------------------------------------------
    def find_similar(self, text: str, corpus_texts: list,
                     corpus_labels: list, top_n: int = 3) -> list:
        """
        Encontra notícias similares na base de treinamento
        usando similaridade de cosseno.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        processed    = preprocess(text)
        query_vec    = self.vectorizer.transform([processed])
        corpus_vecs  = self.vectorizer.transform(corpus_texts)
        sims         = cosine_similarity(query_vec, corpus_vecs)[0]
        top_idx      = np.argsort(sims)[::-1][:top_n]

        label_names = {0: "Fake", 1: "Verdadeira"}
        return [
            {
                "rank":          int(i + 1),
                "similaridade":  round(float(sims[idx]), 4),
                "label":         label_names.get(corpus_labels[idx], "?"),
                "trecho":        corpus_texts[idx][:150] + "...",
            }
            for i, idx in enumerate(top_idx)
        ]


# ------------------------------------------------------------------
# Função utilitária para uso via CLI
# ------------------------------------------------------------------
def predict_from_file(txt_path: str, model_name: str = "svm",
                      vectorizer_type: str = "tfidf",
                      output_path: Optional[str] = None) -> dict:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    predictor = FakeNewsPredictor(model_name, vectorizer_type)
    result    = predictor.predict(text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = output_path or os.path.join(OUTPUT_DIR, "resultado.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nResultado salvo em: {out}")
    return result


if __name__ == "__main__":
    import sys
    txt  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "input", "noticia.txt")
    model = sys.argv[2] if len(sys.argv) > 2 else "svm"
    predict_from_file(txt, model_name=model)
