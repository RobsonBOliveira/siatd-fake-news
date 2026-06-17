"""
src/eda/exploratory_analysis.py
Módulo de Análise Exploratória de Dados (EDA) para o SIATD de Fake News.

Fornece estatísticas gerais, análise textual, frequência de palavras,
análise de vocabulário e avaliação de balanceamento do dataset.
"""

import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Import opcional com fallback (scipy)
# ---------------------------------------------------------------------------

try:
    from scipy.stats import pointbiserialr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Classe auxiliar para estatísticas descritivas
# ---------------------------------------------------------------------------

class TextStats:
    """
    Calcula estatísticas descritivas (média, mediana, desvio padrão,
    mínimo, máximo) a partir de uma Series numérica.
    """

    def __init__(self, series: pd.Series) -> None:
        """
        Parameters
        ----------
        series : pd.Series
            Série numérica contendo os valores a serem sumarizados.
        """
        self.series = series.dropna()

    def compute(self) -> Dict[str, float]:
        """
        Calcula as estatísticas descritivas da série.

        Returns
        -------
        dict
            Dicionário com as chaves: mean, median, std, min, max.
        """
        if len(self.series) == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0,
                    "min": 0.0, "max": 0.0}
        return {
            "mean":   round(float(self.series.mean()), 2),
            "median": round(float(self.series.median()), 2),
            "std":    round(float(self.series.std()), 2),
            "min":    int(self.series.min()),
            "max":    int(self.series.max()),
        }


# ---------------------------------------------------------------------------
# Classe principal de EDA
# ---------------------------------------------------------------------------

class ExploratoryAnalysis:
    """
    Orquestrador da Análise Exploratória de Dados.

    Carrega o CSV do Fake.Br Corpus e expõe métodos para cada dimensão
    da análise: visão geral, estatísticas de texto, frequência de palavras,
    vocabulário e balanceamento.

    Parameters
    ----------
    csv_path : str
        Caminho para o arquivo CSV do dataset.

    Raises
    ------
    FileNotFoundError
        Se o arquivo CSV não for encontrado.
    ValueError
        Se o CSV não contiver as colunas esperadas (texto + rótulo).
    """

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {csv_path}\n"
                "Verifique o caminho e tente novamente."
            )
        self.csv_path = csv_path
        self.df = self._load_data()

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """
        Carrega e normaliza o CSV do dataset.

        Aplica a mesma lógica de normalização de colunas de
        ``trainer.load_dataset()``, porém preserva os rótulos como
        strings (``fake``/``true``) para legibilidade na EDA.

        Returns
        -------
        pd.DataFrame
            DataFrame com as colunas ``text`` e ``label``.

        Raises
        ------
        ValueError
            Se as colunas obrigatórias não forem encontradas.
        """
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Remove coluna de índice se existir
        if "index" in df.columns:
            df.drop("index", axis=1, inplace=True)

        # Normaliza nome da coluna de texto
        for col in ["preprocessed_news", "text", "body", "news"]:
            if col in df.columns:
                df.rename(columns={col: "text"}, inplace=True)
                break
        else:
            raise ValueError(
                "Coluna de texto não encontrada no CSV. "
                "Esperada uma das colunas: 'preprocessed_news', 'text', 'body', 'news'."
            )

        # Normaliza nome da coluna de rótulo
        for col in ["label", "class", "target"]:
            if col in df.columns:
                df.rename(columns={col: "label"}, inplace=True)
                break
        else:
            raise ValueError(
                "Coluna de rótulo não encontrada no CSV. "
                "Esperada uma das colunas: 'label', 'class', 'target'."
            )

        # Limpeza
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(str).str.strip().str.lower()

        # Mapeia variações de rótulo para fake/true
        label_map = {
            "fake": "fake", "false": "fake", "0": "fake",
            "true": "true", "real": "true", "1": "true",
        }
        df["label"] = df["label"].map(label_map)
        df = df.dropna(subset=["label"])

        print(f"  Dataset carregado: {len(df)} amostras")
        print(f"  Distribuição: {df['label'].value_counts().to_dict()}")
        return df

    # ------------------------------------------------------------------
    # 1. Visão Geral do Dataset
    # ------------------------------------------------------------------

    def dataset_overview(self) -> Dict:
        """
        Gera estatísticas gerais sobre o dataset.

        Returns
        -------
        dict
            Chaves: total_samples, fake_count, true_count,
            fake_pct, true_pct, null_counts, duplicate_count.
        """
        label_counts = self.df["label"].value_counts()
        fake_count = int(label_counts.get("fake", 0))
        true_count = int(label_counts.get("true", 0))
        total = len(self.df)

        null_counts = self.df.isnull().sum().to_dict()
        duplicate_count = int(self.df.duplicated(subset=["text"]).sum())

        return {
            "total_samples": total,
            "fake_count":    fake_count,
            "true_count":    true_count,
            "fake_pct":      round(fake_count / total * 100, 2) if total else 0.0,
            "true_pct":      round(true_count / total * 100, 2) if total else 0.0,
            "null_counts":   {k: int(v) for k, v in null_counts.items()},
            "duplicate_count": duplicate_count,
        }

    # ------------------------------------------------------------------
    # 2. Estatísticas de Texto
    # ------------------------------------------------------------------

    def text_statistics(self) -> Dict:
        """
        Calcula estatísticas de comprimento dos textos (caracteres e
        palavras), globalmente e separadas por classe.

        .. note::
           A contagem de sentenças não é calculada para textos
           pré-processados porque toda a pontuação (incluindo
           pontos-finais) foi removida. O campo ``sentence_count``
           é retornado como ``None`` com uma observação.

        Returns
        -------
        dict
            Estrutura aninhada com ``overall``, ``fake``, ``true``
            e ``sentence_note``.
        """
        # Colunas derivadas
        char_counts = self.df["text"].str.len()
        word_counts = self.df["text"].str.split().str.len()

        stats = {
            "overall": {
                "char_count": TextStats(char_counts).compute(),
                "word_count": TextStats(word_counts).compute(),
            },
            "sentence_note": (
                "Contagem de sentenças não disponível: o texto está "
                "pré-processado e a pontuação foi removida. Para análise "
                "de sentenças, utilize o CSV original (data/raw/)."
            ),
        }

        for label in ["fake", "true"]:
            mask = self.df["label"] == label
            stats[label] = {
                "char_count": TextStats(char_counts[mask]).compute(),
                "word_count": TextStats(word_counts[mask]).compute(),
            }

        return stats

    # ------------------------------------------------------------------
    # 3. Frequência de Palavras
    # ------------------------------------------------------------------

    def word_frequency(self, top_n: int = 20) -> Dict:
        """
        Calcula as palavras mais frequentes no corpus, globalmente e
        por classe.

        Como o texto já está pré-processado (stopwords removidas,
        tokenizado), a contagem é feita diretamente por split.

        Parameters
        ----------
        top_n : int
            Número de palavras mais frequentes a retornar (padrão: 20).

        Returns
        -------
        dict
            Chaves: ``overall``, ``fake``, ``true``. Cada valor é uma
            lista de ``{"word": str, "count": int, "frequency": float}``.
        """
        def _top_words(text_series: pd.Series, n: int) -> List[Dict]:
            all_words: List[str] = []
            for text in text_series:
                all_words.extend(text.split())
            total = len(all_words)
            counter = Counter(all_words)
            return [
                {
                    "word": word,
                    "count": count,
                    "frequency": round(count / total * 100, 2) if total else 0.0,
                }
                for word, count in counter.most_common(n)
            ]

        return {
            "overall": _top_words(self.df["text"], top_n),
            "fake":    _top_words(self.df[self.df["label"] == "fake"]["text"], top_n),
            "true":    _top_words(self.df[self.df["label"] == "true"]["text"], top_n),
        }

    # ------------------------------------------------------------------
    # 4. Análise de Vocabulário
    # ------------------------------------------------------------------

    def vocabulary_analysis(self) -> Dict:
        """
        Analisa o tamanho e a diversidade do vocabulário.

        Returns
        -------
        dict
            Chaves: ``overall``, ``fake``, ``true`` com sub-chaves:
            total_words, unique_words, unique_ratio.
        """
        def _vocab(text_series: pd.Series) -> Dict:
            all_words: List[str] = []
            for text in text_series:
                all_words.extend(text.split())
            total = len(all_words)
            unique = len(set(all_words))
            return {
                "total_words":  total,
                "unique_words": unique,
                "unique_ratio": round(unique / total, 4) if total else 0.0,
            }

        return {
            "overall": _vocab(self.df["text"]),
            "fake":    _vocab(self.df[self.df["label"] == "fake"]["text"]),
            "true":    _vocab(self.df[self.df["label"] == "true"]["text"]),
        }

    # ------------------------------------------------------------------
    # 5. Análise de Balanceamento
    # ------------------------------------------------------------------

    def balance_analysis(self) -> Dict:
        """
        Avalia o balanceamento entre as classes ``fake`` e ``true``.

        Returns
        -------
        dict
            Chaves: imbalance_ratio, assessment, ml_impact, recommendation.
        """
        label_counts = self.df["label"].value_counts()
        fake_n = int(label_counts.get("fake", 0))
        true_n = int(label_counts.get("true", 0))

        min_class = min(fake_n, true_n)
        max_class = max(fake_n, true_n)
        ratio = round(min_class / max_class, 4) if max_class else 0.0

        if ratio >= 0.90:
            assessment = "Balanceado"
            ml_impact = (
                "Modelos de ML não serão penalizados pelo desbalanceamento. "
                "Métricas como accuracy são confiáveis."
            )
            recommendation = (
                "Nenhuma técnica de balanceamento é necessária. "
                "Pode-se utilizar os dados como estão."
            )
        elif ratio >= 0.70:
            assessment = "Levemente desbalanceado"
            ml_impact = (
                "Impacto leve: modelos podem ter viés sutil em direção "
                "à classe majoritária. Recomenda-se monitorar precision/recall."
            )
            recommendation = (
                "Considere usar class_weight='balanced' ou stratified k-fold."
            )
        elif ratio >= 0.50:
            assessment = "Desbalanceado"
            ml_impact = (
                "Impacto moderado: a classe minoritária pode ter recall "
                "significativamente menor."
            )
            recommendation = (
                "Utilize SMOTE, class_weight='balanced' ou undersampling "
                "da classe majoritária."
            )
        else:
            assessment = "Altamente desbalanceado"
            ml_impact = (
                "Impacto severo: a classe minoritária pode ser ignorada "
                "pelo modelo."
            )
            recommendation = (
                "Necessário oversampling (SMOTE) + class_weight='balanced' "
                "ou coleta de mais dados da classe minoritária."
            )

        return {
            "imbalance_ratio": ratio,
            "fake_count":      fake_n,
            "true_count":      true_n,
            "assessment":      assessment,
            "ml_impact":       ml_impact,
            "recommendation":  recommendation,
        }

    # ------------------------------------------------------------------
    # 6. Análise de Correlações
    # ------------------------------------------------------------------

    def correlation_analysis(self, top_n: int = 50, mi_top_n: int = 20) -> Dict:
        """
        Análise de correlação entre features textuais e o label.

        Calcula duas medidas:

        1. **Mutual Information (MI)** entre as *top-N* palavras mais
           frequentes e o label — mede o poder discriminativo de cada
           palavra. Retorna as *mi_top_n* palavras com maior MI.
        2. **Correlação ponto-bisserial** entre ``char_count`` /
           ``word_count`` e o label binário (``fake`` = 0, ``true`` = 1).

        .. note::
           A correlação ponto-bisserial depende do **scipy**. Se o
           pacote não estiver instalado, ``char_corr`` e ``word_corr``
           serão ``None``.

        Parameters
        ----------
        top_n : int
            Número de palavras mais frequentes a considerar para a MI
            (padrão: 50).
        mi_top_n : int
            Número de palavras com maior MI a retornar (padrão: 20).

        Returns
        -------
        dict
            Chaves:

            - ``mi_words`` : list of dict
              Lista de ``{"word": str, "mi_score": float}`` ordenada
              por MI decrescente (tamanho ``mi_top_n``).
            - ``char_corr`` : dict or None
              ``{"correlation": float, "p_value": float}`` para a
              correlação ponto-bisserial com a contagem de caracteres.
            - ``word_corr`` : dict or None
              ``{"correlation": float, "p_value": float}`` para a
              correlação ponto-bisserial com a contagem de palavras.
        """
        from sklearn.feature_selection import mutual_info_classif

        # --------------------------------------------------------------
        # 6.1 Mutual Information — top-N palavras × label
        # --------------------------------------------------------------

        # Obtém as top-N palavras mais frequentes (geral)
        freq_data = self.word_frequency(top_n=top_n)
        top_words = [item["word"] for item in freq_data["overall"]]

        # Monta matriz documento-termo (contagem) para essas palavras
        dt_matrix = np.zeros((len(self.df), len(top_words)), dtype=np.int32)
        for i, text in enumerate(self.df["text"]):
            words = text.split()
            for j, word in enumerate(top_words):
                dt_matrix[i, j] = words.count(word)

        # Binariza o label: fake=0, true=1
        y = (self.df["label"] == "true").astype(int).values

        # Calcula Mutual Information
        mi_scores = mutual_info_classif(dt_matrix, y, random_state=42)

        # Ordena por MI decrescente e seleciona as top mi_top_n
        mi_word_scores = sorted(
            [
                {"word": word, "mi_score": round(float(score), 6)}
                for word, score in zip(top_words, mi_scores)
            ],
            key=lambda x: x["mi_score"],
            reverse=True,
        )[:mi_top_n]

        # --------------------------------------------------------------
        # 6.2 Correlação ponto-bisserial — comprimento × label
        # --------------------------------------------------------------

        char_counts = self.df["text"].str.len().values.astype(float)
        word_counts = self.df["text"].str.split().str.len().values.astype(float)

        char_corr = None
        word_corr = None

        if HAS_SCIPY:
            char_r, char_p = pointbiserialr(char_counts, y)
            word_r, word_p = pointbiserialr(word_counts, y)
            char_corr = {
                "correlation": round(float(char_r), 6),
                "p_value":     round(float(char_p), 6),
            }
            word_corr = {
                "correlation": round(float(word_r), 6),
                "p_value":     round(float(word_p), 6),
            }
        else:
            print(
                "  [!] scipy não disponível — correlação ponto-bisserial "
                "ignorada."
            )
            print("       Instale com: pip install scipy")

        return {
            "mi_words":  mi_word_scores,
            "char_corr": char_corr,
            "word_corr": word_corr,
        }

    # ------------------------------------------------------------------
    # Execução completa
    # ------------------------------------------------------------------

    def run_all(self) -> Dict:
        """
        Executa todas as análises e retorna um dicionário agregado.

        Returns
        -------
        dict
            Chaves: dataset_overview, text_statistics, word_frequency,
            vocabulary_analysis, balance_analysis, correlation_analysis.
        """
        print("\n=== ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ===")

        print("\n[1/6] Visão geral do dataset...")
        overview = self.dataset_overview()

        print("[2/6] Estatísticas de texto...")
        text_stats = self.text_statistics()

        print("[3/6] Frequência de palavras...")
        freq = self.word_frequency()

        print("[4/6] Análise de vocabulário...")
        vocab = self.vocabulary_analysis()

        print("[5/6] Análise de balanceamento...")
        balance = self.balance_analysis()

        print("[6/6] Análise de correlações...")
        corr = self.correlation_analysis()

        print("[OK] EDA concluída.\n")
        return {
            "dataset_overview":     overview,
            "text_statistics":      text_stats,
            "word_frequency":       freq,
            "vocabulary_analysis":  vocab,
            "balance_analysis":     balance,
            "correlation_analysis": corr,
        }


# ---------------------------------------------------------------------------
# Execução standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    default_csv = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "data", "preprocessed", "pre-processed.csv"
    )
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else default_csv

    eda = ExploratoryAnalysis(csv_arg)
    results = eda.run_all()
    print(json.dumps(results, ensure_ascii=False, indent=2))
