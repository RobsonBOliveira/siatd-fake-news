"""
src/eda/visualization.py
Geração de gráficos para a Análise Exploratória de Dados (EDA).

Produz gráficos de distribuição de classes, histogramas, boxplots,
frequência de palavras e nuvens de palavras (word clouds).
Todos os gráficos são salvos em ``output/eda/``.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Imports opcionais com fallback
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

EDA_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "output", "eda"
)

# Paleta de cores consistente
COLOR_FAKE  = "#E74C3C"   # vermelho
COLOR_TRUE  = "#2ECC71"   # verde
COLOR_PALETTE = [COLOR_FAKE, COLOR_TRUE]

# Configuração global do seaborn
if HAS_PLOT:
    sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Cria o diretório de saída se não existir."""
    os.makedirs(path, exist_ok=True)


def _save_and_close(fig: "plt.Figure", filepath: str) -> str:
    """Salva a figura em PNG e fecha-a para liberar memória."""
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico salvo: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# 1. Distribuição de Classes
# ---------------------------------------------------------------------------

def plot_class_distribution(
    labels: pd.Series,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Gera gráfico de barras e gráfico de pizza com a distribuição das classes.

    Parameters
    ----------
    labels : pd.Series
        Série com os rótulos (``fake`` / ``true``).
    output_dir : str, optional
        Diretório de saída. Padrão: ``output/eda/``.

    Returns
    -------
    dict
        Caminhos dos arquivos gerados: ``bar``, ``pie``.
    """
    out = output_dir or EDA_OUTPUT_DIR
    _ensure_dir(out)

    if not HAS_PLOT:
        print("  [!] matplotlib/seaborn não disponíveis — gráficos ignorados.")
        return {}

    counts = labels.value_counts()
    paths: Dict[str, str] = {}

    # --- Barras ---
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[COLOR_FAKE, COLOR_TRUE], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Distribuição de Classes – Notícias", fontsize=14, fontweight="bold")
    ax.set_ylabel("Quantidade")
    ax.set_ylim(0, counts.max() * 1.1)
    paths["bar"] = _save_and_close(fig, os.path.join(out, "class_distribution_bar.png"))

    # --- Pizza ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=[COLOR_FAKE, COLOR_TRUE], startangle=90,
           explode=(0.02, 0.02), textprops={"fontsize": 13})
    ax.set_title("Distribuição de Classes – Notícias", fontsize=14, fontweight="bold")
    paths["pie"] = _save_and_close(fig, os.path.join(out, "class_distribution_pie.png"))

    return paths


# ---------------------------------------------------------------------------
# 2. Histogramas de Comprimento de Texto
# ---------------------------------------------------------------------------

def plot_text_length_histograms(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Gera histogramas sobrepostos (fake vs true) para contagem de
    caracteres e de palavras.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas ``text`` e ``label``.
    output_dir : str, optional
        Diretório de saída. Padrão: ``output/eda/``.

    Returns
    -------
    dict
        Caminhos: ``char_count``, ``word_count``.
    """
    out = output_dir or EDA_OUTPUT_DIR
    _ensure_dir(out)

    if not HAS_PLOT:
        print("  [!] matplotlib/seaborn não disponíveis — histogramas ignorados.")
        return {}

    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    paths: Dict[str, str] = {}

    for metric, xlabel, fname in [
        ("char_count", "Número de Caracteres", "char_count_histogram.png"),
        ("word_count", "Número de Palavras",    "word_count_histogram.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, color, alpha in [("fake", COLOR_FAKE, 0.5),
                                     ("true", COLOR_TRUE, 0.4)]:
            subset = df[df["label"] == label][metric]
            sns.histplot(subset, kde=True, color=color, alpha=alpha,
                         label=label.capitalize(), ax=ax, bins=40)
        ax.set_title(f"Distribuição de {xlabel} por Classe", fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequência")
        ax.legend()
        paths[metric + "_hist"] = _save_and_close(fig, os.path.join(out, fname))

    return paths


# ---------------------------------------------------------------------------
# 3. Boxplots de Comprimento de Texto
# ---------------------------------------------------------------------------

def plot_text_length_boxplots(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Gera boxplots lado a lado (fake vs true) para contagem de
    caracteres e de palavras.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas ``text`` e ``label``.
    output_dir : str, optional
        Diretório de saída. Padrão: ``output/eda/``.

    Returns
    -------
    dict
        Caminhos: ``char_count``, ``word_count``.
    """
    out = output_dir or EDA_OUTPUT_DIR
    _ensure_dir(out)

    if not HAS_PLOT:
        print("  [!] matplotlib/seaborn não disponíveis — boxplots ignorados.")
        return {}

    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["label_display"] = df["label"].str.capitalize()
    paths: Dict[str, str] = {}

    for metric, ylabel, fname in [
        ("char_count", "Número de Caracteres", "char_count_boxplot.png"),
        ("word_count", "Número de Palavras",   "word_count_boxplot.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(x="label_display", y=metric, data=df,
                    hue="label_display",
                    palette={"Fake": COLOR_FAKE, "True": COLOR_TRUE},
                    width=0.4, ax=ax, legend=False)
        ax.set_title(f"Boxplot de {ylabel} por Classe", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        paths[metric + "_box"] = _save_and_close(fig, os.path.join(out, fname))

    return paths


# ---------------------------------------------------------------------------
# 4. Top Palavras
# ---------------------------------------------------------------------------

def plot_top_words(
    freq_data: Dict[str, List[Dict]],
    top_n: int = 20,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Gera gráficos de barras horizontais com as palavras mais frequentes.

    Parameters
    ----------
    freq_data : dict
        Resultado de ``ExploratoryAnalysis.word_frequency()``.
        Espera as chaves ``overall``, ``fake``, ``true``.
    top_n : int
        Número de palavras a exibir (padrão: 20).
    output_dir : str, optional
        Diretório de saída. Padrão: ``output/eda/``.

    Returns
    -------
    dict
        Caminhos: ``overall``, ``fake``, ``true``.
    """
    out = output_dir or EDA_OUTPUT_DIR
    _ensure_dir(out)

    if not HAS_PLOT:
        print("  [!] matplotlib/seaborn não disponíveis — top words ignorados.")
        return {}

    colors = {"overall": "#3498DB", "fake": COLOR_FAKE, "true": COLOR_TRUE}
    titles = {
        "overall": "Top Palavras – Geral",
        "fake":    "Top Palavras – Fake",
        "true":    "Top Palavras – Verdadeira",
    }
    paths: Dict[str, str] = {}

    for key in ["overall", "fake", "true"]:
        items = freq_data.get(key, [])[:top_n]
        if not items:
            continue

        words = [it["word"] for it in items][::-1]
        counts = [it["count"] for it in items][::-1]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(words, counts, color=colors[key], edgecolor="white", height=0.7)
        ax.set_title(titles[key], fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequência")
        for i, val in enumerate(counts):
            ax.text(val + max(counts) * 0.005, i, str(val), va="center", fontsize=9)

        fname = f"top{top_n}_words_{key}.png"
        paths["top_" + key] = _save_and_close(fig, os.path.join(out, fname))

    return paths


# ---------------------------------------------------------------------------
# 5. Nuvens de Palavras (Word Cloud)
# ---------------------------------------------------------------------------

def plot_wordclouds(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Gera word clouds para o corpus completo e por classe (fake/true).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas ``text`` e ``label``.
    output_dir : str, optional
        Diretório de saída. Padrão: ``output/eda/``.

    Returns
    -------
    dict
        Caminhos: ``overall``, ``fake``, ``true``.
    """
    out = output_dir or EDA_OUTPUT_DIR
    _ensure_dir(out)

    if not HAS_WORDCLOUD:
        print("  [!] wordcloud não instalado — nuvens de palavras ignoradas.")
        print("       Instale com: pip install wordcloud")
        return {}

    wc = WordCloud(
        width=900, height=450,
        background_color="white",
        colormap="viridis",
        max_words=200,
        collocations=False,
        random_state=42,
    )

    paths: Dict[str, str] = {}
    labels = {"overall": None, "fake": "fake", "true": "true"}
    titles = {
        "overall": "Nuvem de Palavras – Geral",
        "fake":    "Nuvem de Palavras – Notícias Fake",
        "true":    "Nuvem de Palavras – Notícias Verdadeiras",
    }

    for key, label_filter in labels.items():
        if label_filter:
            text = " ".join(df[df["label"] == label_filter]["text"].tolist())
        else:
            text = " ".join(df["text"].tolist())

        if not text.strip():
            print(f"  [!] Texto vazio para '{key}' — word cloud ignorada.")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        cloud_img = wc.generate(text)
        ax.imshow(cloud_img, interpolation="bilinear")
        ax.set_title(titles[key], fontsize=16, fontweight="bold")
        ax.axis("off")

        fname = f"wordcloud_{key}.png"
        paths["wc_" + key] = _save_and_close(fig, os.path.join(out, fname))

    return paths


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------

def generate_all_charts(
    df: pd.DataFrame,
    freq_data: Dict[str, List[Dict]],
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Gera todos os gráficos da EDA e salva-os no diretório de saída.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas ``text`` e ``label``.
    freq_data : dict
        Resultado de ``ExploratoryAnalysis.word_frequency()``.
    output_dir : str, optional
        Diretório de saída. Padrão: ``output/eda/``.

    Returns
    -------
    dict
        Dicionário aninhado com os caminhos de cada categoria de gráfico.
        Chaves: class_distribution, text_histograms, text_boxplots,
        top_words, wordclouds.
    """
    out = output_dir or EDA_OUTPUT_DIR
    _ensure_dir(out)

    print("\n=== GERANDO GRÁFICOS EDA ===")

    print("\n[1/5] Distribuição de classes...")
    class_dist = plot_class_distribution(df["label"], out)

    print("[2/5] Histogramas de comprimento...")
    histograms = plot_text_length_histograms(df, out)

    print("[3/5] Boxplots de comprimento...")
    boxplots = plot_text_length_boxplots(df, out)

    print("[4/5] Top palavras...")
    top = plot_top_words(freq_data, output_dir=out)

    print("[5/5] Nuvens de palavras...")
    clouds = plot_wordclouds(df, out)

    print("[OK] Gráficos concluídos.\n")
    return {
        "class_distribution": class_dist,
        "text_histograms":    histograms,
        "text_boxplots":      boxplots,
        "top_words":          top,
        "wordclouds":         clouds,
    }


# ---------------------------------------------------------------------------
# Execução standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.eda.exploratory_analysis import ExploratoryAnalysis

    default_csv = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "data", "preprocessed", "pre-processed.csv"
    )
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else default_csv

    eda = ExploratoryAnalysis(csv_arg)
    results = eda.run_all()
    paths = generate_all_charts(eda.df, results["word_frequency"])
    print("Arquivos gerados:", json.dumps(paths, ensure_ascii=False, indent=2))
