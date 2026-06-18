"""
src/prediction/visualization.py
Gera grafico com os resultados da predicao do SIATD.

O grafico exibe:
  - Classificacao (Fake / Verdadeira) com nivel de confianca
  - Barras de probabilidade para cada classe
  - Top-10 palavras mais relevantes do texto
  - Modelo (preditor) e vetorizador utilizados

O nome do arquivo segue o padrao:
  prediction_<vetorizador>_<modelo>.png
"""

import os
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")

# Paleta de cores consistente
COLOR_FAKE = "#E74C3C"       # vermelho
COLOR_TRUE = "#2ECC71"       # verde
COLOR_CONF_HIGH = "#27AE60"
COLOR_CONF_MEDIUM = "#F39C12"
COLOR_CONF_LOW = "#E74C3C"
BG_COLOR = "#FAFAFA"


def plot_prediction_result(result: dict,
                           output_dir: Optional[str] = None) -> str:
    """
    Gera e salva o grafico de resultado da predicao.

    Args:
        result: dicionario retornado por FakeNewsPredictor.predict().
        output_dir: diretorio de saida (padrao: output/).

    Returns:
        Caminho do arquivo PNG gerado.
    """
    if not HAS_PLOT:
        print("  [AVISO] matplotlib/seaborn nao disponiveis – grafico ignorado.")
        return ""

    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    modelo = result.get("modelo_utilizado", "desconhecido")
    vetorizador = result.get("vetorizador", "desconhecido")
    filename = f"prediction_{vetorizador}_{modelo}.png"
    filepath = os.path.join(out_dir, filename)

    # ------------------------------------------------------------------
    # Extrai dados do resultado
    # ------------------------------------------------------------------
    classificacao = result["classificacao"]
    prob_fake = float(result["probabilidade"]["Fake"].replace("%", ""))
    prob_true = float(result["probabilidade"]["Verdadeira"].replace("%", ""))
    conf_nivel = result["confianca"]["nivel"]
    conf_score = result["confianca"]["score"]
    palavras = result.get("palavras_relevantes", [])

    # Cor do destaque conforme a classe predita
    destaque_cor = COLOR_FAKE if classificacao == "Fake" else COLOR_TRUE

    # Cor do nivel de confianca
    conf_color = {
        "Alto": COLOR_CONF_HIGH,
        "Medio": COLOR_CONF_MEDIUM,
        "Baixo": COLOR_CONF_LOW,
    }.get(conf_nivel, "#333333")

    # ------------------------------------------------------------------
    # Cria a figura
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9), facecolor=BG_COLOR)

    # -- Titulo principal --
    fig.suptitle(
        f"SIATD – Resultado da Predicao",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # Subtitulo com modelo + vetorizador
    fig.text(
        0.5, 0.93,
        f"Modelo: {modelo.upper()}    |    Vetorizador: {vetorizador.upper()}",
        ha="center", fontsize=11, color="#555555",
        fontfamily="monospace",
    )

    # ------------------------------------------------------------------
    # Painel A: Classificacao + Confianca (topo esquerdo)
    # ------------------------------------------------------------------
    ax_class = fig.add_axes([0.08, 0.60, 0.38, 0.28])
    ax_class.set_facecolor(BG_COLOR)
    ax_class.axis("off")

    # Circulo com a classificacao
    circle = plt.Circle((0.5, 0.55), 0.35, color=destaque_cor,
                         alpha=0.15, transform=ax_class.transAxes)
    ax_class.add_patch(circle)

    ax_class.text(
        0.5, 0.72, "CLASSIFICACAO",
        ha="center", va="center", fontsize=11,
        fontweight="bold", color="#666666",
        transform=ax_class.transAxes,
    )
    ax_class.text(
        0.5, 0.52, classificacao.upper(),
        ha="center", va="center", fontsize=38,
        fontweight="bold", color=destaque_cor,
        transform=ax_class.transAxes,
    )

    # Nivel de confianca
    ax_class.text(
        0.5, 0.25, f"Confianca: {conf_nivel.upper()} ({conf_score:.2%})",
        ha="center", va="center", fontsize=14,
        fontweight="bold", color=conf_color,
        transform=ax_class.transAxes,
    )

    # ------------------------------------------------------------------
    # Painel B: Barras de probabilidade (topo direito)
    # ------------------------------------------------------------------
    ax_prob = fig.add_axes([0.55, 0.60, 0.38, 0.28])
    ax_prob.set_facecolor(BG_COLOR)

    classes_labels = ["Fake", "Verdadeira"]
    probs = [prob_fake, prob_true]
    bar_colors = [COLOR_FAKE, COLOR_TRUE]

    bars = ax_prob.barh(classes_labels, probs, color=bar_colors,
                        height=0.5, edgecolor="white", linewidth=1.5)

    # Anotacoes nas barras
    for bar, val in zip(bars, probs):
        ax_prob.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=13,
            fontweight="bold", color="#333333",
        )

    ax_prob.set_xlim(0, 115)
    ax_prob.set_xlabel("Probabilidade (%)", fontsize=10, color="#555555")
    ax_prob.set_title("Probabilidades por Classe", fontsize=12,
                      fontweight="bold", color="#333333", pad=10)
    ax_prob.tick_params(axis="y", labelsize=11)
    ax_prob.tick_params(axis="x", labelsize=9)
    ax_prob.spines["top"].set_visible(False)
    ax_prob.spines["right"].set_visible(False)
    ax_prob.spines["left"].set_color("#DDDDDD")
    ax_prob.spines["bottom"].set_color("#DDDDDD")

    # ------------------------------------------------------------------
    # Painel C: Palavras mais relevantes (metade inferior)
    # ------------------------------------------------------------------
    ax_words = fig.add_axes([0.08, 0.08, 0.85, 0.42])
    ax_words.set_facecolor(BG_COLOR)

    if palavras:
        top_n = min(10, len(palavras))
        top_words = palavras[:top_n]
        termos = [w["termo"] for w in top_words][::-1]
        pesos = [w["peso"] for w in top_words][::-1]

        # Gradiente de cor baseado na classificacao
        word_colors = [destaque_cor for _ in termos]
        # Ajusta alpha conforme o peso relativo
        max_peso = max(pesos) if pesos else 1
        alphas = [max(0.3, p / max_peso) for p in pesos]

        bar_colors_words = []
        for c, a in zip(word_colors, alphas):
            # Converte hex para rgba com alpha
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(c)
            bar_colors_words.append((*rgb, a))

        bars_w = ax_words.barh(termos, pesos, color=bar_colors_words,
                               height=0.6, edgecolor="white", linewidth=1.0)

        # Anotacoes
        for bar, val in zip(bars_w, pesos):
            ax_words.text(
                bar.get_width() + max(pesos) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9,
                color="#555555",
            )

        ax_words.set_xlabel("Peso no Vetor de Features", fontsize=10, color="#555555")
    else:
        ax_words.text(0.5, 0.5, "Nenhuma palavra relevante encontrada.",
                      ha="center", va="center", fontsize=12, color="#999999",
                      transform=ax_words.transAxes)

    ax_words.set_title(f"Top-{len(termos) if palavras else 0} Palavras Mais Relevantes",
                       fontsize=12, fontweight="bold", color="#333333", pad=10)
    ax_words.tick_params(axis="y", labelsize=10)
    ax_words.tick_params(axis="x", labelsize=9)
    ax_words.spines["top"].set_visible(False)
    ax_words.spines["right"].set_visible(False)
    ax_words.spines["left"].set_color("#DDDDDD")
    ax_words.spines["bottom"].set_color("#DDDDDD")

    # ------------------------------------------------------------------
    # Rodape com timestamp (opcional)
    # ------------------------------------------------------------------
    from datetime import datetime
    fig.text(
        0.5, 0.01,
        f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ha="center", fontsize=8, color="#AAAAAA",
    )

    # Salva
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)

    print(f"  Grafico de predicao salvo: {filepath}")
    return filepath
