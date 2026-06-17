"""
src/eda/report_generator.py
Gerador automático de relatório Markdown da Análise Exploratória de Dados.

Produz ``output/eda/eda_report.md`` com tabelas-resumo, estatísticas,
interpretações e referências aos gráficos gerados.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "output", "eda"
)


def generate_report(
    analysis_results: Dict,
    chart_paths: Dict[str, Dict[str, str]],
    output_path: str,
) -> str:
    """
    Gera um relatório completo em Markdown com os resultados da EDA.

    Parameters
    ----------
    analysis_results : dict
        Resultado de ``ExploratoryAnalysis.run_all()``.
    chart_paths : dict
        Resultado de ``visualization.generate_all_charts()``.
    output_path : str
        Caminho onde o arquivo ``.md`` será salvo.

    Returns
    -------
    str
        Caminho absoluto do relatório gerado.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    overview = analysis_results["dataset_overview"]
    text_stats = analysis_results["text_statistics"]
    freq = analysis_results["word_frequency"]
    vocab = analysis_results["vocabulary_analysis"]
    balance = analysis_results["balance_analysis"]

    lines: List[str] = []
    _ = lines.append  # alias

    # Achata chart_paths aninhado para lookup simples por chave
    flat_charts: Dict[str, str] = {}
    for category, paths in chart_paths.items():
        if isinstance(paths, dict):
            for key, path in paths.items():
                flat_charts[key] = path
        elif isinstance(paths, str):
            flat_charts[category] = paths

    # =================================================================
    # Cabeçalho
    # =================================================================
    _("# Análise Exploratória de Dados (EDA)")
    _(f"**Data de geração:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    _("")
    _("---")
    _("")

    # =================================================================
    # 1. Resumo do Dataset
    # =================================================================
    _("## 1. Resumo do Dataset")
    _("")
    _("| Métrica | Valor |")
    _("|---------|-------|")
    _(f"| Total de notícias | **{overview['total_samples']:,}** |".replace(",", "."))
    _(f"| Notícias Fake | {overview['fake_count']:,} ({overview['fake_pct']}%) |".replace(",", "."))
    _(f"| Notícias Verdadeiras | {overview['true_count']:,} ({overview['true_pct']}%) |".replace(",", "."))
    _(f"| Registros duplicados | {overview['duplicate_count']:,} |".replace(",", "."))
    _("")

    null_counts = overview.get("null_counts", {})
    null_total = sum(null_counts.values())
    if null_total > 0:
        _("### Valores Nulos")
        _("")
        _("| Coluna | Nulos |")
        _("|--------|-------|")
        for col, count in null_counts.items():
            if count > 0:
                _(f"| `{col}` | {count:,} |".replace(",", "."))
        _("")
    else:
        _("✅ **Nenhum valor nulo encontrado.**")
        _("")

    if overview["duplicate_count"] > 0:
        _("⚠️ Foram encontrados registros duplicados (baseados na coluna de texto). "
          "Recomenda-se removê-los antes do treinamento.")
    else:
        _("✅ **Nenhum registro duplicado encontrado.**")
    _("")

    # =================================================================
    # 2. Estatísticas de Texto
    # =================================================================
    _("## 2. Estatísticas de Texto")
    _("")

    for group, group_label in [("overall", "Geral"), ("fake", "Fake"), ("true", "Verdadeira")]:
        ts = text_stats.get(group, {})
        if not ts:
            continue
        _(f"### {group_label}")
        _("")
        _("| Métrica | Caracteres | Palavras |")
        _("|---------|-----------|----------|")
        _(_stat_line("Média",        ts, "mean"))
        _(_stat_line("Mediana",      ts, "median"))
        _(_stat_line("Desvio Padrão", ts, "std"))
        _(_stat_line("Mínimo",       ts, "min"))
        _(_stat_line("Máximo",       ts, "max"))
        _("")

    # Nota sobre sentenças
    sentence_note = text_stats.get("sentence_note", "")
    if sentence_note:
        _(f"> ℹ️ **Nota:** {sentence_note}")
        _("")

    # =================================================================
    # 3. Frequência de Palavras
    # =================================================================
    _("## 3. Frequência de Palavras (Top 20)")
    _("")

    for key, title in [("overall", "Geral"), ("fake", "Fake"), ("true", "Verdadeira")]:
        items = freq.get(key, [])
        if not items:
            continue
        _(f"### {title}")
        _("")
        _("| # | Palavra | Contagem | Frequência (%) |")
        _("|---|---------|----------|----------------|")
        for i, item in enumerate(items, 1):
            _(f"| {i} | `{item['word']}` | {item['count']:,} | {item['frequency']}% |".replace(",", "."))
        _("")

    # =================================================================
    # 4. Análise de Vocabulário
    # =================================================================
    _("## 4. Análise de Vocabulário")
    _("")
    _("| Classe | Total de Palavras | Palavras Únicas | Razão Únicas/Total |")
    _("|--------|-------------------|-----------------|---------------------|")
    for key, label in [("overall", "Geral"), ("fake", "Fake"), ("true", "Verdadeira")]:
        v = vocab.get(key, {})
        _(f"| {label} | {v.get('total_words', 0):,} | {v.get('unique_words', 0):,} | {v.get('unique_ratio', 0):.4f} |".replace(",", "."))
    _("")

    ratio_overall = vocab.get("overall", {}).get("unique_ratio", 0)
    if ratio_overall > 0.5:
        _("🔍 O vocabulário possui **alta diversidade léxica** "
          f"(razão únicas/total = {ratio_overall:.4f}), o que indica textos "
          "com pouca repetição de termos.")
    else:
        _("🔍 O vocabulário possui **diversidade léxica moderada** "
          f"(razão únicas/total = {ratio_overall:.4f}), o que indica "
          "certa repetição de termos entre os textos.")
    _("")

    # =================================================================
    # 5. Análise de Balanceamento
    # =================================================================
    _("## 5. Análise de Balanceamento")
    _("")
    _("| Métrica | Valor |")
    _("|---------|-------|")
    _(f"| Razão de desbalanceamento | {balance['imbalance_ratio']:.4f} |")
    _(f"| Fake | {balance['fake_count']:,} |".replace(",", "."))
    _(f"| Verdadeira | {balance['true_count']:,} |".replace(",", "."))
    _(f"| Avaliação | **{balance['assessment']}** |")
    _("")
    _(f"**Impacto nos modelos de ML:** {balance['ml_impact']}")
    _("")
    _(f"**Recomendação:** {balance['recommendation']}")
    _("")

    # =================================================================
    # 6. Correlações
    # =================================================================
    _("## 6. Correlações")
    _("")

    corr = analysis_results.get("correlation_analysis", {})

    # --- 6.1 Mutual Information ---
    mi_words = corr.get("mi_words", [])
    if mi_words:
        _("### 6.1 Mutual Information: Top Palavras × Label")
        _("")
        _(
            "*Quanto maior o MI, mais a palavra discrimina entre "
            "notícias fake e verdadeiras.*"
        )
        _("")
        _("| # | Palavra | Mutual Information |")
        _("|---|---------|---------------------|")
        for i, item in enumerate(mi_words, 1):
            _(f"| {i} | `{item['word']}` | {item['mi_score']:.6f} |")
        _("")

    # --- 6.2 Correlação ponto-bisserial ---
    char_corr = corr.get("char_corr")
    word_corr = corr.get("word_corr")
    if char_corr is not None or word_corr is not None:
        _("### 6.2 Correlação Ponto-Bisserial: Comprimento × Classe")
        _("")
        _("*Mede a associação linear entre o comprimento do texto "
          "(caracteres/palavras) e o label binário (fake=0, true=1).*")
        _("")
        _("| Métrica | Correlação (r) | p-valor |")
        _("|---------|---------------|---------|")
        if char_corr is not None:
            _(f"| Caracteres | {char_corr['correlation']:.6f} | {char_corr['p_value']:.6f} |")
        if word_corr is not None:
            _(f"| Palavras | {word_corr['correlation']:.6f} | {word_corr['p_value']:.6f} |")
        _("")

        # Interpretação
        if char_corr is not None and abs(char_corr["correlation"]) > 0.2:
            direction = (
                "mais longas" if char_corr["correlation"] > 0
                else "mais curtas"
            )
            _(
                f"🔍 Correlação moderada entre caracteres e classe "
                f"(r = {char_corr['correlation']:.3f}): "
                f"textos verdadeiros tendem a ser **{direction}**."
            )
        elif char_corr is not None:
            _(
                "🔍 A correlação entre comprimento e classe é **fraca** "
                f"(r = {char_corr['correlation']:.3f}), indicando que o "
                "modelo não usará o comprimento como atalho preditivo."
            )
        _("")

    # =================================================================
    # 7. Interpretação e Insights
    # =================================================================
    _("## 7. Interpretação e Insights")
    _("")

    # Identificar disparidade de comprimento
    fake_char_mean = text_stats.get("fake", {}).get("char_count", {}).get("mean", 0)
    true_char_mean = text_stats.get("true", {}).get("char_count", {}).get("mean", 0)
    fake_word_mean = text_stats.get("fake", {}).get("word_count", {}).get("mean", 0)
    true_word_mean = text_stats.get("true", {}).get("word_count", {}).get("mean", 0)

    insights: List[str] = []

    if true_char_mean > 0 and fake_char_mean > 0:
        ratio_char = true_char_mean / fake_char_mean
        if ratio_char > 1.5:
            insights.append(
                f"- **Disparidade de comprimento:** As notícias verdadeiras são, "
                f"em média, **{ratio_char:.1f}x mais longas** que as falsas "
                f"({true_char_mean:.0f} vs {fake_char_mean:.0f} caracteres). "
                f"Isso pode fazer com que o modelo aprenda a distinguir classes "
                f"pelo comprimento do texto em vez do conteúdo — considere "
                f"normalizar ou balancear esta característica."
            )
        elif ratio_char < 0.67:
            insights.append(
                f"- **Disparidade de comprimento (invertida):** As notícias falsas "
                f"são **{1/ratio_char:.1f}x mais longas** que as verdadeiras."
            )

    if true_word_mean > 0 and fake_word_mean > 0:
        ratio_word = true_word_mean / fake_word_mean
        if 0.8 <= ratio_word <= 1.2:
            insights.append(
                "- **Comprimento equilibrado:** O número médio de palavras é "
                "similar entre as classes, o que é desejável para o modelo "
                "não usar o comprimento como atalho."
            )

    if balance["imbalance_ratio"] >= 0.90:
        insights.append(
            "- **Dataset balanceado:** A proporção equilibrada entre classes "
            "elimina a necessidade de técnicas de reamostragem e torna a "
            "acurácia uma métrica confiável."
        )

    # Comparar diversidade de vocabulário entre classes
    fake_unique_ratio = vocab.get("fake", {}).get("unique_ratio", 0)
    true_unique_ratio = vocab.get("true", {}).get("unique_ratio", 0)
    if true_unique_ratio > 0 and fake_unique_ratio > 0:
        if abs(true_unique_ratio - fake_unique_ratio) > 0.1:
            richer = "verdadeiras" if true_unique_ratio > fake_unique_ratio else "falsas"
            insights.append(
                f"- **Diversidade léxica:** As notícias **{richer}** possuem "
                f"vocabulário mais diverso (razão únicas/total: "
                f"{max(true_unique_ratio, fake_unique_ratio):.4f} vs "
                f"{min(true_unique_ratio, fake_unique_ratio):.4f})."
            )

    if not insights:
        insights.append("- Nenhum padrão atípico identificado na análise exploratória.")
        insights.append("- O dataset aparenta estar bem-formado para tarefas de classificação.")

    for ins in insights:
        _(ins)

    _("")

    # =================================================================
    # 8. Referência das Visualizações
    # =================================================================
    _("## 8. Referência das Visualizações")
    _("")

    def _img(key: str, caption: str) -> None:
        """Renderiza uma imagem ou aviso de indisponibilidade."""
        if key in flat_charts:
            fname = os.path.basename(flat_charts[key])
            _(f"![{caption}]({fname})")
            _(f"*{caption}*")
        else:
            _(f"*({caption} — não disponível)*")
        _("")

    _("### Distribuição de Classes")
    _img("bar", "Gráfico de barras — Distribuição de classes")
    _img("pie", "Gráfico de pizza — Distribuição de classes")

    _("### Comprimento dos Textos")
    _img("char_count_hist", "Histograma — Caracteres por classe")
    _img("word_count_hist", "Histograma — Palavras por classe")

    _("### Boxplots")
    _img("char_count_box", "Boxplot — Caracteres por classe")
    _img("word_count_box", "Boxplot — Palavras por classe")

    _("### Top 20 Palavras")
    _img("top_overall", "Top 20 palavras — Geral")
    _img("top_fake", "Top 20 palavras — Fake")
    _img("top_true", "Top 20 palavras — Verdadeira")

    _("### Nuvens de Palavras")
    _img("wc_overall", "Word Cloud — Geral")
    _img("wc_fake", "Word Cloud — Fake")
    _img("wc_true", "Word Cloud — Verdadeira")

    _("### Correlações")
    _img("mi_heatmap", "Mutual Information — Top 20 palavras × Label")

    _("---")
    _("")
    _("*Relatório gerado automaticamente pelo módulo EDA do SIATD.*")

    # =================================================================
    # Escrita
    # =================================================================
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Relatório EDA salvo: {output_path}")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_line(label: str, stats: Dict, key: str) -> str:
    """
    Formata uma linha da tabela de estatísticas com valor para caracteres e palavras.

    Parameters
    ----------
    label : str
        Nome da métrica (ex.: "Média").
    stats : dict
        Sub-dicionário com ``char_count`` e ``word_count``.
    key : str
        Chave dentro de cada sub-métrica (ex.: ``"mean"``).

    Returns
    -------
    str
        Linha Markdown: ``| Média | 2828 | 366 |``
    """
    char_val = stats.get("char_count", {}).get(key, "-")
    word_val = stats.get("word_count", {}).get(key, "-")

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return str(v)

    return f"| **{label}** | {_fmt(char_val)} | {_fmt(word_val)} |"


# ---------------------------------------------------------------------------
# Execução standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from src.eda.exploratory_analysis import ExploratoryAnalysis
    from src.eda.visualization import generate_all_charts

    default_csv = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "data", "preprocessed", "pre-processed.csv"
    )
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else default_csv

    eda = ExploratoryAnalysis(csv_arg)
    results = eda.run_all()
    charts = generate_all_charts(
        eda.df,
        results["word_frequency"],
        results.get("correlation_analysis"),
    )
    rpt_path = generate_report(
        results, charts,
        os.path.join(OUTPUT_DIR, "eda_report.md")
    )
    print(f"\nRelatório: {rpt_path}")
