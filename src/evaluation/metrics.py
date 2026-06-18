"""
src/evaluation/metrics.py
Cálculo e exibição de métricas de avaliação dos modelos.
"""

import numpy as np
import os
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")

# Cores consistentes por modelo
MODEL_COLORS = {
    "naive_bayes":   "#3498DB",  # azul
    "svm":           "#E74C3C",  # vermelho
    "random_forest": "#2ECC71",  # verde
}
MODEL_LABELS = {
    "naive_bayes":   "Naive Bayes",
    "svm":           "SVM",
    "random_forest": "Random Forest",
}


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n--- Métricas: {model_name} ---")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1_score']:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Fake", "Verdadeira"], zero_division=0))

    if HAS_PLOT:
        _plot_confusion_matrix(metrics["confusion_matrix"], model_name)

    return metrics


def _plot_confusion_matrix(cm: list, model_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(cm), annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake", "Verdadeira"],
        yticklabels=["Fake", "Verdadeira"], ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Confusion Matrix – {model_name}")
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Matriz salva: {path}")


def _plot_model_comparison(all_metrics: dict) -> str:
    """
    Gera grafico de barras agrupadas comparando Accuracy, Precision,
    Recall e F1-Score entre os tres modelos.

    Returns:
        Caminho do arquivo PNG gerado, ou string vazia se matplotlib indisponivel.
    """
    if not HAS_PLOT:
        return ""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, "model_comparison.png")

    model_names = list(all_metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    n_models = len(model_names)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    bar_width = 0.25
    # Centraliza o grupo de barras
    offsets = np.linspace(
        -bar_width * (n_models - 1) / 2,
        bar_width * (n_models - 1) / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for i, name in enumerate(model_names):
        values = [all_metrics[name][k] for k in metric_keys]
        color = MODEL_COLORS.get(name, "#999999")
        label = MODEL_LABELS.get(name, name)
        bars = ax.bar(
            x + offsets[i], values, bar_width,
            color=color, alpha=0.85, edgecolor="white",
            linewidth=0.8, label=label, zorder=3,
        )
        # Anotacoes de valor no topo de cada barra
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11, color="#555555")
    ax.set_title("Comparacao de Modelos – Metricas de Avaliacao",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9,
              edgecolor="#CCCCCC")
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor="none")
    plt.close(fig)
    return filepath


def compare_models(all_metrics: dict):
    """Exibe tabela comparativa, gera grafico e salva JSON com resultados."""
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE MODELOS")
    print("=" * 60)
    header = f"{'Modelo':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "model_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {out_path}")

    # Gera grafico comparativo
    chart_path = _plot_model_comparison(all_metrics)
    if chart_path:
        print(f"Grafico comparativo salvo em: {chart_path}")
