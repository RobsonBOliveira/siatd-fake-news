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


def compare_models(all_metrics: dict):
    """Exibe tabela comparativa e salva JSON com resultados."""
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
