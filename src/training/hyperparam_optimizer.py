"""
src/training/hyperparam_optimizer.py
Otimizacao de hiperparametros via GridSearchCV para os tres modelos do SATD.
"""

import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------------
# Grades de hiperparametros para cada modelo
# ---------------------------------------------------------------------------

NAIVE_BAYES_GRID = {
    "alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
    "fit_prior": [True, False],
}
"""10 combinacoes (5 x 2)."""

SVM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "max_iter": [1000, 2000, 3000],
    "loss": ["hinge", "squared_hinge"],
}
"""24 combinacoes (4 x 3 x 2)."""

RANDOM_FOREST_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
"""24 combinacoes (2 x 3 x 2 x 2). Grid reduzido para evitar travamento
por consumo excessivo de memoria no Windows com GridSearchCV + n_jobs=-1.
Ainda cobre bem o espaco de busca: arvores, profundidade, split e leaf."""

# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------


def get_param_grid(model_name: str) -> dict:
    """Retorna a grade de hiperparametros para o modelo indicado."""
    grids = {
        "naive_bayes": NAIVE_BAYES_GRID,
        "svm": SVM_GRID,
        "random_forest": RANDOM_FOREST_GRID,
    }
    if model_name not in grids:
        raise ValueError(
            f"Modelo '{model_name}' desconhecido. "
            f"Opcoes: {list(grids.keys())}"
        )
    return grids[model_name]


def get_base_estimator(model_name: str):
    """
    Retorna uma instancia NAO treinada do estimador base para o modelo.

    Parametros fixos (nao otimizaveis):
      - LinearSVC: dual='auto', random_state=42
      - RandomForest: n_jobs=-1, random_state=42
    """
    if model_name == "naive_bayes":
        return MultinomialNB()
    elif model_name == "svm":
        return LinearSVC(dual="auto", random_state=42)
    elif model_name == "random_forest":
        # n_jobs=1 para evitar nested parallelism: o GridSearchCV ja gerencia
        # o paralelismo externo (via n_jobs). Com n_jobs=-1 aqui, cada worker
        # do GridSearch tentaria usar todos os nucleos, gerando contencao
        # e travamento no Windows (especialmente com matrizes esparsas).
        return RandomForestClassifier(n_jobs=1, random_state=42)
    else:
        raise ValueError(
            f"Modelo '{model_name}' desconhecido. "
            f"Opcoes: naive_bayes, svm, random_forest"
        )


def optimize_model(model_name: str, X_train, y_train, cv: int = 3,
                   verbose: int = 1):
    """
    Executa GridSearchCV para o modelo indicado.

    Args:
        model_name: "naive_bayes", "svm" ou "random_forest".
        X_train: matriz de features de treino (sparse ou densa).
        y_train: rotulos de treino.
        cv: numero de folds para validacao cruzada.
        verbose: nivel de verbosidade do GridSearchCV.

    Returns:
        tuple[dict, float]: (melhores_parametros, melhor_score_f1_cv)
    """
    param_grid = get_param_grid(model_name)
    estimator = get_base_estimator(model_name)

    # Limita paralelismo para evitar travamento no Windows:
    # - n_jobs=-1 com matrix esparsa + RandomForest esgota a RAM rapidamente
    #   (cada worker do joblib recebe uma copia completa dos dados).
    # - Usamos no maximo 2 workers, que e seguro para a maioria das maquinas.
    # - pre_dispatch='n_jobs' evita que o GridSearchCV prepare todos os jobs
    #   de uma vez, o que tambem consumiria memoria excessiva.
    safe_n_jobs = min(os.cpu_count() or 2, 2)

    gs = GridSearchCV(
        estimator,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=safe_n_jobs,
        verbose=verbose,
        return_train_score=False,
        pre_dispatch=safe_n_jobs,
    )
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    best_score = gs.best_score_

    print(f"\n  Melhores hiperparametros para {model_name}:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    print(f"  F1 medio (CV={cv}): {best_score:.4f}")

    return best_params, best_score
