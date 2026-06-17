"""
run.py
Ponto de entrada principal do SATD de Fake News.

Uso:
  python run.py train   <caminho_csv> [--vec tfidf|bow] [--eda]
  python run.py predict <caminho_txt> [--model naive_bayes|svm|random_forest]
  python run.py eda     <caminho_csv> [--top-n N]
  python run.py demo    (usa dados sintéticos para demonstração)
"""

import argparse
import os
import sys
import json

# Garante saída UTF-8 no terminal Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))


def _run_eda(csv_path: str, top_n: int = 20):
    """Executa o pipeline de EDA e retorna (results, chart_paths, report_path)."""
    from src.eda.exploratory_analysis import ExploratoryAnalysis
    from src.eda.visualization import generate_all_charts
    from src.eda.report_generator import generate_report

    eda_output = os.path.join(os.path.dirname(__file__), "output", "eda")

    eda = ExploratoryAnalysis(csv_path)
    results = eda.run_all()
    chart_paths = generate_all_charts(
        eda.df, results["word_frequency"],
        results.get("correlation_analysis"), eda_output,
    )
    report_path = generate_report(results, chart_paths,
                                  os.path.join(eda_output, "eda_report.md"))
    return results, chart_paths, report_path


def cmd_train(args):
    # EDA opcional antes do treinamento
    if hasattr(args, "eda") and args.eda:
        print("\n>>> Executando EDA antes do treinamento...")
        _, _, rpt = _run_eda(args.csv)
        print(f">>> Relatório EDA: {rpt}")
        print(">>> EDA concluida. Iniciando treinamento...\n")

    from src.training.trainer import train_all
    from src.evaluation.metrics import compare_models

    print(f"\n>>> Iniciando treinamento com: {args.csv}")
    print(f">>> Vetorizador: {args.vec}")
    metrics = train_all(
        csv_path=args.csv,
        vectorizer_type=args.vec,
        already_preprocessed=not args.raw,
    )
    compare_models(metrics)
    print("\n[OK] Treinamento concluído.")


def cmd_eda(args):
    """Executa a Análise Exploratória de Dados."""
    print(f"\n>>> Iniciando EDA com: {args.csv}")
    _, _, rpt = _run_eda(args.csv, top_n=args.top_n)
    print(f"\n>>> Relatório EDA salvo em: {rpt}")
    print("[OK] EDA concluida.")


def cmd_predict(args):
    from src.prediction.predictor import predict_from_file

    print(f"\n>>> Analisando notícia: {args.txt}")
    print(f">>> Modelo: {args.model} | Vetorizador: {args.vec}")
    result = predict_from_file(args.txt, model_name=args.model,
                                vectorizer_type=args.vec)
    return result


def cmd_demo(_args):
    """
    Demonstração completa com dados sintéticos para validar o pipeline
    sem precisar do dataset real.
    """
    import pandas as pd
    import numpy as np
    import tempfile
    from src.training.trainer import train_all
    from src.evaluation.metrics import compare_models
    from src.prediction.predictor import FakeNewsPredictor

    print("\n" + "=" * 60)
    print("  MODO DEMONSTRAÇÃO (dados sintéticos)")
    print("=" * 60)

    # Gera dataset sintético balanceado
    np.random.seed(42)
    fake_vocab = [
        "vacina", "microchip", "governo", "esconde", "verdade", "revelado",
        "conspiracao", "mentira", "proibido", "censurado", "illuminati",
        "fraude", "manipulacao", "segredo", "oculto",
    ]
    true_vocab = [
        "pesquisa", "estudo", "universidade", "cientistas", "dados",
        "resultado", "analise", "publicado", "revista", "relatorio",
        "governo", "ministro", "anunciou", "aprovado", "conferencia",
    ]

    def gen_text(vocab, n_words=30):
        return " ".join(np.random.choice(vocab, n_words))

    n = 400
    texts  = [gen_text(fake_vocab) for _ in range(n // 2)] + \
             [gen_text(true_vocab) for _ in range(n // 2)]
    labels = ["fake"] * (n // 2) + ["true"] * (n // 2)

    df = pd.DataFrame({"text": texts, "label": labels})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        df.to_csv(f, index=False)
        csv_path = f.name

    print(f"  Dataset sintético: {n} amostras ({n//2} fake, {n//2} true)")

    # Treina
    metrics = train_all(csv_path, vectorizer_type="tfidf",
                         already_preprocessed=True)
    compare_models(metrics)

    # Predição de exemplo
    print("\n>>> Testando predição com texto de exemplo...")
    predictor = FakeNewsPredictor(model_name="svm", vectorizer_type="tfidf")
    sample    = "governo esconde verdade sobre vacina microchip conspiracao revelado"
    result    = predictor.predict(sample)

    print("\n--- Resultado do SATD ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    os.unlink(csv_path)
    print("\n[OK] Demonstração concluída.")


def main():
    parser = argparse.ArgumentParser(
        description="SATD – Sistema de Apoio à Tomada de Decisão para Fake News"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Treina os modelos")
    p_train.add_argument("csv", help="Caminho para o CSV do Fake.Br Corpus")
    p_train.add_argument("--vec", choices=["tfidf", "bow"], default="tfidf",
                          help="Tipo de vetorizador (padrão: tfidf)")
    p_train.add_argument("--raw", action="store_true",
                          help="Aplica pré-processamento (para CSVs não processados)")
    p_train.add_argument("--eda", action="store_true",
                          help="Executa EDA antes do treinamento")

    # predict
    p_pred = sub.add_parser("predict", help="Classifica uma notícia .txt")
    p_pred.add_argument("txt", help="Caminho para o arquivo .txt da notícia")
    p_pred.add_argument("--model", choices=["naive_bayes", "svm", "random_forest"],
                         default="svm", help="Modelo a usar (padrão: svm)")
    p_pred.add_argument("--vec", choices=["tfidf", "bow"], default="tfidf",
                         help="Vetorizador utilizado no treino (padrão: tfidf)")

    # eda
    p_eda = sub.add_parser("eda", help="Executa Análise Exploratória de Dados")
    p_eda.add_argument("csv", help="Caminho para o CSV do Fake.Br Corpus")
    p_eda.add_argument("--top-n", type=int, default=20,
                        help="Número de palavras mais frequentes (padrão: 20)")

    # demo
    sub.add_parser("demo", help="Demonstração completa com dados sintéticos")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "eda":
        cmd_eda(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
