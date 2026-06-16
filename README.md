# SIATD – Sistema inteligente de Apoio à Tomada de Decisão para a verificação de notícias.

Sistema de aprendizado de máquina para análise de notícias em português,
baseado no [Fake.Br Corpus](https://github.com/roneysco/Fake.br-Corpus).

---

## Estrutura do Projeto

```
siatd_fake_news/
├── data/
│   ├── preprocessed/           
│   │   └── pre-processed.csv    ← CSV do Fake.Br Corpus
│   └── raw/                     ← CSVs originais (se desejar aplicar pré-proc. próprio)
├── models/                      ← Modelos persistidos (pós-treino)
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── random_forest.pkl
│   ├── tfidf.pkl
│   └── meta.pkl
├── src/
│   ├── eda/
│   │   ├── exploratory_analysis.py ← Análise exploratória de dados
│   │   ├── visualization.py        ← Gráficos e word clouds
│   │   └── report_generator.py     ← Relatório automático (Markdown)
│   ├── preprocessing/
│   │   └── text_processor.py    ← Pipeline de limpeza e normalização
│   ├── feature_extraction/
│   │   └── vectorizer.py        ← TF-IDF e Bag of Words
│   ├── training/
│   │   └── trainer.py           ← Treinamento dos 3 modelos
│   ├── evaluation/
│   │   └── metrics.py           ← Métricas e matrizes de confusão
│   └── prediction/
│       └── predictor.py         ← SIATD: probabilidade, confiança, palavras
├── input/                       ← Notícia a classificar
│   └── noticia.txt
├── output/
│   ├── eda/                     ← Saídas da EDA (gráficos + relatório)
│   │   ├── eda_report.md
│   │   ├── class_distribution_*.png
│   │   ├── char_count_*.png
│   │   ├── word_count_*.png
│   │   ├── top20_words_*.png
│   │   └── wordcloud_*.png
│   ├── resultado.json           ← Saída do SIATD
│   ├── model_comparison.json    ← Métricas comparativas
│   └── confusion_matrix_*.png   ← Matrizes de confusão
├── run.py                       ← Ponto de entrada principal
├── clean.py                     ← Rotina de limpeza
├── requirements.txt
└── .gitignore
```

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Uso

### 0. Análise Exploratória de Dados (EDA)

Antes do treinamento, recomenda-se executar a EDA para compreender
as características do dataset:

```bash
python run.py eda data/preprocessed/pre-processed.csv
```

O comando gera gráficos e um relatório Markdown em `output/eda/`:

| Artefato | Descrição |
|----------|-----------|
| `class_distribution_bar.png` / `pie.png` | Distribuição das classes (fake/true) |
| `char_count_histogram.png` / `boxplot.png` | Comprimento dos textos em caracteres |
| `word_count_histogram.png` / `boxplot.png` | Comprimento dos textos em palavras |
| `top20_words_overall.png` / `fake.png` / `true.png` | Top-20 palavras mais frequentes |
| `wordcloud_overall.png` / `fake.png` / `true.png` | Nuvens de palavras |
| `eda_report.md` | Relatório completo com tabelas e interpretações |

#### Integração com o treinamento

Para executar a EDA automaticamente antes do treinamento:

```bash
python run.py train data/preprocessed/pre-processed.csv --eda
```

O fluxo completo é: **Carregamento → EDA → Pré-processamento → Vetorização → Treinamento → Avaliação**.

---

### 1. Treinamento

Coloque o CSV do Fake.Br Corpus em `data/preprocessed/` e execute:

```bash
python run.py train data/preprocessed/pre-processed.csv
```

O CSV deve conter as colunas `text` (ou `preprocessed_news`) e `label` (`fake`/`true`).

Por padrão, assume-se que o texto já está pré-processado (pasta `preprocessed`).
Para aplicar o pipeline de limpeza em CSVs brutos:

```bash
python run.py train data/raw/nome_do_arquivo.csv --raw
```

#### Opções de vetorizador

```bash
# TF-IDF (padrão, recomendado)
python run.py train data/preprocessed/nome_do_arquivo.csv --vec tfidf

# Bag of Words
python run.py train data/preprocessed/nome_do_arquivo.csv --vec bow
```

---

### 2. Predição

Coloque o arquivo `.txt` da notícia em `input/` e execute:

```bash
python run.py predict input/nome_do_arquivo.txt
```

Escolhendo o modelo:

```bash
python run.py predict input/nome_do_arquivo.txt --model naive_bayes
python run.py predict input/nome_do_arquivo.txt --model svm           # padrão
python run.py predict input/nome_do_arquivo.txt --model random_forest
```

---

### 3. Demonstração (sem dados reais)

```bash
python run.py demo
```

Gera dados sintéticos, treina e executa uma predição de ponta a ponta.

---

## Saída do SIATD (`output/resultado.json`)

```json
{
  "classificacao": "Fake",
  "probabilidade": {
    "Fake": "87.0%",
    "Verdadeira": "13.0%"
  },
  "confianca": {
    "nivel": "Alto",
    "score": 0.87
  },
  "palavras_relevantes": [
    { "termo": "governo esconde", "peso": 0.38 },
    { "termo": "conspiracao", "peso": 0.35 }
  ],
  "texto_processado": "governo esconde verdade vacina...",
  "modelo_utilizado": "svm",
  "vetorizador": "tfidf"
}
```

| Campo               | Descrição |
|---------------------|-----------|
| `classificacao`     | `Fake` ou `Verdadeira` |
| `probabilidade`     | Percentual estimado para cada classe |
| `confianca.nivel`   | `Baixo` (<65%), `Médio` (65–80%), `Alto` (>80%) |
| `palavras_relevantes` | Top-15 termos TF-IDF que contribuíram para a predição |
| `texto_processado`  | Texto após o pipeline de limpeza (trecho) |

---

## Pipeline de Pré-Processamento

O módulo `src/preprocessing/text_processor.py` reproduz o padrão do Fake.Br Corpus:

1. Conversão para minúsculas
2. Remoção de URLs
3. Remoção de números isolados
4. Remoção de pontuação
5. Remoção de acentos (unidecode)
6. Remoção de espaços extras
7. Tokenização + remoção de stopwords PT (NLTK)

---

## Modelos Implementados

| Modelo         | Classe sklearn            | `predict_proba`              |
|----------------|---------------------------|------------------------------|
| Naive Bayes    | `MultinomialNB`           | Nativo                       |
| SVM            | `LinearSVC` + Calibração  | Via `CalibratedClassifierCV` |
| Random Forest  | `RandomForestClassifier`  | Nativo                       |

---

## Métricas Avaliadas

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (PNG em `output/`)
- Comparação entre modelos (`output/model_comparison.json`)

---
