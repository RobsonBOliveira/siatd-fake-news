# SATD – Sistema de Apoio à Tomada de Decisão para Fake News

Sistema de aprendizado de máquina para análise de notícias em português,
baseado no [Fake.Br Corpus](https://github.com/roneysco/Fake.br-Corpus).

---

## Estrutura do Projeto

```
fake_news_satd/
├── data/
│   ├── preprocessed/      ← CSVs do Fake.Br Corpus (pasta "preprocessed")
│   ├── raw/               ← CSVs originais (se desejar aplicar pré-proc. próprio)
│   ├── train/             ← Splits de treino (gerado automaticamente)
│   └── test/              ← Splits de teste (gerado automaticamente)
├── models/
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── random_forest.pkl
│   ├── tfidf.pkl          ← Vetorizador persistido
│   └── meta.pkl
├── src/
│   ├── preprocessing/
│   │   └── text_processor.py    ← Pipeline de limpeza e normalização
│   ├── feature_extraction/
│   │   └── vectorizer.py        ← TF-IDF e Bag of Words
│   ├── training/
│   │   └── trainer.py           ← Treinamento dos 3 modelos
│   ├── evaluation/
│   │   └── metrics.py           ← Métricas e matrizes de confusão
│   └── prediction/
│       └── predictor.py         ← SATD: probabilidade, confiança, palavras
├── input/
│   └── noticia.txt              ← Notícia a classificar
├── output/
│   ├── resultado.json           ← Saída do SATD
│   ├── model_comparison.json    ← Métricas comparativas
│   └── confusion_matrix_*.png   ← Matrizes de confusão
├── run.py                       ← Ponto de entrada principal
└── requirements.txt
```

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Uso

### 1. Treinamento

Coloque o CSV do Fake.Br Corpus em `data/preprocessed/` e execute:

```bash
python run.py train data/preprocessed/fake_news.csv
```

O CSV deve conter as colunas `text` (ou `preprocessed_news`) e `label` (`fake`/`true`).

Por padrão, assume-se que o texto já está pré-processado (pasta `preprocessed`).
Para aplicar o pipeline de limpeza em CSVs brutos:

```bash
python run.py train data/raw/fake_news.csv --raw
```

#### Opções de vetorizador

```bash
# TF-IDF (padrão, recomendado)
python run.py train data/preprocessed/fake_news.csv --vec tfidf

# Bag of Words
python run.py train data/preprocessed/fake_news.csv --vec bow
```

---

### 2. Predição

Coloque o arquivo `.txt` da notícia em `input/` e execute:

```bash
python run.py predict input/noticia.txt
```

Escolhendo o modelo:

```bash
python run.py predict input/noticia.txt --model naive_bayes
python run.py predict input/noticia.txt --model svm           # padrão
python run.py predict input/noticia.txt --model random_forest
```

---

### 3. Demonstração (sem dados reais)

```bash
python run.py demo
```

Gera dados sintéticos, treina e executa uma predição de ponta a ponta.

---

## Saída do SATD (`output/resultado.json`)

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

| Modelo         | Classe sklearn            | `predict_proba` |
|----------------|---------------------------|-----------------|
| Naive Bayes    | `MultinomialNB`           | Nativo          |
| SVM            | `LinearSVC` + Calibração  | Via `CalibratedClassifierCV` |
| Random Forest  | `RandomForestClassifier`  | Nativo          |

---

## Métricas Avaliadas

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (PNG em `output/`)
- Comparação entre modelos (`output/model_comparison.json`)

---

## Roadmap

- [ ] Integração com Word2Vec / FastText
- [ ] BERTimbau (transformers)
- [ ] Interface web (Flask/Streamlit)
- [ ] Busca por similaridade na base de treino
- [ ] Relatório PDF exportável
