# Santander Customer Transaction Prediction

Projeto de classificação binária aplicado ao desafio [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) do Kaggle.

**Objetivo:** prever se um cliente realizará uma transação futura, independentemente do valor.  
**Métrica:** AUC-ROC  
**Disciplina:** Aulas de Dados e Aprendizado de Máquina — Engenharia Elétrica / IFES

## Resultados

| Modelo | AUC-ROC | Average Precision | F1 (classe positiva) |
|---|---|---|---|
| **Naive Bayes** | **0,8884** | **0,5837** | **0,4812** |
| SVM (sub 20k) | 0,8676 | 0,5260 | 0,4235 |
| Random Forest | 0,8399 | 0,4428 | 0,1202 |

> AUC e AP do SVM avaliados em subamostra estratificada de 20.000 amostras (limitação computacional do kernel RBF).

## Estrutura

```
├── eda.py                  # Análise exploratória
├── train.py                # Treinamento dos 3 modelos
├── analise.py              # Análise comparativa e geração de plots
├── relatorio.tex           # Relatório em LaTeX
├── relatorio.pdf           # Relatório compilado
├── analise_output/         # Curvas ROC, P-R, matrizes de confusão, importância de features
├── models/                 # Modelos serializados (naive_bayes, svm, scaler)
└── data/                   # sample_submission.csv (train/test não incluídos — >100MB)
```

## Como reproduzir

**1. Instalar dependências**
```bash
pip install pandas numpy scikit-learn matplotlib
```

**2. Baixar os dados** em [kaggle.com/competitions/santander-customer-transaction-prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) e colocar em `data/`.

**3. Executar**
```bash
python eda.py       # análise exploratória → eda_output/
python train.py     # treina modelos → models/ + submissions
python analise.py   # análise comparativa → analise_output/
```

## Modelos

- **Random Forest** — ensemble de 200 árvores, `class_weight='balanced'`, validação com `StratifiedKFold(5)`
- **Naive Bayes Gaussiano** — classificador probabilístico, assume independência condicional entre features
- **SVM (kernel RBF)** — subamostra estratificada de 20k amostras por limitação de custo $O(n^2)$

Todos os modelos usam `StandardScaler` e `StratifiedKFold(5)` para validação cruzada.
