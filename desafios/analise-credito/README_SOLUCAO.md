# Solução — Análise de Crédito

![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white "Python 3.10+")
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white "Scikit-Learn")
![LightGBM](https://img.shields.io/badge/LightGBM-000000?style=for-the-badge "LightGBM")

## Objetivo

Investigar fatores associados à inadimplência e avaliar uma **Regressão Logística** para apoiar a identificação de propostas de maior risco.

Os dados são sintéticos e educacionais. O modelo não deve ser usado em decisões reais sem validação adicional.

## Glossário

| Termo | Definição |
|-------|-----------|
| **Champion** | Modelo principal (Regressão Logística), escolhido pela interpretabilidade e conformidade regulatória |
| **Challenger** | Modelo alternativo (LightGBM), otimizado para performance preditiva máxima |
| **PR-AUC** | *Precision-Recall Area Under Curve* — métrica principal para dados desbalanceados (34% inadimplentes) |
| **ROC-AUC** | *Receiver Operating Characteristic Area Under Curve* — métrica secundária |
| **Odds Ratio** | Razão de chances: `exp(coeficiente)`. Para variáveis padronizadas, indica mudança por 1 desvio-padrão |
| **dataset_tratado.csv** | Saída do script `tratamento_dados.py` — dados limpos com variável derivada |
| **Threshold** | Limiar de probabilidade para classificar como inadimplente (otimizado via CV no treino) |
| **class_weight='balanced'** | Ajusta pesos das classes inversamente proporcionais à frequência no treino |

## Estrutura

- `dataset.csv`: dados brutos fornecidos pelo desafio.
- `tratamento_dados.py`: tratamento determinístico e engenharia da variável derivada.
- `dataset_tratado.csv`: saída reproduzível do tratamento.
- `analise_credito.ipynb`: análise exploratória, preparação, modelagem e avaliação.
- `apresentacao.md`: roteiro para apresentação da análise.
- `requirements.txt`: dependências Python.

## Execução

Na pasta `desafios/analise-credito`:

```bash
python -m venv venv
```

Ativação do ambiente virtual:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Prompt de Comando)
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

Instalação e execução:

```bash
python -m pip install -r requirements.txt
python tratamento_dados.py
jupyter notebook analise_credito.ipynb
```

O notebook deve ser executado do início ao fim. Ele também recria `dataset_tratado.csv`.

## Tratamento dos dados

- Remove espaços extras em `estado`, `escolaridade` e `finalidade`.
- Corrige 15 idades exatamente dez vezes maiores que uma idade plausível.
- Marca 7 tempos de emprego acima de 30 anos como ausentes; a imputação ocorre somente no treino, dentro do pipeline.
- Remove 19 duplicatas exatas adicionais.
- Cria `razao_emprestimo_renda = valor_solicitado / renda_mensal`.

Resultado: 4.981 propostas, 12 colunas e 27 estados padronizados.

## Modelagem

A variável resposta é `inadimplente`, uma classificação binária. Foi usada divisão estratificada 80/20, `random_state=42` e pipeline com:

- imputação da mediana para numéricas;
- padronização das numéricas (apenas Champion/LogisticRegression);
- one-hot encoding das categóricas;
- Regressão Logística com `class_weight='balanced'`;
- Matplotlib e Seaborn para Análise Exploratória e visualização dos resultados.
- Não foi realizada otimização de hiperparâmetros: a Regressão Logística utilizou o valor padrão `C=1.0`; a comparação foi feita entre conjuntos de atributos e não por busca de `C`.

**Otimização de threshold:** O limiar de decisão é selecionado via validação cruzada no treino (maximiza F1-score), não no teste — evitando *test leakage*.

Comparações:

1. baseline que sempre prevê a classe majoritária;
2. `M1`: variáveis originais;
3. `M2`: variáveis originais + razão empréstimo/renda;
4. `M3`: variáveis originais sem `estado` + razão empréstimo/renda.

A seleção é feita pela PR-AUC média em validação cruzada estratificada de 5 partes no treino. O conjunto de teste é reservado para a estimativa final.

## Resultados no teste

| Modelo | Accuracy | ROC-AUC | PR-AUC | Recall | Precisão | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.656 | 0.500 | 0.344 | 0.000 | 0.000 | 0.000 | 0.500 |
| M1 — originais | 0.727 | 0.760 | 0.633 | 0.449 | 0.650 | 0.531 | 0.661 |
| M2 — originais + razão | 0.732 | 0.766 | 0.643 | 0.458 | 0.660 | 0.540 | 0.667 |
| M3 — sem estado + razão | 0.730 | 0.771 | 0.648 | 0.449 | 0.658 | 0.534 | 0.663 |

O modelo selecionado pela PR-AUC média na validação cruzada foi o **M3**, com PR-AUC média de 0.654 e ROC-AUC média de 0.777. No teste, ele alcançou PR-AUC de 0.648 e ROC-AUC de 0.771.

**Com threshold otimizado via CV (0.47):** Recall sobe para 0.723, F1 para 0.627, Balanced Accuracy para 0.709.

Em termos práticos, o modelo M3 com threshold otimizado captura ~72% dos reais inadimplentes. A precisão de 0.554 indica que, das propostas alertadas como risco, o modelo acerta em ~55% das vezes. A decisão final deve considerar o custo financeiro de falsos positivos e falsos negativos.

## Abordagem Champion vs. Challenger

Para refletir cenários reais da indústria de crédito, este projeto adota uma abordagem de dois modelos:

**Modelo Interpretável — Champion (Regressão Logística M3):** Foco principal do projeto. Escolhido por permitir extração de Odds Ratios, atendendo a requisitos regulatórios e de governança (explicabilidade nas recusas de crédito). Usa `class_weight='balanced'` e threshold otimizado via CV.

**Modelo de Alta Performance — Challenger (LightGBM):** Prova de conceito apresentada no final do notebook. Usa **o mesmo pré-processamento do Champion** (SimpleImputer + OneHotEncoder, sem StandardScaler) para comparação justa — apenas o algoritmo difere. Threshold otimizado via CV no treino (maximiza F1). `class_weight='balanced'` para lidar com desbalanceamento.

## Interpretação

As associações mais relevantes no M3 (com `class_weight='balanced'`) foram:

### Variáveis numéricas (Odds Ratio por 1 desvio-padrão)

- `razao_emprestimo_renda`: OR ≈ 1.878 — maior comprometimento da renda ↑ risco
- `dividas_ativas`: OR ≈ 1.578 — mais dívidas ativas ↑ risco
- `valor_solicitado`: OR ≈ 1.140 — maior valor solicitado ↑ risco
- `score_credito`: OR ≈ 0.449 — maior score ↓ risco
- `renda_mensal`: OR ≈ 0.715 — maior renda ↓ risco
- `tempo_emprego_anos`: OR ≈ 0.771 — mais tempo de emprego ↓ risco
- `prazo_meses`: OR ≈ 0.795 — maior prazo ↓ risco
- `idade`: OR ≈ 0.896 — maior idade ↓ risco (efeito moderado)

### Variáveis categóricas (Odds Ratio vs categoria de referência)

- `finalidade_Saúde`: OR ≈ 1.136 — finalidade saúde ↑ risco vs referência
- `escolaridade_Ensino Médio`: OR ≈ 1.080 — ensino médio ↑ risco vs referência
- `escolaridade_Pós-graduação`: OR ≈ 1.042 — pós-graduação ↑ risco vs referência
- `escolaridade_Ensino Superior`: OR ≈ 0.907 — ensino superior ↓ risco vs referência
- `escolaridade_Ensino Fundamental`: OR ≈ 0.867 — ensino fundamental ↓ risco vs referência
- `finalidade_Reforma`: OR ≈ 0.926 — reforma ↓ risco vs referência
- `finalidade_Veículo`: OR ≈ 0.947 — veículo ↓ risco vs referência
- `finalidade_Educação`: OR ≈ 0.952 — educação ↓ risco vs referência
- `finalidade_Viagem`: OR ≈ 0.956 — viagem ↓ risco vs referência
- `finalidade_Outros`: OR ≈ 0.975 — outros ↓ risco vs referência

> **Nota:** Odds ratios são associações condicionais do modelo, **não efeitos causais**. Variáveis numéricas são padronizadas (OR por 1 DP); categóricas comparam contra a categoria de referência omitida pelo OneHotEncoder.

## Limitações e próximos cuidados

- Dados sintéticos, sem garantia de validade externa.
- Ausência de dimensão temporal para avaliar estabilidade e drift.
- Limiar otimizado por F1 no CV não incorpora custos reais de falso positivo/negativo.
- Variáveis geográficas (`estado`) podem atuar como proxies; por isso foi comparado e não mantido no modelo selecionado.
- Uso real exige validação temporal, análise de fairness, governança, monitoramento de drift e revisão humana.