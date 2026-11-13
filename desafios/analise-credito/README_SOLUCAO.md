# Solução — Análise de Crédito
![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-000000?style=for-the-badge)

## Objetivo

Investigar fatores associados à inadimplência e avaliar uma **Regressão Logística** para apoiar a identificação de propostas de maior risco.

Os dados são sintéticos e educacionais. O modelo não deve ser usado em decisões reais sem validação adicional.

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
- padronização das numéricas;
- one-hot encoding das categóricas;
- Regressão Logística.
- Matplotlib e Seaborn foram utilizados para a Análise Exploratória e para a visualização dos resultados.
- Não foi realizada otimização de hiperparâmetros: a Regressão Logística utilizou o valor padrão `C=1.0`; a comparação foi feita entre conjuntos de atributos e não por busca de `C`.


Comparações:

1. baseline que sempre prevê a classe majoritária;
2. `M1`: variáveis originais;
3. `M2`: variáveis originais + razão empréstimo/renda;
4. `M3`: variáveis originais sem `estado` + razão empréstimo/renda.

A seleção é feita pela PR-AUC média em validação cruzada estratificada de 5 partes no treino. O conjunto de teste é reservado para a estimativa final.

## Resultados no teste

| Modelo | Accuracy | ROC-AUC | PR-AUC | Recall | Precisão | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0,656 | 0,500 | 0,344 | 0,000 | 0,000 | 0,000 | 0,500 |
| M1 — originais | 0,727 | 0,760 | 0,633 | 0,449 | 0,650 | 0,531 | 0,661 |
| M2 — originais + razão | 0,732 | 0,766 | 0,643 | 0,458 | 0,660 | 0,540 | 0,667 |
| M3 — sem estado + razão | 0,730 | 0,771 | 0,648 | 0,449 | 0,658 | 0,534 | 0,663 |

O modelo selecionado pela PR-AUC média na validação cruzada foi o **M3**, com PR-AUC média de 0,654 e ROC-AUC média de 0,777. No teste, ele alcançou PR-AUC de 0,648 e ROC-AUC de 0,771.
Em termos práticos, um recall de 0,449 significa que o modelo M3 captura quase metade dos reais inadimplentes. Uma precisão de 0,658 indica que, de todas as propostas alertadas como risco, o modelo acerta em cerca de 66% das vezes, reduzindo o risco de bloquear excessivamente bons pagadores. A decisão final ainda deve considerar o custo financeiro de falsos positivos e falsos negativos.
## Abordagem Champion vs. Challenger

Para refletir cenários reais da indústria de crédito, este projeto adota uma abordagem de dois modelos:

**Modelo Interpretável (Regressão Logística - M3):** O foco principal do projeto. Escolhido por permitir a extração de Odds Ratios, atendendo a rigorosos requisitos regulatórios e de governança (garantindo a explicabilidade nas recusas de crédito).

**Modelo de Alta Performance (LightGBM + Threshold Tuning):** Apresentado no final do notebook como modelo Challenger (prova de conceito). Utiliza imputação avançada (KNNImputer) e otimização de limiar para lidar com o desbalanceamento. Este modelo sacrifica a explicabilidade linear direta em troca da maximização da métrica PR-AUC e do Recall (captura de inadimplentes).

## Interpretação

As associações mais relevantes no M3 foram:

- `razao_emprestimo_renda`: odds ratio aproximado de 1,716 por desvio-padrão;
- `dividas_ativas`: odds ratio aproximado de 1,568 por desvio-padrão;
- `score_credito`: odds ratio aproximado de 0,454 por desvio-padrão;
- `renda_mensal`: odds ratio aproximado de 0,684 por desvio-padrão;
- `tempo_emprego_anos`: odds ratio aproximado de 0,776 por desvio-padrão.

Odds ratios são associações condicionais do modelo, não efeitos causais.

## Limitações e próximos cuidados

- Dados sintéticos, sem garantia de validade externa.
- Ausência de dimensão temporal para avaliar estabilidade e drift.
- Limiar 0,5 não incorpora custos reais de falso positivo e falso negativo.
- Variáveis geográficas podem atuar como proxies; por isso `estado` foi comparado e não foi mantido no modelo selecionado.
- Uso real exige validação temporal, fairness, governança, monitoramento e revisão humana.
