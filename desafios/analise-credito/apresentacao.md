# Roteiro de apresentação — Análise de Crédito

## Slide 1 — Problema e objetivo

- CredAIR possui histórico de propostas de empréstimo.
- Objetivo: identificar propostas com maior risco de inadimplência.
- Resposta: `inadimplente`, classificação binária.

## Slide 2 — Dados disponíveis

- 5.000 propostas brutas.
- Perfil do solicitante, renda, score, pedido, prazo, finalidade e dívidas.
- Dados sintéticos e educacionais.

## Slide 3 — Qualidade dos dados

- 19 duplicatas exatas adicionais.
- 15 idades codificadas dez vezes acima do plausível.
- 7 tempos de emprego acima do domínio observado.
- Estados com espaços extras: 85 representações reduzidas para 27 estados.
- Não havia ausências no arquivo bruto.

## Slide 4 — Tratamento e atributo derivado

- Padronização textual.
- Correção das idades identificadas.
- Tempos inválidos marcados como ausentes e imputados somente no treino.
- Remoção de duplicatas.
- Criação de `razao_emprestimo_renda`.

## Slide 5 — Evidências exploratórias

- Inadimplência no quartil inferior de score: 54,4%; no superior: 16,5%.
- Quartil inferior de renda: 50,8%; superior: 22,0%.
- Mais de três dívidas ativas: 48,2%.
- Maior quartil da razão empréstimo/renda: 54,6%.

## Slide 6 — Preparação e comparação

- Divisão estratificada 80/20.
- Imputação, escala e one-hot dentro de pipeline.
- Baseline, M1, M2 e M3.
- Seleção pela PR-AUC média em validação cruzada.

## Slide 7 — Resultados

- Baseline: ROC-AUC 0,500 e PR-AUC 0,344.
- M3 selecionado: ROC-AUC 0,771 e PR-AUC 0,648 no teste.
- Recall de inadimplentes: 0,449.
- Precisão: 0,658.

## Slide 8 — Interpretação

- Maior razão empréstimo/renda e mais dívidas elevam a chance estimada.
- Maior score, renda e tempo de emprego reduzem a chance estimada.
- `estado` foi retirado do modelo final por simplicidade e cautela com proxy geográfico.

## Slide 9 — Limitações

- Dados sintéticos.
- Sem histórico temporal.
- Limiar não calibrado por custo de negócio.
- Associação não implica causalidade.

## Slide 10 — Recomendação

- Usar o modelo como apoio, não decisão automática.
- Definir custos de falso positivo e falso negativo.
- Validar temporalmente, monitorar drift e avaliar fairness antes de produção.
