# Achados do projeto — previsão de retorno diário

## Objetivo
Avaliar se é possível prever o retorno diário (`t+1`) de ativos financeiros
usando indicadores técnicos clássicos (SMA, EMA, RSI, MACD, Bollinger Bands)
e features de lag, comparando modelos lineares e não-lineares contra
baselines simples, com validação walk-forward (sem vazamento temporal).

## Metodologia
- **Validação:** walk-forward com expanding window, 5 folds. Treino sempre
  começa no início dos dados e cresce; teste é sempre o próximo bloco de
  tempo, nunca sobreposto com o treino.
- **Métricas:** MAE, RMSE (magnitude do erro) e directional accuracy
  (acerto do sinal do movimento — mais relevante que magnitude pra decisão
  de compra/venda).
- **Baselines:** `naive` (sempre prevê retorno zero) e `historical_mean`
  (sempre prevê a média histórica de retorno do treino).
- **Modelos testados:** regressão linear, Ridge (com StandardScaler e
  varredura de 13 valores de alpha), XGBoost.
- **Ativos:** BTC (CoinGecko) e AAPL (Yahoo Finance via yfinance).

## Resultados

### BTC
| Modelo | MAE | Directional Accuracy |
|---|---|---|
| linear_regression (reduced features) | 0.01435 | **0.57** |
| historical_mean | 0.01436 | 0.60 |
| naive | 0.01445 | 0.00 |
| ridge (melhor alpha, 1000) | 0.01461 | 0.50 |
| xgboost | 0.01792 | 0.49 |

### AAPL (2023–2026, ~850 linhas de treino no último fold)
| Modelo | MAE | Directional Accuracy |
|---|---|---|
| ridge_alpha_1000 | 0.01156 | 0.50 |
| historical_mean | 0.01159 | 0.52 |
| naive | 0.01160 | 0.00 |
| linear_regression | 0.01169 | 0.44 |
| xgboost | 0.01194 | 0.51 |

### CDI (2023–2026, ~850 linhas de treino no último fold)
| Modelo | MAE | Directional Accuracy |
|---|---|---|
| xgboost | ~0.00005 | **1.00** |
| historical_mean | ~0.00049 | **1.00** |
| ridge_alpha_1000 | ~0.00049 | **1.00** |
| linear_regression | ~0.00049 | **1.00** |
| naive | 0.00049 | 0.00 |

## Interpretação

**Nenhum modelo superou o `historical_mean` de forma clara e consistente
em nenhum dos ativos de risco (BTC e AAPL).** A regressão linear teve seu
melhor resultado no BTC (0.57), mas o mesmo modelo teve o PIOR resultado
na AAPL (0.44, abaixo de acaso) — essa inversão entre ativos é evidência
forte de que o resultado do BTC provavelmente era ruído estatístico
específico daquele período, não um padrão real e generalizável.

O XGBoost, apesar de capaz de capturar relações não-lineares que os
modelos lineares não alcançam, também não superou os baselines em nenhum
dos dois ativos de risco, e mostrou alta variância entre folds
(directional accuracy oscilando entre 0.40 e 0.65 nos mesmos 5 folds da
AAPL).

**O CDI conta uma história completamente diferente, e isso é o achado
mais importante do projeto.** Todos os modelos (exceto o `naive`)
atingiram 100% de directional accuracy, com MAE praticamente zero. Isso
não indica que os modelos "aprenderam" algo sofisticado -- indica que uma
taxa de juros diária tem variação tão pequena de um dia para o outro que
qualquer modelo que capture "amanhã ≈ hoje" acerta quase sempre. O
`naive` (que sempre prevê retorno zero) é o único que falha, porque é o
único que ignora completamente a tendência de alta constante.

**A conclusão central do projeto, então, não é sobre qual modelo é
melhor -- é sobre como a previsibilidade varia por CLASSE de ativo.**
Ativos de risco (ações, cripto) têm retornos diários que se comportam de
forma muito mais próxima de ruído puro do que taxas de juros, que têm
comportamento quase determinístico no curto prazo. Isso é consistente com
a própria definição de "risco" em finanças: renda fixa remunera pouco
risco, ativos de risco remuneram (na teoria) o fato de serem mais difíceis
de prever.

## Conclusão

Com o conjunto de features testado (indicadores técnicos + lags) e o
horizonte de previsão testado (1 dia): **para ativos de risco (BTC,
AAPL), não há evidência de sinal preditivo confiável** que supere uma
baseline ingênua de forma consistente. **Para renda fixa (CDI), a
previsibilidade é alta, mas por um motivo trivial** (baixa variância
inerente à série, não capacidade preditiva real do modelo).

Isso NÃO significa que o projeto falhou — significa que a pergunta
original ("dá pra prever o preço de amanhã com esses dados?") tem uma
resposta honesta, bem fundamentada, e diferente por classe de ativo. Esse
tipo de resultado nuançado, obtido com validação metodologicamente
correta em 3 classes de ativo diferentes, é mais valioso para um
portfólio técnico do que um resultado "positivo" único obtido com
vazamento de dados ou overfitting não detectado — o que é infelizmente
comum em projetos de ML financeiro mal validados.

## Limitações e próximos passos possíveis
- Testado só em 2 ativos e um horizonte (1 dia) — resultado pode não se
  generalizar para horizontes mais longos (semanal/mensal) ou outras
  classes de ativo (renda fixa).
- Features usadas são só técnicas (preço/volume) — não incluem dados
  fundamentalistas, sentimento de notícias, ou dados on-chain (no caso
  de cripto), que poderiam carregar sinal diferente.
- O tamanho de amostra, mesmo no AAPL com ~850 linhas no último fold,
  ainda é pequeno para os padrões de modelos como XGBoost.