# Asset Predictor

Pipeline de ML para avaliar, com validação metodologicamente honesta, se é
possível prever o retorno diário de ativos financeiros a partir de
indicadores técnicos — testado em três classes de ativo (cripto, ações,
renda fixa).

**Achados completos em [`FINDINGS.md`](FINDINGS.md).** Resumo: nenhum
modelo testado bate uma baseline simples de forma confiável em ativos de
risco (BTC, ações); renda fixa é trivialmente previsível por baixa
variância, não por capacidade preditiva real dos modelos.

## Estrutura

```
asset_predictor/
├── data_collectors/     # coleta de dados (interface comum p/ qualquer fonte)
│   ├── base.py            # classe abstrata: retry, validação, merge idempotente
│   ├── crypto.py           # CoinGecko
│   ├── equities.py         # Yahoo Finance (yfinance)
│   └── fixed_income.py     # Banco Central (SGS)
├── features/              # transforma preço bruto em variáveis de modelo
│   ├── technical_indicators.py  # SMA, EMA, RSI, MACD, Bollinger Bands
│   └── build_features.py         # monta a tabela final: indicadores + lags + target
├── models/                # modelos e scripts de comparação
│   ├── baseline.py          # Naive, HistoricalMean, LinearRegression, Ridge
│   ├── xgboost_model.py      # XGBoost
│   ├── compare_final.py       # roda todos os modelos lado a lado
│   ├── compare_ridge.py        # full vs reduced features
│   └── alpha_sweep.py           # varredura de regularização do Ridge
├── backtesting/            # validação walk-forward (sem vazamento temporal)
│   └── walk_forward.py
├── dashboard/               # visualização interativa (Streamlit)
│   └── app.py
├── scripts/
│   └── fetch_data.py          # coleta genérica via linha de comando, qualquer fonte/símbolo
├── data/
│   ├── raw/                    # dados brutos por fonte (fora do git)
│   └── processed/                # features prontas para modelo (fora do git)
├── test_base.py, test_features.py, test_backtesting.py  # testes sem dependência de rede
├── FINDINGS.md               # relatório dos achados
├── requirements.txt
└── .github/workflows/tests.yml  # CI: roda os testes a cada push
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso — do zero até o dashboard

```bash
# 1. Coletar dados (qualquer fonte/símbolo)
python -m scripts.fetch_data --source crypto --symbol BTC --start 2023-01-01
python -m scripts.fetch_data --source equities --symbol AAPL --start 2023-01-01
python -m scripts.fetch_data --source fixed_income --symbol CDI --start 2023-01-01

# 2. Construir features
python -m features.build_features --input data/raw/crypto_coingecko/BTC.parquet --symbol BTC

# 3. Comparar modelos com backtest walk-forward
python -m models.compare_final --input data/processed/BTC.parquet

# 4. Ver tudo no dashboard interativo
streamlit run dashboard/app.py
```

## Rodar os testes

```bash
python test_base.py         # interface de coleta (dados fake, sem rede)
python test_features.py     # indicadores e target (dados sintéticos)
python test_backtesting.py  # walk-forward, checagem de vazamento temporal
```

Esses três testes rodam sem depender de nenhuma API externa — são os que
o CI (`.github/workflows/tests.yml`) executa a cada push.

## Próximos passos possíveis
Ver seção "Limitações e próximos passos" em [`FINDINGS.md`](FINDINGS.md).