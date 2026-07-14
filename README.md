# Asset predictor — coleta de dados

## Estrutura
```
data_collectors/
├── base.py     # interface abstrata (retry, validação de schema, merge idempotente)
├── crypto.py   # coletor CoinGecko (BTC, ETH, SOL, BNB por enquanto)
data/raw/       # dados brutos em parquet, nunca sobrescritos manualmente
test_base.py    # teste da interface com dados fake (sem rede)
```

## Como usar

```bash
pip install -r requirements.txt
python -m data_collectors.crypto
```

Isso busca o histórico de BTC desde 01/01/2026 até hoje e salva em
`data/raw/crypto_coingecko/BTC.parquet`. Rodar de novo não duplica nada —
o `fetch_and_save` retoma a partir da última data já salva.

## Próximos passos sugeridos

1. **`equities.py`** — mesmo padrão, usando `yfinance`. Ações têm calendário
   de pregão (sem fim de semana), então cuidado ao alinhar com cripto depois.
2. **`fixed_income.py`** — API do Banco Central (SGS) ou Tesouro Direto.
   Não terá `volume` real — deixe como `NaN`, o `base.py` já trata isso.
3. Endpoint `/ohlc` da CoinGecko para granularidade melhor (o `market_chart/range`
   usado aqui só traz `close`, então open/high/low são aproximados).
4. Um `scripts/run_daily.py` que itera sobre um `config.yaml` com todos os
   símbolos e chama `fetch_and_save` para cada um — vira seu job de cron.

## Testar sem rede
```bash
python test_base.py
```
Valida a interface (schema, idempotência) com um coletor fake, sem bater na API.
