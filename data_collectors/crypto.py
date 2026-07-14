"""
Coletor de cripto via CoinGecko (API pública, sem necessidade de key para uso básico).

Docs: https://www.coingecko.com/en/api/documentation
Endpoint usado: /coins/{id}/market_chart/range (histórico de preço + volume)
"""

from datetime import date, datetime
import requests
import pandas as pd

from .base import BaseCollector

# CoinGecko usa "ids" próprios, não tickers. Mapeamento dos mais comuns.
SYMBOL_TO_COINGECKO_ID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
}


class CoinGeckoCollector(BaseCollector):
    source_name = "crypto_coingecko"
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, vs_currency: str = "usd", **kwargs):
        super().__init__(**kwargs)
        self.vs_currency = vs_currency

    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        coin_id = SYMBOL_TO_COINGECKO_ID.get(symbol.upper())
        if coin_id is None:
            raise ValueError(
                f"Símbolo '{symbol}' não mapeado. Adicione em SYMBOL_TO_COINGECKO_ID "
                f"(veja o id correto em https://api.coingecko.com/api/v3/coins/list)."
            )

        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": self.vs_currency,
            "from": int(datetime.combine(start, datetime.min.time()).timestamp()),
            "to": int(datetime.combine(end, datetime.min.time()).timestamp()),
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        # CoinGecko retorna preço e volume separados, cada um como [timestamp, valor].
        # Não há open/high/low nesse endpoint gratuito -- para OHLC completo seria
        # necessário o endpoint /coins/{id}/ohlc (granularidade mais limitada).
        prices = pd.DataFrame(payload["total_volumes"], columns=["ts", "volume"])
        price_df = pd.DataFrame(payload["prices"], columns=["ts", "close"])
        df = price_df.merge(prices, on="ts")

        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        # Endpoint não traz OHLC real; aproxima open/high/low pelo close.
        # Troque pelo endpoint /ohlc se precisar de granularidade intradiária de verdade.
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]

        return df[["date", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    collector = CoinGeckoCollector(raw_dir="data/raw")
    df = collector.fetch_and_save("BTC", start=date(2026, 1, 1), end=date.today())
    print(df.tail())
