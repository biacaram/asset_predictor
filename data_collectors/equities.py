"""
Coletor de ações via yfinance (biblioteca que raspa dados do Yahoo Finance,
sem necessidade de API key).

Funciona com tickers de qualquer bolsa que o Yahoo Finance cobre -- para
ações brasileiras, use o sufixo ".SA" (ex: "PETR4.SA", "VALE3.SA").
Para ações americanas, o ticker puro já funciona (ex: "AAPL", "MSFT").
"""

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .base import BaseCollector


class YFinanceCollector(BaseCollector):
    source_name = "equities_yfinance"

    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        # yfinance trata o parâmetro `end` como exclusivo -- soma um dia
        # pra garantir que o último dia pedido seja incluído.
        hist = ticker.history(start=start, end=end + timedelta(days=1))

        if hist.empty:
            raise ValueError(
                f"yfinance não retornou dados para '{symbol}' no período {start} -> {end}. "
                f"Confira se o ticker está correto (ações brasileiras usam sufixo '.SA')."
            )

        hist = hist.reset_index()
        hist = hist.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # yfinance retorna datas com timezone (ex: America/New_York) -- remove
        # pra ficar consistente com os outros coletores, que usam datas naive.
        hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)

        return hist[["date", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    collector = YFinanceCollector(raw_dir="data/raw")
    df = collector.fetch_and_save("AAPL", start=date(2023, 1, 1), end=date.today())
    print(df.tail())
