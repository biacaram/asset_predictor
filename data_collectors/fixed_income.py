"""
Coletor de renda fixa via API SGS (Sistema Gerenciador de Séries Temporais)
do Banco Central do Brasil. API pública, sem necessidade de chave.

Docs: https://dadosabertos.bcb.gov.br/dataset/xxxx-serie (busque o código
da série desejada em https://www3.bcb.gov.br/sgspub)

IMPORTANTE -- diferença conceitual dos outros coletores:
CDI e Selic não têm "preço" no sentido de ações/cripto -- são TAXAS diárias
(% ao dia). Pra caber no mesmo schema (date, open, high, low, close, volume)
usado pelo resto do pipeline, convertemos a taxa diária num ÍNDICE
ACUMULADO sintético, começando em 100 -- exatamente como fundos DI e ETFs
de renda fixa fazem na prática. Assim, o "close" desse coletor representa
quanto R$ 100 investidos no início do período valeriam accumulando a taxa
diária, e os indicadores técnicos (SMA, RSI etc.) fazem sentido em cima
desse índice normalmente.

open/high/low são iguais ao close (não existe variação intradiária nesse
tipo de série) e volume é sempre NaN (não se aplica).
"""

from datetime import date

import pandas as pd
import requests

from .base import BaseCollector

# Códigos das séries mais usadas no SGS do BCB.
# Busque outros códigos em https://www3.bcb.gov.br/sgspub
SERIES_CODES = {
    "CDI": 12,      # CDI diário (% a.d.)
    "SELIC": 11,     # Taxa Selic diária (% a.d.)
    "IPCA": 433,     # IPCA mensal (% a.m.) -- cuidado, essa é mensal, não diária
}


class BCBCollector(BaseCollector):
    source_name = "fixed_income_bcb"
    BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        code = SERIES_CODES.get(symbol.upper())
        if code is None:
            raise ValueError(
                f"Símbolo '{symbol}' não mapeado. Opções disponíveis: "
                f"{list(SERIES_CODES.keys())}. Adicione novos códigos em SERIES_CODES "
                f"(consulte https://www3.bcb.gov.br/sgspub)."
            )

        url = self.BASE_URL.format(code=code)
        params = {
            "formato": "json",
            "dataInicial": start.strftime("%d/%m/%Y"),
            "dataFinal": end.strftime("%d/%m/%Y"),
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        if not payload:
            raise ValueError(f"BCB não retornou dados para '{symbol}' no período {start} -> {end}")

        df = pd.DataFrame(payload)
        df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["daily_rate_pct"] = df["valor"].astype(float)

        df = df.sort_values("date").reset_index(drop=True)

        # Converte taxa diária (%) em índice acumulado sintético, começando
        # em 100 -- é isso que vira "close" pro resto do pipeline.
        df["close"] = 100 * (1 + df["daily_rate_pct"] / 100).cumprod()
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = float("nan")

        # Mantemos a taxa diária original como coluna extra -- útil de
        # inspecionar depois, mesmo não sendo uma das colunas padrão.
        return df[["date", "open", "high", "low", "close", "volume", "daily_rate_pct"]]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    collector = BCBCollector(raw_dir="data/raw")
    df = collector.fetch_and_save("CDI", start=date(2023, 1, 1), end=date.today())
    print(df.tail())
