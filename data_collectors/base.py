"""
Interface base para coletores de dados de ativos.

Todo coletor (cripto, ações, renda fixa) deve herdar de BaseCollector
e implementar `_fetch_raw`. Isso garante que o resto do pipeline sempre
recebe um DataFrame no mesmo formato, não importa a fonte.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from pathlib import Path
import time
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Colunas padrão que todo coletor deve retornar.
# Ativos sem volume (ex: alguns títulos de renda fixa) devem preencher com NaN.
STANDARD_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


class BaseCollector(ABC):
    """Classe abstrata para coleta de dados de preços."""

    source_name: str = "base"

    def __init__(self, raw_dir: str = "data/raw", max_retries: int = 3, backoff_seconds: float = 2.0):
        self.raw_dir = Path(raw_dir) / self.source_name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    @abstractmethod
    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Implementado por cada subclasse. Deve retornar um DataFrame
        com (no mínimo) as colunas de STANDARD_COLUMNS, podendo conter
        colunas extras específicas da fonte.
        """
        raise NotImplementedError

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Busca dados com retry + backoff exponencial. Ponto único de entrada
        usado pelo resto do pipeline — não chame _fetch_raw diretamente.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = self._fetch_raw(symbol, start, end)
                df = self._validate_and_normalize(df, symbol)
                logger.info(f"[{self.source_name}] {symbol}: {len(df)} linhas ({start} -> {end})")
                return df
            except Exception as e:
                last_error = e
                wait = self.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    f"[{self.source_name}] {symbol}: tentativa {attempt}/{self.max_retries} "
                    f"falhou ({e}). Aguardando {wait:.1f}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise RuntimeError(f"[{self.source_name}] Falha ao buscar {symbol} após {self.max_retries} tentativas") from last_error

    def _validate_and_normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Garante schema consistente antes de devolver o DataFrame."""
        missing = set(["date", "open", "high", "low", "close"]) - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias faltando: {missing}")

        df = df.copy()
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"])
        if "volume" not in df.columns:
            df["volume"] = float("nan")

        extra_cols = [c for c in df.columns if c not in STANDARD_COLUMNS]
        df = df[STANDARD_COLUMNS + extra_cols]
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        return df.reset_index(drop=True)

    def fetch_and_save(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Busca dados novos e faz merge idempotente com o que já existe em disco.
        Rodar isso duas vezes no mesmo dia não duplica nada.
        """
        path = self.raw_dir / f"{symbol.replace('/', '-')}.parquet"

        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=STANDARD_COLUMNS)
        fetch_start = existing["date"].max().date() if not existing.empty else start

        new_data = self.fetch(symbol, fetch_start, end)

        if not existing.empty:
            combined = pd.concat([existing, new_data]).drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
        else:
            combined = new_data

        combined.to_parquet(path, index=False)
        logger.info(f"[{self.source_name}] {symbol}: salvo em {path} ({len(combined)} linhas totais)")
        return combined
