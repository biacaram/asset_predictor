"""
Indicadores técnicos clássicos, implementados só com pandas (sem ta-lib,
que dá trabalho pra instalar em alguns sistemas). Cada função recebe uma
série de preços (geralmente `close`) e devolve uma nova série alinhada
pelo mesmo índice.
"""

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Média móvel simples."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Média móvel exponencial (dá mais peso aos preços recentes)."""
    return series.ewm(span=span, adjust=False).mean()


def returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Retorno percentual simples entre t e t-periods."""
    return series.pct_change(periods=periods)


def volatility(series: pd.Series, window: int) -> pd.Series:
    """Desvio padrão dos retornos diários numa janela -- proxy de volatilidade."""
    daily_returns = returns(series, periods=1)
    return daily_returns.rolling(window=window).std()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index. Varia de 0 a 100.
    Acima de 70 costuma indicar sobrecompra, abaixo de 30 sobrevenda
    -- mas trate isso como feature, não como regra de decisão pronta.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).
    Retorna um DataFrame com as colunas: macd_line, signal_line, macd_hist.
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line

    return pd.DataFrame({
        "macd_line": macd_line,
        "signal_line": signal_line,
        "macd_hist": hist,
    })


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger: banda central (SMA), superior e inferior."""
    mid = sma(series, window)
    std = series.rolling(window=window).std()

    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": mid + num_std * std,
        "bb_lower": mid - num_std * std,
    })
