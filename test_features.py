"""Teste manual do pipeline de features com preço sintético (sem depender de dado real)."""
import numpy as np
import pandas as pd

from features.build_features import add_technical_indicators, add_lag_features, add_target


def make_fake_prices(n=200):
    dates = pd.date_range("2025-01-01", periods=n)
    rng = np.random.default_rng(42)
    # random walk simples pra simular preço
    returns_series = rng.normal(0, 0.02, n)
    close = 100 * (1 + returns_series).cumprod()
    return pd.DataFrame({
        "date": dates,
        "symbol": "FAKE",
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1000, 5000, n),
    })


if __name__ == "__main__":
    df = make_fake_prices()

    df = add_technical_indicators(df)
    expected_indicator_cols = ["sma_7", "sma_21", "ema_12", "ema_26", "rsi_14",
                                "macd_line", "bb_upper", "bb_position"]
    for col in expected_indicator_cols:
        assert col in df.columns, f"coluna faltando: {col}"
    print(f"OK: indicadores técnicos calculados ({len(expected_indicator_cols)} colunas verificadas)")

    df = add_lag_features(df)
    assert "close_lag_7" in df.columns and "return_lag_1" in df.columns
    print("OK: lag features calculadas")

    df = add_target(df, horizon=1)
    assert "target_return" in df.columns
    # checagem manual: target da linha 0 deve bater com o retorno realizado
    manual_target = df["close"].iloc[1] / df["close"].iloc[0] - 1
    assert abs(df["target_return"].iloc[0] - manual_target) < 1e-9
    print("OK: target calculado e validado manualmente")

    clean = df.dropna()
    print(f"\nLinhas totais: {len(df)}, linhas limpas (sem NaN): {len(clean)}")
    print(f"Colunas finais: {len(df.columns)}")
    print("\nAmostra:")
    print(clean[["date", "close", "sma_7", "rsi_14", "target_return"]].head(3))

    print("\nTodos os testes passaram.")
