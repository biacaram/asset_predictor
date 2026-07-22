"""
Teste end-to-end: gera preço sintético, cria features, roda walk-forward
backtesting comparando NaiveBaseline, HistoricalMeanBaseline e LinearRegression.

Isso NÃO é um teste de "o modelo precisa ser bom" -- é um teste de que o
PIPELINE funciona sem erro e sem vazamento de dados óbvio.
"""
import numpy as np
import pandas as pd

from features.build_features import add_technical_indicators, add_lag_features, add_target
from models.baseline import NaiveBaseline, HistoricalMeanBaseline, LinearRegressionModel
from backtesting.walk_forward import run_backtest, summarize_backtest, walk_forward_splits


def make_fake_prices(n=400, seed=42):
    dates = pd.date_range("2024-01-01", periods=n)
    rng = np.random.default_rng(seed)
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


def test_no_leakage_in_splits():
    """Garante que treino nunca contém índices >= ao menor índice do teste."""
    for train_idx, test_idx in walk_forward_splits(n_samples=300, n_splits=5, min_train_size=100, test_size=20):
        assert train_idx.max() < test_idx.min(), "VAZAMENTO: treino contém índice futuro em relação ao teste"
    print("OK: nenhum vazamento temporal encontrado nos splits")


if __name__ == "__main__":
    test_no_leakage_in_splits()

    df = make_fake_prices()
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_target(df, horizon=1)
    df = df.dropna().reset_index(drop=True)
    print(f"OK: features construídas -> {len(df)} linhas, {len(df.columns)} colunas")

    feature_cols = [c for c in df.columns if c not in ["date", "symbol", "target_return"]]

    models = {
        "naive": lambda: NaiveBaseline(),
        "historical_mean": lambda: HistoricalMeanBaseline(),
        "linear_regression": lambda: LinearRegressionModel(),
    }

    results = run_backtest(
        df,
        feature_cols=feature_cols,
        target_col="target_return",
        models=models,
        n_splits=5,
        min_train_size=150,
        test_size=20,
    )

    assert len(results) == 5 * len(models), "número de resultados inesperado"
    print(f"OK: backtest rodou -> {len(results)} linhas de resultado ({5} folds x {len(models)} modelos)")

    summary = summarize_backtest(results)
    print("\nResumo do backtest (dados sintéticos, não espere nada 'bom' aqui):")
    print(summary)

    print("\nTodos os testes passaram.")
