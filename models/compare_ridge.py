"""
Diagnóstico: a regressão linear simples perdeu para os baselines no
backtest anterior, com directional accuracy abaixo de 50%. Hipótese:
overfitting causado por features redundantes/correlacionadas.

Este script testa a hipótese rodando 4 variações lado a lado:
  1. LinearRegression, features completas   (o que já rodamos)
  2. LinearRegression, features reduzidas
  3. Ridge (alpha=1.0),  features completas
  4. Ridge (alpha=1.0),  features reduzidas

Se a hipótese estiver certa, esperamos ver directional_accuracy subir
(idealmente > 50%) e mae_mean cair nas versões com Ridge e/ou features
reduzidas, em relação à LinearRegression com features completas.

Uso:
    python -m models.compare_ridge --input data/processed/BTC.parquet
"""

import argparse

import pandas as pd

from features.build_features import get_reduced_feature_set, get_model_feature_columns
from models.baseline import NaiveBaseline, HistoricalMeanBaseline, LinearRegressionModel, RidgeRegressionModel
from backtesting.walk_forward import run_backtest, summarize_backtest


def run_comparison(df: pd.DataFrame, n_splits=5, min_train_size=100, test_size=20) -> pd.DataFrame:
    all_cols = get_model_feature_columns(df, reduced=False)
    reduced_cols = get_model_feature_columns(df, reduced=True)

    print(f"Features completas: {len(all_cols)} colunas")
    print(f"Features reduzidas: {len(reduced_cols)} colunas "
          f"(removidas: {sorted(set(all_cols) - set(reduced_cols))})\n")

    configs = [
        ("full_features", all_cols),
        ("reduced_features", reduced_cols),
    ]

    all_results = []
    for feature_set_name, cols in configs:
        models = {
            "naive": lambda: NaiveBaseline(),
            "historical_mean": lambda: HistoricalMeanBaseline(),
            "linear_regression": lambda: LinearRegressionModel(),
            "ridge_alpha_1": lambda: RidgeRegressionModel(alpha=1.0),
            "ridge_alpha_10": lambda: RidgeRegressionModel(alpha=10.0),
        }
        results = run_backtest(
            df, feature_cols=cols, target_col="target_return", models=models,
            n_splits=n_splits, min_train_size=min_train_size, test_size=test_size,
        )
        results["feature_set"] = feature_set_name
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet processado (saída do build_features)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    results = run_comparison(df, args.n_splits, args.min_train_size, args.test_size)

    summary = results.groupby(["feature_set", "model_name"]).agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        directional_accuracy_mean=("directional_accuracy", "mean"),
    ).sort_values(["feature_set", "mae_mean"])

    print(summary.to_string())


if __name__ == "__main__":
    main()