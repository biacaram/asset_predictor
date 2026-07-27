"""
Comparação final entre todos os modelos testados no projeto até agora,
usando reduced_features (que já provou ser melhor que o conjunto completo
nos testes anteriores com a família linear).

Uso:
    python -m models.compare_final --input data/processed/BTC.parquet
"""

import argparse

import pandas as pd

from features.build_features import get_model_feature_columns
from models.baseline import NaiveBaseline, HistoricalMeanBaseline, LinearRegressionModel, RidgeRegressionModel
from models.xgboost_model import XGBoostModel
from backtesting.walk_forward import run_backtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    feature_cols = get_model_feature_columns(df)
    print(f"Usando reduced_features: {len(feature_cols)} colunas\n")

    models = {
        "naive": lambda: NaiveBaseline(),
        "historical_mean": lambda: HistoricalMeanBaseline(),
        "linear_regression": lambda: LinearRegressionModel(),
        "ridge_alpha_1000": lambda: RidgeRegressionModel(alpha=1000),
        "xgboost": lambda: XGBoostModel(),
    }

    results = run_backtest(
        df, feature_cols=feature_cols, target_col="target_return", models=models,
        n_splits=args.n_splits, min_train_size=args.min_train_size, test_size=args.test_size,
    )

    summary = results.groupby("model_name").agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        directional_accuracy_mean=("directional_accuracy", "mean"),
    ).sort_values("mae_mean")

    print(summary.to_string())

    # Detalhe por fold para o xgboost -- útil pra ver se ele é consistente
    # entre folds ou se só "acertou" um período específico por sorte.
    print("\nDetalhe do XGBoost por fold:")
    print(results[results["model_name"] == "xgboost"][
        ["fold", "train_size", "test_size", "mae", "directional_accuracy"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()