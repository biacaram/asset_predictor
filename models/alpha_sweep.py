"""
Varredura de alpha para o Ridge. No teste anterior, alpha=1 e alpha=10
mostraram um padrão intrigante: MAE melhora com mais regularização, mas
directional accuracy piora. Este script testa uma faixa mais ampla e fina
de alpha pra ver se existe um "ponto ótimo" onde os dois melhoram juntos,
ou se essa troca (trade-off) é inerente ao problema com esse conjunto de
features.

Uso:
    python -m models.alpha_sweep --input data/processed/BTC.parquet
    python -m models.alpha_sweep --input data/processed/BTC.parquet --feature-set reduced
"""

import argparse

import pandas as pd

from features.build_features import get_model_feature_columns
from models.baseline import RidgeRegressionModel, LinearRegressionModel, HistoricalMeanBaseline
from backtesting.walk_forward import run_backtest

# Faixa ampla, log-espaçada -- cobre desde quase nenhuma regularização
# até regularização bem forte, pra ver o comportamento nos dois extremos.
ALPHA_VALUES = [0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]


def sweep_alpha(
    df: pd.DataFrame,
    feature_cols: list,
    alphas: list = None,
    n_splits: int = 5,
    min_train_size: int = 100,
    test_size: int = 20,
) -> pd.DataFrame:
    alphas = alphas or ALPHA_VALUES

    models = {f"ridge_alpha_{a}": (lambda a=a: RidgeRegressionModel(alpha=a)) for a in alphas}
    # Referências pra comparação, sempre no mesmo backtest pra ser justo
    models["linear_regression"] = lambda: LinearRegressionModel()
    models["historical_mean"] = lambda: HistoricalMeanBaseline()

    results = run_backtest(
        df, feature_cols=feature_cols, target_col="target_return", models=models,
        n_splits=n_splits, min_train_size=min_train_size, test_size=test_size,
    )

    summary = results.groupby("model_name").agg(
        mae_mean=("mae", "mean"),
        directional_accuracy_mean=("directional_accuracy", "mean"),
    ).reset_index()

    # Extrai o valor de alpha pra poder ordenar numericamente na tabela final
    def extract_alpha(name):
        if name.startswith("ridge_alpha_"):
            return float(name.replace("ridge_alpha_", ""))
        return None

    summary["alpha"] = summary["model_name"].apply(extract_alpha)
    ridge_rows = summary[summary["alpha"].notna()].sort_values("alpha")
    other_rows = summary[summary["alpha"].isna()]

    return pd.concat([ridge_rows, other_rows], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--feature-set", choices=["full", "reduced"], default="reduced")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    all_cols = get_model_feature_columns(df, reduced=False)
    feature_cols = get_model_feature_columns(df, reduced=True) if args.feature_set == "reduced" else all_cols

    print(f"Feature set: {args.feature_set} ({len(feature_cols)} colunas)\n")

    summary = sweep_alpha(
        df, feature_cols,
        n_splits=args.n_splits, min_train_size=args.min_train_size, test_size=args.test_size,
    )

    cols_to_show = ["model_name", "alpha", "mae_mean", "directional_accuracy_mean"]
    print(summary[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()