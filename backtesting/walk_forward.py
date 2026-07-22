"""
Walk-forward validation: a forma correta de validar modelos em séries
temporais. A diferença crucial pro k-fold tradicional (que você provavelmente
já usou em outros projetos de ML) é que AQUI a ordem importa -- o modelo
nunca pode ver dados do futuro durante o treino, nem por acidente.

Erros clássicos que isso evita:
1. Split aleatório (train_test_split com shuffle=True) -- deixaria o modelo
   treinar em dados de terça-feira e "prever" segunda-feira. Parece ótimo
   no backtest e é completamente inútil na vida real.
2. Normalizar/escalar os dados ANTES de separar treino e teste -- a média e
   desvio padrão calculados incluiriam informação do futuro (do teste).
3. Escolher hiperparâmetros olhando a performance no teste final -- isso é
   "vazamento" indireto: você está otimizando pro exame que já viu a prova.

Como funciona aqui: expanding window. A cada fold, o treino cresce (sempre
começa no início dos dados) e o teste é o próximo bloco de tempo, nunca
sobreposto com o treino.

    Fold 1: [treino........][teste]
    Fold 2: [treino............][teste]
    Fold 3: [treino................][teste]
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


def walk_forward_splits(n_samples: int, n_splits: int, min_train_size: int, test_size: int):
    """
    Gera índices (train_idx, test_idx) para expanding window walk-forward.
    Os índices assumem que o DataFrame já está ORDENADO POR DATA -- se não
    estiver, ordene antes de chamar isso.
    """
    max_start = n_samples - test_size
    if min_train_size >= max_start:
        raise ValueError(
            f"Dados insuficientes: min_train_size={min_train_size} + test_size={test_size} "
            f"excede n_samples={n_samples}. Reduza os parâmetros ou use mais dados."
        )

    fold_starts = np.linspace(min_train_size, max_start, n_splits, dtype=int)

    for train_end in fold_starts:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(train_end + test_size, n_samples))
        if len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fração de vezes que o modelo acertou a DIREÇÃO do movimento (subiu/desceu),
    independente de acertar a magnitude. Em finanças isso costuma importar
    mais que o erro absoluto -- uma estratégia de trade se importa com sinal,
    não com o valor exato do retorno previsto.
    """
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


@dataclass
class FoldResult:
    fold: int
    model_name: str
    train_size: int
    test_size: int
    mae: float
    rmse: float
    directional_accuracy: float


def run_backtest(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    models: dict,  # {"nome": callable_que_retorna_novo_modelo}
    n_splits: int = 5,
    min_train_size: int = 100,
    test_size: int = 20,
) -> pd.DataFrame:
    """
    Roda walk-forward backtesting para vários modelos ao mesmo tempo, pra
    comparação justa -- todos veem exatamente os mesmos folds de treino/teste.

    `models` é um dict de nome -> factory (função sem argumentos que retorna
    uma instância NOVA do modelo), porque cada fold precisa de um modelo
    treinado do zero -- reusar a mesma instância entre folds vazaria
    informação de folds anteriores.

    Retorna um DataFrame com uma linha por (fold, modelo).
    """
    df = df.sort_values("date").reset_index(drop=True)
    X_full = df[feature_cols]
    y_full = df[target_col].values

    results = []

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_splits(len(df), n_splits, min_train_size, test_size), start=1
    ):
        X_train, y_train = X_full.iloc[train_idx], y_full[train_idx]
        X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]

        for model_name, model_factory in models.items():
            model = model_factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = float(np.mean(np.abs(y_test - preds)))
            rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
            dir_acc = directional_accuracy(y_test, preds)

            results.append(FoldResult(
                fold=fold,
                model_name=model_name,
                train_size=len(train_idx),
                test_size=len(test_idx),
                mae=mae,
                rmse=rmse,
                directional_accuracy=dir_acc,
            ))

    return pd.DataFrame([r.__dict__ for r in results])


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Agrega os folds por modelo -- média e desvio padrão das métricas."""
    return results.groupby("model_name").agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        directional_accuracy_mean=("directional_accuracy", "mean"),
        n_folds=("fold", "count"),
    ).sort_values("mae_mean")
