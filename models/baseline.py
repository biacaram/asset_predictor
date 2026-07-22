"""
Modelos baseline. O objetivo deles não é serem bons -- é serem o "chão"
que qualquer modelo mais sofisticado precisa superar para justificar sua
complexidade. Se o XGBoost não bate o NaiveBaseline de forma consistente
no backtest, o XGBoost não está agregando nada.

Todos os modelos seguem a mesma interface: fit(X, y) e predict(X),
igual ao padrão do scikit-learn, pra poderem ser trocados livremente
no código de backtesting.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class NaiveBaseline:
    """
    Prevê sempre retorno = 0 (ou seja, "o preço amanhã será igual ao de hoje").
    É o baseline mais básico possível em séries financeiras.
    """

    def fit(self, X, y):
        return self  # não há nada para aprender

    def predict(self, X):
        return np.zeros(len(X))


class HistoricalMeanBaseline:
    """
    Prevê sempre o retorno médio observado no conjunto de treino.
    Mais forte que o NaiveBaseline quando o ativo tem tendência
    consistente (ex: um bull market longo).
    """

    def __init__(self):
        self.mean_return_ = None

    def fit(self, X, y):
        self.mean_return_ = np.mean(y)
        return self

    def predict(self, X):
        if self.mean_return_ is None:
            raise RuntimeError("Chame fit() antes de predict()")
        return np.full(len(X), self.mean_return_)


class LinearRegressionModel:
    """
    Wrapper fino sobre LinearRegression do sklearn, só para manter a
    mesma interface fit/predict dos baselines acima.
    """

    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def coefficients(self) -> pd.Series:
        """Útil para inspecionar quais features o modelo está usando mais."""
        return pd.Series(self.model.coef_, index=self.model.feature_names_in_).sort_values(
            key=abs, ascending=False
        )
