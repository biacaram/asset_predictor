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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


class RidgeRegressionModel:
    """
    Regressão linear com regularização L2 (Ridge). O termo de regularização
    penaliza coeficientes grandes, o que ataca diretamente dois problemas
    da LinearRegressionModel pura quando há muitas features correlacionadas:

    1. Multicolinearidade -- SMA/EMA/lags de close carregam informação
       redundante, o que deixa os coeficientes da regressão comum instáveis
       (pequenas mudanças no treino geram coeficientes bem diferentes).
    2. Overfitting -- com poucas centenas de linhas de treino e dezenas de
       features, o modelo sem penalização tende a "decorar" ruído do
       período de treino em vez de aprender um padrão que generaliza.

    IMPORTANTE: Ridge penaliza o TAMANHO dos coeficientes, então só faz
    sentido se as features estiverem na mesma escala -- caso contrário a
    penalização castiga injustamente features de escala maior (ex: preço
    em milhares) frente a features de escala pequena (ex: RSI de 0 a 100).
    Por isso o StandardScaler entra ANTES do Ridge, dentro de um Pipeline.
    O scaler é ajustado (fit) só com dados de treino a cada fold -- o
    Pipeline garante isso automaticamente, sem vazar estatística do teste.

    `alpha` controla a força da regularização: quanto maior, mais os
    coeficientes são "puxados" para perto de zero. alpha=1.0 é um ponto de
    partida razoável; vale testar uma faixa (0.1, 1.0, 10.0, 100.0) depois.
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, **kwargs)),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    @property
    def coefficients(self) -> pd.Series:
        ridge = self.pipeline.named_steps["ridge"]
        return pd.Series(ridge.coef_, index=ridge.feature_names_in_).sort_values(
            key=abs, ascending=False
        )