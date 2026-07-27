"""
XGBoost para o problema de previsão de retorno. Diferente da família linear
(LinearRegression, Ridge), o XGBoost constrói árvores de decisão em sequência,
cada uma corrigindo o erro da anterior -- isso permite capturar interações
NÃO-LINEARES entre features (ex: "RSI alto E volatilidade baixa" pode ter um
efeito diferente do que a soma dos efeitos de RSI alto e volatilidade baixa
isoladamente, algo que nenhum modelo linear consegue representar).

Os hiperparâmetros default aqui já vêm com regularização razoável pensando
no tamanho pequeno do dataset (algumas centenas de linhas por fold) --
árvores rasas e poucos estimadores, pra reduzir o risco do mesmo problema
de overfitting que vimos na regressão linear com features completas.
"""

import pandas as pd
from xgboost import XGBRegressor


class XGBoostModel:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,          # raso de propósito -- árvores profundas decoram ruído fácil com pouco dado
            learning_rate=learning_rate,   # baixo, compensado pelo n_estimators -- aprendizado mais estável
            subsample=subsample,            # cada árvore vê só 80% das linhas -- reduz variância
            colsample_bytree=colsample_bytree,  # cada árvore vê só 80% das features -- idem
            reg_alpha=reg_alpha,            # regularização L1
            reg_lambda=reg_lambda,          # regularização L2
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importance(self) -> pd.Series:
        """
        Quais features o modelo mais usou para dividir as árvores.
        Diferente dos coeficientes da regressão linear, isso não tem sinal
        (positivo/negativo) -- só indica relevância, não direção do efeito.
        """
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_,
        ).sort_values(ascending=False)
