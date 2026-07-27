"""
Constrói a tabela de features a partir de um parquet bruto (o que os
coletores salvam em data/raw/) e salva o resultado em data/processed/.

Uso:
    python -m features.build_features --input data/raw/crypto_coingecko/BTC.parquet --symbol BTC
"""

import argparse
from pathlib import Path

import pandas as pd

from features.technical_indicators import sma, ema, returns, volatility, rsi, macd, bollinger_bands

# Quantos dias no passado usar como lag features.
LAG_PERIODS = [1, 2, 3, 7, 14]

# Quantos dias à frente prever. target = retorno de close entre hoje e t+HORIZON.
HORIZON = 1


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["sma_7"] = sma(close, 7)
    df["sma_21"] = sma(close, 21)
    df["ema_12"] = ema(close, 12)
    df["ema_26"] = ema(close, 26)
    df["volatility_7"] = volatility(close, 7)
    df["volatility_21"] = volatility(close, 21)
    df["rsi_14"] = rsi(close, 14)

    macd_df = macd(close)
    df = pd.concat([df, macd_df], axis=1)

    bb_df = bollinger_bands(close)
    df = pd.concat([df, bb_df], axis=1)

    # Distância do preço até as bandas -- geralmente mais informativo pro
    # modelo do que os valores absolutos das bandas.
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    return df


def add_lag_features(df: pd.DataFrame, column: str = "close", lags: list = None) -> pd.DataFrame:
    df = df.copy()
    lags = lags or LAG_PERIODS
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        df[f"return_lag_{lag}"] = returns(df[column], periods=lag)
    return df


def add_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Define o alvo como o retorno percentual entre t e t+horizon.
    Usar retorno em vez de preço absoluto é quase sempre melhor:
    o modelo aprende um padrão relativo, não fica preso à escala de preço.
    """
    df = df.copy()
    df["target_return"] = df["close"].shift(-horizon) / df["close"] - 1
    return df


def get_reduced_feature_set(all_columns: list) -> list:
    """
    Remove features redundantes do conjunto completo -- pensado para testar
    a hipótese de que multicolinearidade está causando overfitting na
    regressão linear.

    SMA e EMA carregam basicamente a mesma informação (tendência de preço
    suavizada); mantemos só EMA. Os lags de close puro (`close_lag_N`)
    também são redundantes com `return_lag_N`, que já captura a variação
    -- mantemos só os retornos, que é a versão mais informativa pro modelo.
    """
    exclude = {"date", "symbol", "target_return"}
    drop_prefixes = ("sma_",)
    drop_exact_prefix = "close_lag_"

    reduced = [
        c for c in all_columns
        if c not in exclude
        and not c.startswith(drop_prefixes)
        and not c.startswith(drop_exact_prefix)
    ]
    return reduced


def get_model_feature_columns(df: pd.DataFrame, reduced: bool = True) -> list:
    """
    Ponto único de verdade pra decidir quais colunas viram feature de
    modelo. Além da redução de redundância (get_reduced_feature_set),
    remove qualquer coluna que seja NaN em 100% das linhas -- necessário
    pra renda fixa, onde `volume` não existe e ficaria NaN o tempo todo,
    o que quebraria modelos do sklearn (não aceitam NaN nativamente).

    Sempre use esta função (em vez de montar a lista de colunas na mão)
    ao chamar run_backtest -- assim a lógica de "quais colunas são
    features válidas" fica num só lugar.
    """
    cols = get_reduced_feature_set(df.columns.tolist()) if reduced else [
        c for c in df.columns if c not in {"date", "symbol", "target_return"}
    ]
    return [c for c in cols if df[c].notna().any()]


def build_features(raw_path: str, symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(raw_path)
    df = df.sort_values("date").reset_index(drop=True)

    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_target(df)

    # Linhas com NaN no começo (por causa das janelas de indicadores) e no
    # fim (por causa do target futuro) não servem pra treino -- mas vale
    # manter separado de "descartar" para você decidir o que fazer com elas.
    # `volume` é excluído do critério de descarte porque séries de renda
    # fixa (CDI, Selic) não têm volume por definição -- seria NaN em todas
    # as linhas, e um dropna() ingênuo apagaria o DataFrame inteiro.
    n_before = len(df)
    dropna_cols = [c for c in df.columns if c != "volume"]
    df_clean = df.dropna(subset=dropna_cols).reset_index(drop=True)
    n_after = len(df_clean)
    print(f"[{symbol}] {n_before} linhas brutas -> {n_after} linhas utilizáveis "
          f"({n_before - n_after} descartadas por NaN de janela/horizonte)")

    return df_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Caminho do parquet bruto")
    parser.add_argument("--symbol", required=True, help="Símbolo do ativo, ex: BTC")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    df = build_features(args.input, args.symbol)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.symbol}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Salvo em {out_path} ({len(df)} linhas, {len(df.columns)} colunas)")


if __name__ == "__main__":
    main()