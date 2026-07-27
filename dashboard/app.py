"""
Dashboard do asset_predictor. Mostra preço + indicadores do ativo escolhido
e roda o backtest walk-forward ao vivo, comparando baselines vs modelos.

Uso:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# O Streamlit roda o script com a pasta dashboard/ no sys.path, não a raiz
# do projeto -- por isso os imports de features/models/backtesting abaixo
# falhariam sem essa linha, mesmo rodando o comando do lugar certo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from features.build_features import get_model_feature_columns
from models.baseline import NaiveBaseline, HistoricalMeanBaseline, LinearRegressionModel, RidgeRegressionModel
from models.xgboost_model import XGBoostModel
from backtesting.walk_forward import run_backtest

st.set_page_config(page_title="Asset Predictor", layout="wide")

PROCESSED_DIR = Path("data/processed")


@st.cache_data
def load_processed_symbols() -> list:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(p.stem for p in PROCESSED_DIR.glob("*.parquet"))


@st.cache_data
def load_data(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{symbol}.parquet")


@st.cache_data
def compute_backtest(symbol: str, min_train_size: int, test_size: int, n_splits: int) -> pd.DataFrame:
    df = load_data(symbol)
    feature_cols = get_model_feature_columns(df)

    models = {
        "naive": lambda: NaiveBaseline(),
        "historical_mean": lambda: HistoricalMeanBaseline(),
        "linear_regression": lambda: LinearRegressionModel(),
        "ridge_alpha_1000": lambda: RidgeRegressionModel(alpha=1000),
        "xgboost": lambda: XGBoostModel(),
    }

    results = run_backtest(
        df, feature_cols=feature_cols, target_col="target_return", models=models,
        n_splits=n_splits, min_train_size=min_train_size, test_size=test_size,
    )
    return results


def price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", line=dict(color="#4C9AFF")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_21"], name="SMA 21", line=dict(color="#FF8B00", dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="Bollinger Superior",
                              line=dict(color="rgba(150,150,150,0.4)")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="Bollinger Inferior",
                              line=dict(color="rgba(150,150,150,0.4)"), fill="tonexty"))
    fig.update_layout(title=f"{symbol} — preço e bandas de Bollinger", height=420,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], name="RSI 14", line=dict(color="#8777D9")))
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,0,0,0.4)")
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,0,0.4)")
    fig.update_layout(title="RSI (14)", height=250, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 100])
    return fig


def main():
    st.title("📊 Asset Predictor")
    st.caption("Previsão de retorno diário — baselines vs modelos, com validação walk-forward")

    symbols = load_processed_symbols()
    if not symbols:
        st.warning(
            "Nenhum arquivo encontrado em `data/processed/`. Rode "
            "`python -m features.build_features --input ... --symbol ...` primeiro."
        )
        return

    with st.sidebar:
        st.header("Configuração")
        symbol = st.selectbox("Ativo", symbols)
        n_splits = st.slider("Número de folds", 3, 10, 5)
        min_train_size = st.slider("Tamanho mínimo de treino", 30, 300, 100, step=10)
        test_size = st.slider("Tamanho do teste por fold", 5, 50, 20, step=5)
        st.divider()
        st.caption(
            "Achados completos em [FINDINGS.md](FINDINGS.md): nenhum modelo "
            "testado superou o baseline `historical_mean` de forma "
            "consistente entre BTC e AAPL nesse horizonte de 1 dia."
        )

    df = load_data(symbol)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(price_chart(df, symbol), use_container_width=True)
        st.plotly_chart(rsi_chart(df), use_container_width=True)

    with col2:
        st.subheader("Resumo dos dados")
        st.metric("Linhas disponíveis", len(df))
        st.metric("Período", f"{df['date'].min().date()} → {df['date'].max().date()}")
        st.metric("Retorno médio diário", f"{df['target_return'].mean():.4%}")
        st.metric("Volatilidade média (21d)", f"{df['volatility_21'].mean():.4%}")

    st.divider()
    st.subheader("Backtest walk-forward")

    try:
        results = compute_backtest(symbol, min_train_size, test_size, n_splits)
    except ValueError as e:
        st.error(f"Não foi possível rodar o backtest: {e}")
        return

    summary = results.groupby("model_name").agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        directional_accuracy_mean=("directional_accuracy", "mean"),
        n_folds=("fold", "count"),
    ).sort_values("mae_mean")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(
            summary.style.format({
                "mae_mean": "{:.5f}", "rmse_mean": "{:.5f}",
                "directional_accuracy_mean": "{:.0%}",
            }).background_gradient(subset=["directional_accuracy_mean"], cmap="RdYlGn"),
            use_container_width=True,
        )

    with col_b:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary.index, y=summary["directional_accuracy_mean"],
                              marker_color="#4C9AFF"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="acaso (50%)")
        fig.update_layout(title="Directional accuracy por modelo", height=350,
                           margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detalhe por fold"):
        st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()