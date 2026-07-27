"""
Script genérico de coleta -- evita ter que editar data_collectors/*.py
toda vez que você quiser um símbolo ou período diferente.

Uso:
    python -m scripts.fetch_data --source crypto --symbol BTC --start 2023-01-01
    python -m scripts.fetch_data --source equities --symbol AAPL --start 2023-01-01
    python -m scripts.fetch_data --source equities --symbol PETR4.SA --start 2023-01-01

IMPORTANTE sobre re-coletar com período maior:
O fetch_and_save() é idempotente "pra frente" -- se você já tem dados salvos
e pede um `start` mais antigo que o que já existe, ele NÃO vai voltar no
tempo pra preencher o histórico anterior (só estende dali pra frente).
Se quiser mais histórico do que já tem salvo, use --force-refetch pra
apagar o arquivo raw existente e buscar tudo de novo do zero.
"""

import argparse
from datetime import date, datetime
from pathlib import Path

from data_collectors.crypto import CoinGeckoCollector
from data_collectors.equities import YFinanceCollector
from data_collectors.fixed_income import BCBCollector

COLLECTORS = {
    "crypto": CoinGeckoCollector,
    "equities": YFinanceCollector,
    "fixed_income": BCBCollector,
}


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=COLLECTORS.keys())
    parser.add_argument("--symbol", required=True, help="Ex: BTC, AAPL, PETR4.SA")
    parser.add_argument("--start", type=parse_date, required=True, help="Formato: AAAA-MM-DD")
    parser.add_argument("--end", type=parse_date, default=date.today(), help="Formato: AAAA-MM-DD (default: hoje)")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--force-refetch", action="store_true",
        help="Apaga o parquet existente antes de buscar -- necessário se você quer um "
             "histórico mais antigo do que o que já está salvo (veja nota no topo do arquivo)."
    )
    args = parser.parse_args()

    collector_cls = COLLECTORS[args.source]
    collector = collector_cls(raw_dir=args.raw_dir)

    if args.force_refetch:
        existing_path = collector.raw_dir / f"{args.symbol.replace('/', '-')}.parquet"
        if existing_path.exists():
            existing_path.unlink()
            print(f"Removido: {existing_path}")

    df = collector.fetch_and_save(args.symbol, start=args.start, end=args.end)
    print(f"\n{args.symbol}: {len(df)} linhas salvas, de {df['date'].min().date()} até {df['date'].max().date()}")


if __name__ == "__main__":
    main()