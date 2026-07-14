"""
Teste manual do BaseCollector com uma fonte fake (sem rede),
só para validar: validação de schema, salvamento, e merge idempotente.
"""
from datetime import date
import pandas as pd

from data_collectors.base import BaseCollector


class FakeCollector(BaseCollector):
    source_name = "fake_source"

    def __init__(self, canned_data, **kwargs):
        super().__init__(**kwargs)
        self.canned_data = canned_data
        self.call_count = 0

    def _fetch_raw(self, symbol, start, end):
        self.call_count += 1
        df = self.canned_data[(self.canned_data["date"] >= pd.Timestamp(start)) &
                               (self.canned_data["date"] <= pd.Timestamp(end))]
        return df.copy()


def make_canned_data():
    dates = pd.date_range("2026-01-01", "2026-01-10")
    return pd.DataFrame({
        "date": dates,
        "open": range(100, 110),
        "high": range(101, 111),
        "low": range(99, 109),
        "close": range(100, 110),
        "volume": range(1000, 1010),
    })


if __name__ == "__main__":
    import shutil
    shutil.rmtree("data/raw/fake_source", ignore_errors=True)

    canned = make_canned_data()
    collector = FakeCollector(canned, raw_dir="data/raw")

    # 1a chamada: busca tudo de 1 a 5 de janeiro
    df1 = collector.fetch_and_save("TEST", start=date(2026, 1, 1), end=date(2026, 1, 5))
    assert len(df1) == 5, f"esperado 5 linhas, veio {len(df1)}"
    print(f"OK: primeira carga -> {len(df1)} linhas, colunas: {list(df1.columns)}")

    # 2a chamada: mesmo intervalo -> não deve duplicar (idempotência)
    df2 = collector.fetch_and_save("TEST", start=date(2026, 1, 1), end=date(2026, 1, 5))
    assert len(df2) == 5, f"esperava continuar 5 linhas (idempotente), veio {len(df2)}"
    print(f"OK: segunda carga idempotente -> {len(df2)} linhas")

    # 3a chamada: estende até dia 10 -> deve crescer sem duplicar as antigas
    df3 = collector.fetch_and_save("TEST", start=date(2026, 1, 1), end=date(2026, 1, 10))
    assert len(df3) == 10, f"esperado 10 linhas após extensão, veio {len(df3)}"
    assert df3["date"].is_monotonic_increasing
    assert not df3["date"].duplicated().any()
    print(f"OK: extensão sem duplicatas -> {len(df3)} linhas")

    print("\nTodos os testes passaram.")
    print(df3.tail(3))
