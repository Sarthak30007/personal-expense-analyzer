import numpy as np
import pandas as pd

from src.features import FORECAST_FEATURES, make_supervised, next_month_row


def _monthly(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "total_spend": np.arange(1, n + 1) * 1000.0,
            "n_txn": 50,
            "fixed_spend": 7000.0,
        },
        index=idx,
    )


def test_supervised_rows_use_only_past_months():
    monthly = _monthly()
    data = make_supervised(monthly)
    assert len(data) == len(monthly) - 3
    first = data.iloc[0]  # target = month 4 (4000)
    assert first["target"] == 4000.0
    assert (
        first["lag_1"] == 3000.0
        and first["lag_2"] == 2000.0
        and first["lag_3"] == 1000.0
    )
    assert (
        first["roll_mean_3"] == 2000.0
    )  # mean of months 1-3, NOT including the target


def test_next_month_row_shape_and_values():
    monthly = _monthly()
    month, row = next_month_row(monthly)
    assert month == pd.Timestamp("2024-11-01")
    assert list(row.columns) == FORECAST_FEATURES
    assert row["lag_1"].iloc[0] == 10000.0
