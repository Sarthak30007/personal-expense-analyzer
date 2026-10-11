"""Stage 3: feature engineering.

Two independent feature sets:
  * Categorizer -> TF-IDF (word + char n-grams) on `description_clean` + binned log(amount)
  * Forecaster  -> monthly aggregates with lag / rolling / calendar features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer

from src.utils import CLEAN_CSV

TEXT_COL = "description_clean"
AMOUNT_COL = "amount"

FIXED_CATEGORIES = ("Rent", "Bills")
N_LAGS = 3
FORECAST_FEATURES = [
    "lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_std_3",
    "fixed_lag_1", "month_sin", "month_cos",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_clean(path: Path = CLEAN_CSV) -> pd.DataFrame:
    """Load a processed CSV. Empty cleaned descriptions come back as NaN -> make them ''."""
    df = pd.read_csv(path, parse_dates=["date"], encoding="utf-8")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


# --------------------------------------------------------------------------- #
# Categorizer features
# --------------------------------------------------------------------------- #
def split_labelled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Debit rows only -> (labelled, unlabelled)."""
    debit = df[df["type"] == "debit"]
    return debit[debit["category"].notna()], debit[debit["category"].isna()]


def build_preprocessor(
    word_ngram: tuple[int, int] = (1, 2),
    char_ngram: tuple[int, int] = (2, 5),
    use_amount: bool = True,
) -> ColumnTransformer:
    """TF-IDF on text (word + char n-grams) and, optionally, binned log-amount.

    Amount helps when the description is uninformative (e.g. 'PAYTM').
    Everything is fitted inside a Pipeline, so no information leaks from the test set.
    """
    transformers = [
        ("word", TfidfVectorizer(ngram_range=word_ngram, sublinear_tf=True), TEXT_COL),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram,
                                 sublinear_tf=True, min_df=2), TEXT_COL),
    ]
    if use_amount:
        amount_pipe = Pipeline([
            ("log", FunctionTransformer(np.log1p)),
            ("bins", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="uniform")),
        ])
        transformers.append(("amount", amount_pipe, [AMOUNT_COL]))
    return ColumnTransformer(transformers)


# --------------------------------------------------------------------------- #
# Forecaster features
# --------------------------------------------------------------------------- #
def monthly_aggregates(df: pd.DataFrame, category_col: str = "category_filled",
                       exclude_outliers: bool = False) -> pd.DataFrame:
    """One row per calendar month (debits only): total, #transactions, fixed-cost spend.

    With `exclude_outliers=True`, transactions flagged in `is_outlier` (e.g. a one-off
    laptop purchase) are left out, so the series reflects *regular* spending, which
    is what a monthly budget forecast should predict.
    """
    debit = df[df["type"] == "debit"].copy()
    if exclude_outliers and "is_outlier" in debit.columns:
        debit = debit[~debit["is_outlier"].astype(bool)]
    if category_col not in debit.columns:
        category_col = "category"
    cat = debit[category_col].fillna("Unknown")
    debit["month_start"] = debit["date"].dt.to_period("M").dt.to_timestamp()

    monthly = debit.groupby("month_start").agg(
        total_spend=(AMOUNT_COL, "sum"), n_txn=(AMOUNT_COL, "size"))
    fixed = debit[cat.isin(FIXED_CATEGORIES)].groupby("month_start")[AMOUNT_COL].sum()
    monthly["fixed_spend"] = fixed.reindex(monthly.index).fillna(0.0)
    return monthly.asfreq("MS").fillna(0.0)


def forecast_feature_row(totals: pd.Series, fixed: pd.Series, month: pd.Timestamp) -> dict:
    """Build the feature vector for `month` using ONLY history strictly before it.

    `totals` / `fixed` must end at the month immediately before `month`.
    The same function builds training rows and the final next-month row, so
    training and inference cannot drift apart (and cannot leak the future).
    """
    last = totals.iloc[-N_LAGS:]
    angle = 2 * np.pi * month.month / 12
    return {
        "lag_1": totals.iloc[-1],
        "lag_2": totals.iloc[-2],
        "lag_3": totals.iloc[-3],
        "roll_mean_3": last.mean(),
        "roll_std_3": last.std(),
        "fixed_lag_1": fixed.iloc[-1],
        "month_sin": np.sin(angle),
        "month_cos": np.cos(angle),
    }


def make_supervised(monthly: pd.DataFrame) -> pd.DataFrame:
    """Turn the monthly series into a supervised table: features + `target`, indexed by month."""
    rows, index = [], []
    for i in range(N_LAGS, len(monthly)):
        hist = monthly.iloc[:i]
        row = forecast_feature_row(hist["total_spend"], hist["fixed_spend"], monthly.index[i])
        row["target"] = monthly["total_spend"].iloc[i]
        rows.append(row)
        index.append(monthly.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="month"))


def next_month_row(monthly: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Feature row for the month after the last one in `monthly`."""
    month = monthly.index[-1] + pd.offsets.MonthBegin(1)
    row = forecast_feature_row(monthly["total_spend"], monthly["fixed_spend"], month)
    return month, pd.DataFrame([row], index=[month])[FORECAST_FEATURES]
    