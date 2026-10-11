"""Helpers used by the Streamlit app (app.py).

Nothing here is Streamlit-specific, so it can be unit-tested or reused elsewhere.

Public API
----------
MIN_MONTHS_FOR_FORECAST   minimum months of history for a forecast on uploaded data
bootstrap_artifacts()     build demo data + train both models if they are missing
validate_upload(raw)      list of human-readable problems with an uploaded CSV
categorize_dataframe()    label unlabelled debit rows with the trained categorizer
regular_monthly(df)       monthly series of *regular* spend (outliers excluded)
quick_forecast(monthly)   fit a small model on the user's own history and forecast
sample_template()         example CSV (str) for the download button
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    AMOUNT_COL,
    FORECAST_FEATURES,
    N_LAGS,
    TEXT_COL,
    make_supervised,
    monthly_aggregates,
    next_month_row,
)
from src.preprocess import flag_outliers
from src.utils import (
    CATEGORIZED_CSV,
    CATEGORIZER_PATH,
    CONFIDENCE_THRESHOLD,
    FORECASTER_PATH,
    get_logger,
)

log = get_logger("inference")

REQUIRED_COLUMNS = ("date", "description", "amount", "type")
MAX_UPLOAD_ROWS = 50_000

# make_supervised needs N_LAGS months of history before the first usable row, and we want
# at least ~4 training rows plus a 2-row backtest on top of that.
MIN_MONTHS_FOR_FORECAST = N_LAGS + 6


# --------------------------------------------------------------------------- #
# Start-up: make sure the demo data and trained models exist
# --------------------------------------------------------------------------- #
def artifacts_ready() -> bool:
    return all(p.exists() for p in (CATEGORIZED_CSV, CATEGORIZER_PATH, FORECASTER_PATH))


def bootstrap_artifacts(force: bool = False) -> None:
    """Generate synthetic data and train both models if they are not on disk yet.

    On Streamlit Community Cloud the repo has no data/ or models/ contents (they are
    git-ignored), so the first visit runs the pipeline once (~1 minute). The result
    stays on disk until the app container is rebuilt.
    """
    if artifacts_ready() and not force:
        return
    # Imported here so that importing this module stays cheap.
    from src import forecast, generate_data, preprocess, train_categorizer

    log.info(
        "Artifacts missing: running generate_data -> preprocess -> categorizer -> forecast"
    )
    generate_data.main([])
    preprocess.main()
    train_categorizer.main()  # must run before forecast: forecast uses its predictions
    forecast.main()


# --------------------------------------------------------------------------- #
# Uploaded CSV
# --------------------------------------------------------------------------- #
def validate_upload(raw: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list = OK). Run on the raw, uncleaned CSV."""
    problems: list[str] = []
    if raw.empty:
        return ["The file has no rows."]
    if len(raw) > MAX_UPLOAD_ROWS:
        problems.append(
            f"The file has {len(raw):,} rows; the limit is {MAX_UPLOAD_ROWS:,}."
        )

    cols = {str(c).strip().lower() for c in raw.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        problems.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)} (+ optional category). "
            f"Found: {', '.join(map(str, raw.columns))}."
        )
    return problems


def sample_template() -> str:
    """A small example CSV in the format the app expects."""
    return (
        "date,description,amount,type,category\n"
        "2026-01-01,PG Rent Landlord,6500,debit,Rent\n"
        "2026-01-02,UPI/Dad/123456/CREDIT,38000,credit,Income\n"
        "2026-01-03,UPI/SWIGGY/778899/PAY,320,debit,\n"
        "2026-01-05,UBER INDIA*88231,210,debit,\n"
        "2026-01-08,Amazon Pay India,1499,debit,Shopping\n"
        "2026-01-11,NETFLIX.COM 4417,199,debit,\n"
        "2026-01-14,Apollo Pharmacy POS Pune,640,debit,\n"
        "2026-01-19,Spotify,119,debit,Entertainment\n"
    )


# --------------------------------------------------------------------------- #
# Categorization
# --------------------------------------------------------------------------- #
def categorize_dataframe(
    df: pd.DataFrame, model, threshold: float = CONFIDENCE_THRESHOLD
) -> pd.DataFrame:
    """Fill in missing categories on debit rows using a fitted categorizer pipeline.

    Takes the output of `clean_transactions` and adds the same columns that
    `train_categorizer` writes to transactions_categorized.csv:
    predicted_category, confidence, needs_review, category_filled
    (and refreshes `is_outlier` now that every debit has a category).
    """
    out = df.copy()
    out[TEXT_COL] = out[TEXT_COL].fillna("").astype(str)

    is_debit = out["type"] == "debit"
    if not is_debit.any():
        raise ValueError(
            "No debit (spending) transactions found: check the `type` column "
            "(use debit/credit, or DR/CR)."
        )

    todo = is_debit & out["category"].isna()
    out["predicted_category"] = pd.Series(
        [None] * len(out), index=out.index, dtype="object"
    )
    out["confidence"] = np.nan
    if todo.any():
        proba = model.predict_proba(out.loc[todo, [TEXT_COL, AMOUNT_COL]])
        out.loc[todo, "predicted_category"] = model.classes_[proba.argmax(axis=1)]
        out.loc[todo, "confidence"] = proba.max(axis=1).round(4)

    out["needs_review"] = (out["confidence"] < threshold).astype(bool)

    filled = out["category"].fillna(out["predicted_category"])
    # Credits without a label are treated as income (they are never sent to the model).
    out["category_filled"] = filled.where(is_debit | filled.notna(), "Income")

    out["is_outlier"] = flag_outliers(out.assign(category=out["category_filled"]))
    return out


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def regular_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly debit totals with one-off large purchases excluded.

    Same recipe as src/forecast.py: outliers are re-scored on the filled-in categories
    first, because the saved CSV flagged them before the NLP model had labelled anything.
    """
    d = df.copy()
    if "category_filled" in d.columns:
        d["is_outlier"] = flag_outliers(d.assign(category=d["category_filled"]))
    return monthly_aggregates(d, exclude_outliers=True)


def _ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=3.0))])


def quick_forecast(monthly: pd.DataFrame) -> dict | None:
    """Fit a small ridge model on the user's own monthly history and forecast next month.

    Returns None when there is too little history. Otherwise a dict with
    month, forecast, mae_model, mae_naive and backtest (a DataFrame of the held-out months).
    """
    if len(monthly) < MIN_MONTHS_FOR_FORECAST:
        return None

    data = make_supervised(monthly)
    n_test = max(2, min(4, len(data) // 4))
    train, test = data.iloc[:-n_test], data.iloc[-n_test:]

    # Honest backtest: fit on the past only, predict the held-out latest months.
    backtest_model = _ridge().fit(train[FORECAST_FEATURES], train["target"])
    pred = backtest_model.predict(test[FORECAST_FEATURES])
    naive = test["lag_1"].to_numpy()

    backtest = pd.DataFrame(
        {
            "Actual": test["target"].round(0),
            "Model": np.round(pred, 0),
            "Naive (last month)": np.round(naive, 0),
        }
    )
    backtest.index = backtest.index.strftime("%Y-%m")

    # Final model uses all history to predict the month after the last one.
    final = _ridge().fit(data[FORECAST_FEATURES], data["target"])
    month, row = next_month_row(monthly)
    forecast = max(0.0, float(final.predict(row)[0]))

    return {
        "month": month,
        "forecast": forecast,
        "mae_model": float(mean_absolute_error(test["target"], pred)),
        "mae_naive": float(mean_absolute_error(test["target"], naive)),
        "backtest": backtest,
    }
