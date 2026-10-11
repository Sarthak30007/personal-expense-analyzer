"""Stage 5B: forecast next month's total spend with lag-feature regression.

Run:  python -m src.forecast

Method
  * Aggregate debits to one row per month.
  * Features for month t use only months < t (lags, rolling mean/std) plus the
    calendar month (sin/cos). See src/features.py.
  * Time-ordered split: the last TEST_MONTHS months are the test set (no shuffling).
  * Model selection with TimeSeriesSplit on the training part only.
  * Compare against naive baselines (last month, 3-month moving average).
  * Refit the winner on all history and predict the next month.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (FORECAST_FEATURES, load_clean, make_supervised, monthly_aggregates,
                          next_month_row)
from src.preprocess import flag_outliers
from src.utils import (CATEGORIZED_CSV, CLEAN_CSV, FIGURES_DIR, FORECASTER_PATH, MODELS_DIR,
                       RANDOM_STATE, ensure_dirs, get_logger)

log = get_logger("forecast")

TEST_MONTHS = 4


def candidate_models() -> dict:
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAPE_%": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 2),
    }


def main() -> None:
    plt.switch_backend("Agg")      # script mode: save figures, never open windows
    ensure_dirs()

    # Prefer the file where unlabelled rows were categorized by the NLP model
    source = CATEGORIZED_CSV if CATEGORIZED_CSV.exists() else CLEAN_CSV
    log.info("Using %s", source.name)
    df = load_clean(source)
    # After NLP categorization every debit has a category, so re-score outliers on it.
    if "category_filled" in df.columns:
        df["is_outlier"] = flag_outliers(df.assign(category=df["category_filled"]))
    # Forecast *regular* spending: one-off big purchases are unpredictable and would
    # dominate the error. Track them separately.
    monthly = monthly_aggregates(df, exclude_outliers=True)
    log.info("Monthly series: %d months, %s to %s", len(monthly),
             monthly.index[0].strftime("%Y-%m"), monthly.index[-1].strftime("%Y-%m"))

    data = make_supervised(monthly)
    train, test = data.iloc[:-TEST_MONTHS], data.iloc[-TEST_MONTHS:]
    X_train, y_train = train[FORECAST_FEATURES], train["target"]
    X_test, y_test = test[FORECAST_FEATURES], test["target"]
    log.info("Train rows: %d | test rows: %d", len(train), len(test))

    results: dict = {
        "baseline_last_month": regression_metrics(y_test, X_test["lag_1"]),
        "baseline_moving_avg_3": regression_metrics(y_test, X_test["roll_mean_3"]),
    }

    # ---- model selection with time-ordered CV (training data only) --------
    tscv = TimeSeriesSplit(n_splits=4)
    cv_mae, fitted, test_pred = {}, {}, {}
    for name, model in candidate_models().items():
        cv_mae[name] = -cross_val_score(model, X_train, y_train, cv=tscv,
                                        scoring="neg_mean_absolute_error").mean()
        model.fit(X_train, y_train)
        fitted[name] = model
        test_pred[name] = model.predict(X_test)
        results[name] = regression_metrics(y_test, test_pred[name])
        results[name]["CV_MAE"] = round(float(cv_mae[name]), 2)

    best_name = min(cv_mae, key=cv_mae.get)
    log.info("Results on the last %d months:\n%s", TEST_MONTHS, pd.DataFrame(results).T.to_string())
    log.info("Best model by time-series CV MAE: %s", best_name)

    # ---- plot actual vs predicted -----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(monthly.index, monthly["total_spend"], marker="o", color="#4c72b0", label="Actual")
    ax.plot(test.index, test_pred[best_name], marker="s", ls="--", color="#dd8452",
            label=f"Predicted ({best_name})")
    ax.plot(test.index, X_test["lag_1"], marker="x", ls=":", color="gray", label="Naive (last month)")
    ax.axvspan(test.index[0] - pd.Timedelta(days=15), test.index[-1] + pd.Timedelta(days=15),
               color="gold", alpha=0.15, label="Test period")
    ax.set_title("Monthly spend: actual vs predicted")
    ax.set_ylabel("Total spend (₹)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "forecast_actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- refit on all history, forecast next month ------------------------
    final = candidate_models()[best_name].fit(data[FORECAST_FEATURES], data["target"])
    month, row = next_month_row(monthly)
    forecast = float(final.predict(row)[0])
    log.info("Forecast for %s: ₹%.0f (last month actual: ₹%.0f)",
             month.strftime("%Y-%m"), forecast, monthly["total_spend"].iloc[-1])

    joblib.dump({"model": final, "features": FORECAST_FEATURES, "best_name": best_name,
                 "next_month": month, "forecast": forecast}, FORECASTER_PATH)
    results.update({"best_model": best_name, "next_month": month.strftime("%Y-%m"),
                    "next_month_forecast": round(forecast, 2)})
    with open(MODELS_DIR / "forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved %s", FORECASTER_PATH)


if __name__ == "__main__":
    main()