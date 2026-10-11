"""Personal Expense Analyzer: Streamlit app.

Run locally:   streamlit run app.py
Deploy:        see DEPLOY.md (Streamlit Community Cloud)
"""

from __future__ import annotations

import io
import json

import joblib
import pandas as pd
import streamlit as st

from src.features import AMOUNT_COL, TEXT_COL, load_clean
from src.inference import (
    MIN_MONTHS_FOR_FORECAST,
    bootstrap_artifacts,
    categorize_dataframe,
    quick_forecast,
    regular_monthly,
    sample_template,
    validate_upload,
)
from src.preprocess import clean_description, clean_transactions
from src.utils import CATEGORIZED_CSV, CATEGORIZER_PATH, FORECASTER_PATH, MODELS_DIR

st.set_page_config(
    page_title="Personal Expense Analyzer", page_icon="💸", layout="wide"
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(
    show_spinner="First run: generating demo data and training models (about a minute)..."
)
def bootstrap() -> bool:
    bootstrap_artifacts()
    return True


@st.cache_resource(show_spinner=False)
def get_model():
    return joblib.load(CATEGORIZER_PATH)


@st.cache_resource(show_spinner=False)
def get_forecast_bundle():
    return joblib.load(FORECASTER_PATH)


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return load_clean(CATEGORIZED_CSV)


@st.cache_data(show_spinner=False)
def load_metrics(name: str) -> dict:
    path = MODELS_DIR / name
    return json.load(open(path, encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner="Cleaning and categorizing your transactions...")
def process_upload(content: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(content))
    problems = validate_upload(raw)
    if problems:
        raise ValueError(" ".join(problems))
    df = clean_transactions(raw)
    if df.empty:
        raise ValueError(
            "No usable rows after cleaning (check date, amount and type values)."
        )
    return categorize_dataframe(df, get_model())


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar: data source + filters
# --------------------------------------------------------------------------- #
bootstrap()

st.sidebar.title("💸 Expense Analyzer")
source = st.sidebar.radio("Data source", ["Demo data", "Upload my CSV"])
is_demo = source == "Demo data"

if is_demo:
    df = load_demo()
else:
    st.sidebar.download_button(
        "Download CSV template",
        sample_template(),
        file_name="expense_template.csv",
        mime="text/csv",
    )
    st.sidebar.caption(
        "Columns: date, description, amount, type (debit/credit), "
        "category (optional)."
    )
    uploaded = st.sidebar.file_uploader("Your bank export (CSV)", type="csv")
    if uploaded is None:
        st.title("Personal Expense Analyzer")
        st.info("Upload a CSV in the sidebar, or switch to **Demo data**.")
        st.stop()
    try:
        df = process_upload(uploaded.getvalue())
    except ValueError as err:
        st.error(str(err))
        st.stop()

d_min, d_max = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", (d_min, d_max), min_value=d_min, max_value=d_max
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    view = df[mask]
else:
    view = df
spend = view[view["type"] == "debit"]

st.sidebar.caption(
    "Uploaded files are processed in memory only. Do not upload real bank data "
    "to a publicly hosted copy of this app."
)

st.title("Personal Expense Analyzer")
st.caption(
    "Demo data (synthetic)" if is_demo else f"Your data: {len(df):,} transactions"
)

tab_overview, tab_cat, tab_review, tab_forecast, tab_try = st.tabs(
    ["Overview", "Categories", "Needs review", "Forecast", "Try the categorizer"]
)

# --------------------------------------------------------------------------- #
with tab_overview:
    total = spend["amount"].sum()
    income = view.loc[view["type"] == "credit", "amount"].sum()
    n_months = max(spend["year_month"].nunique(), 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spend", inr(total))
    c2.metric("Average / month", inr(total / n_months))
    c3.metric("Income", inr(income))
    c4.metric(
        "Savings rate", f"{100 * (income - total) / income:.1f}%" if income else "n/a"
    )

    st.subheader("Monthly spending")
    st.line_chart(spend.groupby("year_month")["amount"].sum())

    st.subheader("Spend by day of week")
    daily = spend.groupby("date")["amount"].sum().asfreq("D", fill_value=0.0)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = daily.groupby(daily.index.dayofweek).mean()
    by_dow.index = [names[i] for i in by_dow.index]
    st.bar_chart(by_dow)

# --------------------------------------------------------------------------- #
with tab_cat:
    st.subheader("Spend by category")
    by_cat = (
        spend.groupby("category_filled")["amount"].sum().sort_values(ascending=False)
    )
    st.bar_chart(by_cat)

    st.subheader("Category mix per month")
    st.bar_chart(
        spend.pivot_table(
            index="year_month",
            columns="category_filled",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
    )

    st.subheader("Top merchants")
    merchants = spend.assign(
        merchant=spend["description_clean"].str.split().str[0].fillna("unknown")
    )
    top = merchants.groupby("merchant")["amount"].agg(total="sum", transactions="size")
    st.dataframe(top.nlargest(10, "total").round(0))

# --------------------------------------------------------------------------- #
with tab_review:
    review = view[view["needs_review"].fillna(False).astype(bool)]
    labelled_by_model = view["predicted_category"].notna().sum()
    st.write(
        f"The model categorized **{labelled_by_model:,}** transactions that had no label; "
        f"**{len(review):,}** of them are low-confidence and worth a manual check."
    )
    st.dataframe(
        review.sort_values("confidence")[
            ["date", "description", "amount", "predicted_category", "confidence"]
        ]
    )

# --------------------------------------------------------------------------- #
with tab_forecast:
    st.caption(
        "Forecasts *regular* monthly spending. One-off large purchases are excluded "
        "because they are unpredictable."
    )
    monthly = regular_monthly(df)
    last_actual = float(monthly["total_spend"].iloc[-1])

    if is_demo:
        bundle = get_forecast_bundle()
        month, forecast = bundle["next_month"], float(bundle["forecast"])
        st.metric(
            f"Forecast for {month:%B %Y}",
            inr(forecast),
            delta=round(forecast - last_actual),
            delta_color="inverse",
        )
        st.caption(
            f"Model: {bundle['best_name']} (vs. last month's actual {inr(last_actual)})"
        )
        chart = pd.DataFrame({"Actual": monthly["total_spend"]})
        chart.loc[month, "Forecast"] = forecast
        chart.loc[monthly.index[-1], "Forecast"] = last_actual
        st.line_chart(chart.sort_index())

        scores = {
            k: v
            for k, v in load_metrics("forecast_metrics.json").items()
            if isinstance(v, dict)
        }
        if scores:
            st.subheader("Backtest on the last 4 months")
            st.dataframe(pd.DataFrame(scores).T)
    else:
        result = quick_forecast(monthly)
        if result is None:
            st.warning(
                f"Need at least {MIN_MONTHS_FOR_FORECAST} months of history to forecast "
                f"(you have {len(monthly)})."
            )
        else:
            st.metric(
                f"Forecast for {result['month']:%B %Y}",
                inr(result["forecast"]),
                delta=round(result["forecast"] - last_actual),
                delta_color="inverse",
            )
            better = result["mae_model"] < result["mae_naive"]
            st.write(
                f"Backtest error: model **{inr(result['mae_model'])}** vs "
                f"'same as last month' **{inr(result['mae_naive'])}** "
                f"({'model is better' if better else 'naive baseline is as good or better'})."
            )
            st.dataframe(result["backtest"])
            st.line_chart(monthly["total_spend"])

# --------------------------------------------------------------------------- #
with tab_try:
    st.write("Type a raw bank description and see how the NLP model categorizes it.")
    text = st.text_input("Bank description", "UPI/SWIGGY/778899/PAY")
    amount = st.number_input("Amount (₹)", min_value=1.0, value=350.0, step=50.0)
    if text.strip():
        model = get_model()
        X = pd.DataFrame(
            {TEXT_COL: [clean_description(text)], AMOUNT_COL: [float(amount)]}
        )
        proba = model.predict_proba(X)[0]
        best = int(proba.argmax())
        st.success(f"**{model.classes_[best]}** ({proba[best]:.0%} confident)")
        st.bar_chart(
            pd.Series(proba, index=model.classes_).sort_values(ascending=False)
        )
    st.caption(
        "The model is trained on synthetic data. For your own merchants, label some "
        "of your transactions and retrain with `python -m src.train_categorizer`."
    )
