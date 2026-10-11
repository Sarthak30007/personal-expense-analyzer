"""Stage 4: exploratory data analysis.

Run as a script:     python -m src.eda            (saves figures to reports/figures/)
Use in a notebook:   from src.eda import *
                     df = load_clean(); plot_monthly_trend(df)

Every plot function takes the processed dataframe and returns a matplotlib Figure.
Category-based plots use the *labelled* debit rows (about 75% of the data), so
they show proportions accurately even though totals are slightly understated.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.features import load_clean
from src.utils import FIGURES_DIR, ensure_dirs, get_logger

log = get_logger("eda")
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _debit(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"] == "debit"]


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    d = _debit(df)
    return d[d["category"].notna()]


def _daily_totals(df: pd.DataFrame) -> pd.Series:
    """Total debit spend per calendar day (days without spending = 0)."""
    return _debit(df).groupby("date")["amount"].sum().asfreq("D", fill_value=0.0)


def _save(fig: plt.Figure, name: str, save: bool) -> plt.Figure:
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Summary tables
# --------------------------------------------------------------------------- #
def summary(df: pd.DataFrame) -> pd.Series:
    d = _debit(df)
    monthly = d.groupby("year_month")["amount"].sum()
    income = df[df["type"] == "credit"]["amount"].sum()
    return pd.Series(
        {
            "period": f"{df['date'].min().date()} to {df['date'].max().date()}",
            "transactions (debit)": len(d),
            "total spend": round(d["amount"].sum(), 2),
            "avg monthly spend": round(monthly.mean(), 2),
            "median transaction": round(d["amount"].median(), 2),
            "total income": round(income, 2),
            "savings rate %": (
                round(100 * (income - d["amount"].sum()) / income, 1)
                if income
                else None
            ),
        }
    )


def detect_recurring(
    df: pd.DataFrame, min_month_share: float = 0.8, max_txn_per_month: float = 1.5
) -> pd.DataFrame:
    """Merchants that show up in most months about once per month (rent, subscriptions, bills)."""
    d = _debit(df).copy()
    d["merchant"] = d["description_clean"].str.split().str[0]
    d = d[d["merchant"].notna()]
    n_months = d["year_month"].nunique()
    rows = []
    for merchant, g in d.groupby("merchant"):
        per_month = g.groupby("year_month").size()
        if (
            len(per_month) >= min_month_share * n_months
            and per_month.mean() <= max_txn_per_month
        ):
            rows.append(
                {
                    "merchant": merchant,
                    "months_present": len(per_month),
                    "avg_amount": round(g["amount"].mean(), 2),
                    "cv": round(g["amount"].std() / g["amount"].mean(), 3),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("avg_amount", ascending=False)
        .reset_index(drop=True)
    )


def top_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["date", "description", "amount", "category"]
    return df[df["is_outlier"]].nlargest(n, "amount")[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_monthly_trend(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    monthly = _debit(df).groupby("year_month")["amount"].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly.index, monthly.values, marker="o", label="Monthly spend")
    ax.plot(
        monthly.index, monthly.rolling(3).mean(), ls="--", label="3-month moving avg"
    )
    ax.set_title("Monthly spending trend")
    ax.set_ylabel("₹")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    return _save(fig, "01_monthly_trend", save)


def plot_category_split(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    totals = _labelled(df).groupby("category")["amount"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    totals.plot.barh(ax=ax, color=sns.color_palette("viridis", len(totals)))
    for i, v in enumerate(totals.values):
        ax.text(v, i, f" {100 * v / totals.sum():.1f}%", va="center")
    ax.set_title("Spend by category (labelled rows)")
    ax.set_xlabel("₹")
    return _save(fig, "02_category_split", save)


def plot_category_month_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    pivot = _labelled(df).pivot_table(
        index="category",
        columns="year_month",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "₹"})
    ax.set_title("Category × month spend heatmap")
    ax.set_xlabel("")
    return _save(fig, "03_category_month_heatmap", save)


def plot_weekday_pattern(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    daily = _daily_totals(df)
    by_dow = daily.groupby(daily.index.dayofweek).mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4c72b0"] * 5 + ["#dd8452"] * 2
    ax.bar(labels, by_dow.values, color=colors)
    weekend, weekday = by_dow.iloc[5:].mean(), by_dow.iloc[:5].mean()
    ax.set_title(
        f"Average daily spend by weekday (weekend {weekend:,.0f} vs weekday {weekday:,.0f})"
    )
    ax.set_ylabel("₹ / day")
    return _save(fig, "04_weekday_pattern", save)


def plot_day_of_month(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    daily = _daily_totals(df)
    by_day = daily.groupby(daily.index.day).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_day.index, by_day.values, color="#55a868")
    ax.set_title("Average daily spend by day of month (rent / bills / salary effects)")
    ax.set_xlabel("Day of month")
    ax.set_ylabel("₹ / day")
    return _save(fig, "05_day_of_month", save)


def plot_top_merchants(df: pd.DataFrame, n: int = 10, save: bool = True) -> plt.Figure:
    d = _debit(df).copy()
    d["merchant"] = d["description_clean"].str.split().str[0].fillna("unknown")
    by_spend = d.groupby("merchant")["amount"].sum().nlargest(n).sort_values()
    by_count = d.groupby("merchant").size().nlargest(n).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    by_spend.plot.barh(ax=axes[0], color="#4c72b0")
    axes[0].set_title(f"Top {n} merchants by total spend")
    by_count.plot.barh(ax=axes[1], color="#c44e52")
    axes[1].set_title(f"Top {n} merchants by frequency")
    fig.tight_layout()
    return _save(fig, "06_top_merchants", save)


def plot_amount_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    lab = _labelled(df)
    order = lab.groupby("category")["amount"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=lab, x="category", y="amount", order=order, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Transaction amount distribution by category (log scale)")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "07_amount_distribution", save)


def plot_category_correlation(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    pivot = _labelled(df).pivot_table(
        index="year_month",
        columns="category",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    )
    corr = pivot.loc[:, pivot.std() > 0].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between monthly category totals")
    return _save(fig, "08_category_correlation", save)


# --------------------------------------------------------------------------- #
def main() -> None:
    plt.switch_backend("Agg")  # script mode: save figures, never open windows
    ensure_dirs()
    df = load_clean()
    print("\n=== SUMMARY ===\n", summary(df).to_string())
    print(
        "\n=== RECURRING TRANSACTIONS ===\n",
        detect_recurring(df).to_string(index=False),
    )
    print(
        "\n=== TOP ANOMALIES (flagged by IQR within category) ===\n",
        top_anomalies(df).to_string(index=False),
    )

    plots = [
        plot_monthly_trend,
        plot_category_split,
        plot_category_month_heatmap,
        plot_weekday_pattern,
        plot_day_of_month,
        plot_top_merchants,
        plot_amount_distribution,
        plot_category_correlation,
    ]
    for fn in plots:
        fig = fn(df)
        plt.close(fig)
    log.info("Saved %d figures to %s", len(plots), FIGURES_DIR)


if __name__ == "__main__":
    main()
