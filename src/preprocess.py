"""Stage 2: clean and standardize the raw transactions.

Run:  python -m src.preprocess

Reads   data/raw/transactions_raw.csv
Writes  data/interim/transactions_parsed.csv     (types fixed, text still raw)
        data/processed/transactions_clean.csv    (model-ready)

Decisions (document these in your README):
  * exact duplicates  -> dropped
  * missing date/amount/description/type -> row dropped (cannot be used)
  * missing category  -> kept (these are the rows the ML model will label)
  * outliers          -> FLAGGED in `is_outlier`, not removed (big purchases are real)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils import CLEAN_CSV, INTERIM_CSV, RAW_CSV, ensure_dirs, get_logger

log = get_logger("preprocess")

DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y"]

# Bank boilerplate that carries no information about the category
NOISE_TOKENS = {
    "upi", "pay", "payment", "neft", "imps", "pos", "order", "txn", "ref",
    "credit", "debit", "okaxis", "okicici", "okhdfc", "ybl",
}


# --------------------------------------------------------------------------- #
# Column-level parsers (small, pure, unit-tested)
# --------------------------------------------------------------------------- #
def parse_amount(series: pd.Series) -> pd.Series:
    """'₹1,250.00', 'Rs. 1250', 'INR 1,250.50', 1250 -> float. Unparseable -> NaN."""
    s = series.astype("object").map(lambda v: "" if pd.isna(v) else str(v))
    s = s.str.replace(r"(?i)rs\.?|inr|₹|,", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def parse_date(series: pd.Series) -> pd.Series:
    """Parse several explicit formats (avoids day/month guessing). Unparseable -> NaT."""
    s = series.astype("object").map(lambda v: "" if pd.isna(v) else str(v).strip())
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = out.isna()
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out


def normalize_type(series: pd.Series) -> pd.Series:
    """'DR', 'Debit', 'DEBIT' -> 'debit'; 'CR', 'Credit' -> 'credit'; else NaN."""
    mapping = {"debit": "debit", "dr": "debit", "credit": "credit", "cr": "credit"}
    key = series.astype("object").map(lambda v: "" if pd.isna(v) else str(v).strip().lower())
    return key.map(mapping)


def clean_description(text) -> str:
    """'UPI/ZOMATO/40213/PAY' -> 'zomato'; 'AMZN MKTP*2K4' -> 'amzn mktp'."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [w for w in t.split() if w not in NOISE_TOKENS and not any(c.isdigit() for c in w)]
    return " ".join(tokens)


def flag_outliers(df: pd.DataFrame, k: float = 1.5, min_group: int = 20) -> pd.Series:
    """IQR rule within each (type, category) group.

    Rows without a category are not scored (their group would mix unrelated
    spending, e.g. rent vs. coffee), and groups that are too small are skipped.
    """
    flags = pd.Series(False, index=df.index)
    known = df[df["category"].notna()]
    key = known["type"].astype(str) + "|" + known["category"].astype(str)
    grp = known.groupby(key)["amount"]
    q1, q3 = grp.transform(lambda s: s.quantile(0.25)), grp.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    size = key.map(key.value_counts())
    out = (known["amount"] > q3 + k * iqr) | (known["amount"] < q1 - k * iqr)
    flags.loc[known.index] = (out & (iqr > 0) & (size >= min_group)).astype(bool)
    return flags


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def parse_and_type(raw: pd.DataFrame) -> pd.DataFrame:
    """Step A: fix dtypes, drop duplicates / unusable rows."""
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "txn_id" not in df.columns:
        df.insert(0, "txn_id", [f"R{i:05d}" for i in range(len(df))])
    if "category" not in df.columns:
        df["category"] = np.nan

    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Dropped %d exact duplicate rows", n0 - len(df))

    df["date"] = parse_date(df["date"])
    df["amount"] = parse_amount(df["amount"])
    df["type"] = normalize_type(df["type"])
    df["description"] = df["description"].astype("object").map(
        lambda v: v.strip() if isinstance(v, str) and v.strip() else np.nan)
    df["category"] = df["category"].astype("object").map(
        lambda v: v.strip().title() if isinstance(v, str) and v.strip() else np.nan)

    before = len(df)
    df = df.dropna(subset=["date", "amount", "description", "type"])
    df = df[df["amount"].abs() > 0]
    log.info("Dropped %d rows with missing/invalid date, amount, description or type", before - len(df))

    df["amount"] = df["amount"].abs()
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step B: text normalization, outlier flag, calendar columns."""
    df = df.copy()
    df["description_clean"] = df["description"].map(clean_description)
    df["is_outlier"] = flag_outliers(df)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek          # 0 = Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    return df.sort_values(["date", "txn_id"]).reset_index(drop=True)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: raw dataframe in, processed dataframe out."""
    return add_features(parse_and_type(raw))


def main() -> None:
    ensure_dirs()
    raw = pd.read_csv(RAW_CSV, encoding="utf-8")
    log.info("Loaded %d raw rows", len(raw))
    log.info("Missing values:\n%s", raw.isna().sum().to_string())

    parsed = parse_and_type(raw)
    parsed.to_csv(INTERIM_CSV, index=False, encoding="utf-8")

    clean = add_features(parsed)
    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")

    debit = clean[clean["type"] == "debit"]
    log.info("Clean rows: %d | debits: %d | labelled debits: %d | outliers flagged: %d",
             len(clean), len(debit), debit["category"].notna().sum(), int(clean["is_outlier"].sum()))
    log.info("Saved %s", CLEAN_CSV)


if __name__ == "__main__":
    main()