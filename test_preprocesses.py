import numpy as np
import pandas as pd

from src.preprocess import (
    clean_description,
    clean_transactions,
    normalize_type,
    parse_amount,
    parse_date,
)


def test_parse_amount_handles_symbols_and_commas():
    s = pd.Series(["₹1,250.00", "Rs. 1250.50", "INR 2,000.00", "300.5", None, "abc"])
    out = parse_amount(s)
    assert out.iloc[0] == 1250.00
    assert out.iloc[1] == 1250.50
    assert out.iloc[2] == 2000.00
    assert out.iloc[3] == 300.5
    assert np.isnan(out.iloc[4]) and np.isnan(out.iloc[5])


def test_parse_date_mixed_formats():
    s = pd.Series(
        ["2025-03-04", "04-03-2025", "04/03/2025", "04 Mar 2025", "not a date"]
    )
    out = parse_date(s)
    assert (out.iloc[:4] == pd.Timestamp("2025-03-04")).all()
    assert pd.isna(out.iloc[4])


def test_normalize_type():
    out = normalize_type(pd.Series(["DR", "Debit", " credit ", "CR", "xyz", None]))
    assert list(out.iloc[:4]) == ["debit", "debit", "credit", "credit"]
    assert out.iloc[4:].isna().all()


def test_clean_description_removes_refs_and_boilerplate():
    assert clean_description("UPI/ZOMATO/40213/PAY") == "zomato"
    assert clean_description("AMZN MKTP*2K4") == "amzn mktp"
    assert clean_description("  POS Starbucks   Pune ") == "starbucks pune"
    assert clean_description(None) == ""


def test_clean_transactions_drops_duplicates_and_bad_rows():
    raw = pd.DataFrame(
        {
            "txn_id": ["a", "a", "b", "c"],
            "date": ["2025-01-05", "2025-01-05", "05/01/2025", "garbage"],
            "description": ["UPI/SWIGGY/1/PAY", "UPI/SWIGGY/1/PAY", "Uber*22", "x"],
            "amount": ["₹250.00", "₹250.00", "Rs. 120", "10"],
            "type": ["DR", "DR", "debit", "debit"],
            "category": ["food", "food", None, None],
        }
    )
    out = clean_transactions(raw)
    assert len(out) == 2  # duplicate + bad date removed
    assert out["category"].iloc[0] == "Food"  # casing normalized
    assert out["category"].isna().sum() == 1  # unlabelled row kept
    assert out["amount"].dtype == float
    assert {"description_clean", "year_month", "is_weekend"} <= set(out.columns)
