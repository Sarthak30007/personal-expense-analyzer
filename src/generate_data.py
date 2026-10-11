"""Stage 1: generate realistic, deliberately messy bank-style transactions.

Run:  python -m src.generate_data [--months 36] [--seed 42]

Outputs
-------
data/raw/transactions_raw.csv   messy export (what a bank would give you)
data/raw/ground_truth.csv       true category for every transaction id, used ONLY
                                to evaluate the categorizer on rows whose label
                                was blanked out.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils import GROUND_TRUTH_CSV, RANDOM_STATE, RAW_CSV, ensure_dirs, get_logger

try:  # Faker is in requirements.txt, but fall back gracefully if missing
    from faker import Faker
except ImportError:  # pragma: no cover
    Faker = None

log = get_logger("generate_data")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# rate = average transactions / month, mu & sigma = lognormal amount parameters
CATEGORY_SPEC = {
    "Food": dict(
        merchants=["Zomato", "Swiggy", "Dominos", "McDonalds", "Starbucks", "Chai Point",
                   "Blinkit", "BigBasket", "Campus Canteen", "Haldirams"],
        rate=28, mu=5.5, sigma=0.55, weekend_boost=1.8),
    "Transport": dict(
        merchants=["Uber", "Ola", "Rapido", "IRCTC", "Indian Oil", "Metro Card Recharge",
                   "RedBus", "Shell Petrol Pump"],
        rate=10, mu=4.8, sigma=0.60, weekend_boost=0.7),
    "Shopping": dict(
        merchants=["Amazon", "Flipkart", "Myntra", "Decathlon", "Ajio", "Croma",
                   "Reliance Trends", "Lifestyle Store"],
        rate=4, mu=6.8, sigma=0.70, weekend_boost=1.5),
    "Entertainment": dict(
        merchants=["BookMyShow", "PVR Cinemas", "Steam Games", "Hotstar", "Gaming Zone",
                   "Cafe Coffee Day Arcade"],
        rate=3, mu=5.5, sigma=0.50, weekend_boost=2.0),
    "Health": dict(
        merchants=["Apollo Pharmacy", "Practo", "Netmeds", "Cult Fit", "MedPlus",
                   "City Hospital"],
        rate=2, mu=6.0, sigma=0.60, weekend_boost=1.0),
    "Education": dict(
        merchants=["Udemy", "Coursera", "Unacademy", "Campus Bookstore", "GeeksforGeeks",
                   "Xerox Shop", "Amazon Books"],
        rate=2, mu=6.5, sigma=0.80, weekend_boost=1.0),
    "Transfers": dict(merchants=[], rate=5, mu=6.2, sigma=0.80, weekend_boost=1.2),
}

# Recurring monthly items: (category, merchant, base amount, day range, jitter)
FIXED_ITEMS = [
    ("Rent", "PG Rent Landlord", 6500, (1, 3), 0.0),
    ("Bills", "Jio Prepaid", 299, (5, 10), 0.0),
    ("Bills", "ACT Fibernet", 799, (8, 12), 0.0),
    ("Bills", "Electricity Board", 650, (14, 18), 0.35),
    ("Entertainment", "Netflix", 199, (11, 13), 0.0),
    ("Entertainment", "Spotify", 119, (19, 21), 0.0),
]

# month -> multiplier on amounts (festive season, semester start, holidays...)
SEASONALITY = {
    "Shopping": {10: 1.6, 11: 1.5, 12: 1.3, 1: 1.1},
    "Entertainment": {6: 1.2, 12: 1.4},
    "Transport": {5: 1.4, 12: 1.3, 10: 1.2},
    "Education": {8: 1.5, 1: 1.4},
    "Health": {2: 1.2, 3: 1.2},
    "Bills": {4: 1.3, 5: 1.5, 6: 1.4},      # summer electricity
    "Food": {10: 1.1, 11: 1.1, 12: 1.15},
}
MONTHLY_TREND = 0.006            # ~0.6% "inflation" per month

ALIASES = {
    "Amazon": ["AMZN MKTP", "Amazon Pay India", "Amazon Seller Services"],
    "Swiggy": ["Bundl Technologies", "Swiggy Instamart"],
    "Zomato": ["Zomato Ltd", "Zomato Gold"],
    "Uber": ["Uber India Systems", "Uber Trip"],
    "Netflix": ["Netflix.com"],
    "Flipkart": ["Flipkart Internet", "FKRT Retail"],
}
WALLETS = ["PAYTM", "PHONEPE", "GPAY", "Amazon Pay", "BHIM UPI"]   # ambiguous descriptors
P_ALIAS = 0.30
P_WALLET = 0.08                  # description gives no hint about the category

TEMPLATES = [
    "UPI/{m}/{ref}/PAY",
    "{M}*{ref}",
    "POS {m} {city}",
    "{m} ORDER {ref}",
    "NEFT-{ref}-{m}",
    "IMPS/{ref}/{m}",
    "UPI-{m}-{ref}@okaxis",
]
CITIES = ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Delhi", "Mumbai"]
FALLBACK_NAMES = ["Rahul Sharma", "Priya Nair", "Amit Verma", "Sneha Reddy", "Karthik Iyer",
                  "Ananya Das", "Rohan Gupta", "Pooja Singh", "Vikram Rao", "Neha Joshi"]

DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y"]
INCOME_AMOUNT = 38000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _NameSource:
    """Random person names (Faker if installed, else a small fallback list)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.fake = None
        if Faker is not None:
            Faker.seed(seed)
            self.fake = Faker("en_IN")

    def name(self) -> str:
        if self.fake is not None:
            return self.fake.name()
        return str(self.rng.choice(FALLBACK_NAMES))


def make_description(merchant: str, rng: np.random.Generator) -> str:
    """Render a merchant name into a cryptic bank-statement string."""
    ref = int(rng.integers(10_000, 999_999))
    template = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
    return template.format(
        m=merchant,
        M=merchant.upper().replace(" ", ""),
        ref=ref,
        city=CITIES[int(rng.integers(len(CITIES)))],
    )


def _day_weights(month_start: pd.Timestamp, weekend_boost: float) -> np.ndarray:
    days = pd.date_range(month_start, periods=month_start.days_in_month)
    w = np.where(days.dayofweek >= 5, weekend_boost, 1.0)
    return w / w.sum()


def _pick_merchant(category: str, spec: dict, rng, names: _NameSource) -> str:
    if rng.random() < P_WALLET:
        return WALLETS[int(rng.integers(len(WALLETS)))]
    if category == "Transfers":
        return names.name()
    merchant = spec["merchants"][int(rng.integers(len(spec["merchants"])))]
    if merchant in ALIASES and rng.random() < P_ALIAS:
        options = ALIASES[merchant]
        merchant = options[int(rng.integers(len(options)))]
    return merchant


# --------------------------------------------------------------------------- #
# Stage 1a: clean ground-truth transactions
# --------------------------------------------------------------------------- #
def generate_clean(months: int, end: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = _NameSource(seed)
    month_starts = pd.date_range(end=end.replace(day=1), periods=months, freq="MS")
    rows: list[dict] = []

    for i, ms in enumerate(month_starts):
        trend = (1 + MONTHLY_TREND) ** i

        # Variable spending
        for cat, spec in CATEGORY_SPEC.items():
            n = int(rng.poisson(spec["rate"]))
            if n == 0:
                continue
            factor = SEASONALITY.get(cat, {}).get(ms.month, 1.0) * trend
            days = rng.choice(np.arange(1, ms.days_in_month + 1), size=n,
                              p=_day_weights(ms, spec["weekend_boost"]))
            for d in days:
                amount = max(10.0, rng.lognormal(spec["mu"], spec["sigma"]) * factor)
                merchant = _pick_merchant(cat, spec, rng, names)
                rows.append(dict(
                    date=ms + pd.Timedelta(days=int(d) - 1),
                    description=make_description(merchant, rng),
                    amount=round(float(amount), 2), type="debit", category=cat))

        # Recurring fixed items
        for cat, merchant, base, (d0, d1), jitter in FIXED_ITEMS:
            factor = SEASONALITY.get(cat, {}).get(ms.month, 1.0) if jitter else 1.0
            amount = base * factor * (1 + rng.uniform(-jitter, jitter)) * trend ** 0.5
            rows.append(dict(
                date=ms + pd.Timedelta(days=int(rng.integers(d0, d1 + 1)) - 1),
                description=make_description(merchant, rng),
                amount=round(float(amount), 2), type="debit", category=cat))

        # Rare one-off big purchases (gives EDA / anomaly detection something to find)
        if rng.random() < 0.06:
            merchant = ["Croma", "Amazon", "Apple Store", "Flipkart"][int(rng.integers(4))]
            rows.append(dict(
                date=ms + pd.Timedelta(days=int(rng.integers(0, ms.days_in_month))),
                description=make_description(merchant, rng),
                amount=round(float(rng.uniform(15_000, 45_000)), 2),
                type="debit", category="Shopping"))

        # Income (credits)
        income_templates = ["UPI/Dad/{ref}/CREDIT", "NEFT-{ref}-MONTHLY ALLOWANCE", "IMPS/{ref}/PARENT"]
        ref = int(rng.integers(10_000, 999_999))
        rows.append(dict(
            date=ms + pd.Timedelta(days=int(rng.integers(0, 3))),
            description=income_templates[int(rng.integers(3))].format(ref=ref),
            amount=round(INCOME_AMOUNT * (1 + rng.uniform(-0.05, 0.05)), 2),
            type="credit", category="Income"))
        if rng.random() < 0.30:
            rows.append(dict(
                date=ms + pd.Timedelta(days=int(rng.integers(0, ms.days_in_month))),
                description=f"UPI/CLIENT PAYMENT/{int(rng.integers(10_000, 999_999))}/CREDIT",
                amount=round(float(rng.lognormal(8.0, 0.4)), 2),
                type="credit", category="Income"))

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:05d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------- #
# Stage 1b: make it messy
# --------------------------------------------------------------------------- #
def _format_amount(a: float, rng: np.random.Generator) -> str:
    style = int(rng.integers(5))
    return [f"₹{a:,.2f}", f"Rs. {a:.2f}", f"{a:,.2f}", f"INR {a:,.2f}", f"{a:.2f}"][style]


def _mess_description(text: str, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.20:
        return text.upper()
    if r < 0.35:
        return text.lower()
    if r < 0.50:
        return f"  {text}  "
    return text


def inject_noise(clean: pd.DataFrame, unlabelled_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    raw = clean.copy()

    # Inconsistent type labels
    debit_variants, credit_variants = ["debit", "Debit", "DEBIT", "DR"], ["credit", "Credit", "CREDIT", "CR"]
    raw["type"] = [
        (debit_variants if t == "debit" else credit_variants)[int(rng.integers(4))]
        for t in raw["type"]
    ]

    # Mixed date formats
    fmts = rng.integers(len(DATE_FORMATS), size=len(raw))
    raw["date"] = [d.strftime(DATE_FORMATS[f]) for d, f in zip(raw["date"], fmts)]

    # Messy descriptions and amounts-as-strings
    raw["description"] = [_mess_description(t, rng) for t in raw["description"]]
    raw["amount"] = [_format_amount(a, rng) for a in raw["amount"]]

    # Inconsistent category casing
    raw["category"] = [
        c.upper() if rng.random() < 0.03 else (c.lower() if rng.random() < 0.03 else c)
        for c in raw["category"]
    ]

    # Missing values
    raw.loc[rng.random(len(raw)) < 0.02, "amount"] = np.nan
    raw.loc[rng.random(len(raw)) < 0.01, "description"] = np.nan

    # Blank out labels on a share of debit rows -> "unlabelled" data for the ML task
    blank = (clean["type"] == "debit") & (rng.random(len(raw)) < unlabelled_frac)
    raw.loc[blank, "category"] = np.nan

    # Exact duplicate rows (~3%), kept next to their originals
    dupes = raw.sample(frac=0.03, random_state=seed)
    raw = pd.concat([raw, dupes]).sort_index(kind="stable").reset_index(drop=True)
    return raw


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic expense data")
    parser.add_argument("--months", type=int, default=36, help="months of history")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--unlabelled-frac", type=float, default=0.25,
                        help="share of debit rows whose category is blanked out")
    args = parser.parse_args(argv)

    ensure_dirs()
    # last day of the previous (complete) month
    end = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)

    clean = generate_clean(args.months, end, args.seed)
    raw = inject_noise(clean, args.unlabelled_frac, args.seed)

    raw.to_csv(RAW_CSV, index=False, encoding="utf-8")
    clean[["txn_id", "type", "category"]].rename(columns={"category": "true_category"}) \
        .to_csv(GROUND_TRUTH_CSV, index=False, encoding="utf-8")

    log.info("Generated %d clean rows -> %d raw rows (with duplicates)", len(clean), len(raw))
    log.info("Date range: %s to %s", clean["date"].min().date(), clean["date"].max().date())
    log.info("Saved %s", RAW_CSV)


if __name__ == "__main__":
    main()