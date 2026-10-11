"""Shared paths, constants and helpers used across the project."""
from __future__ import annotations

import logging
from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# File locations
RAW_CSV = DATA_RAW / "transactions_raw.csv"
GROUND_TRUTH_CSV = DATA_RAW / "ground_truth.csv"
INTERIM_CSV = DATA_INTERIM / "transactions_parsed.csv"
CLEAN_CSV = DATA_PROCESSED / "transactions_clean.csv"
CATEGORIZED_CSV = DATA_PROCESSED / "transactions_categorized.csv"
CATEGORIZER_PATH = MODELS_DIR / "categorizer.joblib"
FORECASTER_PATH = MODELS_DIR / "forecaster.joblib"

# Predictions below this confidence are flagged "needs review"
CONFIDENCE_THRESHOLD = 0.60


def ensure_dirs() -> None:
    """Create all output folders if they do not exist."""
    for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format (configured once)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)   # silence font/category chatter
    return logging.getLogger(name)