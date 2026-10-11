# Architecture

## 1. Pipeline Overview

```
 generate_data.py          preprocess.py            features.py
┌────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│ Synthetic raw  │ ───► │ Clean + type    │ ───► │ Text features +  │
│ transactions   │      │ normalize text  │      │ monthly aggregates│
└───────┬────────┘      └────────┬────────┘      └────────┬─────────┘
        │                        │                        │
 data/raw/*.csv        data/processed/*.csv                 │
                                 │                          │
                                 ▼                          ▼
                        ┌─────────────────┐     ┌────────────────────┐
                        │ 01_eda.ipynb    │     │ ML tasks           │
                        │ (insights,      │     │                    │
                        │  figures)       │     │ A) train_categorizer│
                        └─────────────────┘     │ B) forecast.py     │
                                                └─────────┬──────────┘
                                                          │
                                              models/*.joblib + metrics
```

## 2. Stages

### Stage 1: Data Generation (`src/generate_data.py`)
- Builds ~12-24 months of transactions from merchant templates per category (Food, Transport, Shopping, Bills, Entertainment, Health, Rent, etc.).
- Injects realistic noise: UPI-style descriptions, random reference numbers, case inconsistencies, duplicate rows, missing values, mixed date formats, amounts stored as strings (`"₹1,250.00"`).
- Output: `data/raw/transactions_raw.csv` (columns: `date`, `description`, `amount`, `type`, `category`).
- A fraction of `category` is blanked out to simulate **unlabelled** data.

### Stage 2: Preprocessing (`src/preprocess.py`)
- Parse dates, strip currency symbols, cast amounts to float.
- Drop exact duplicates, handle nulls, separate debit/credit.
- Normalize descriptions: lowercase, remove digits and reference IDs, strip punctuation, collapse whitespace -> `description_clean`.
- Output: `data/processed/transactions_clean.csv`.

### Stage 3: Feature Engineering (`src/features.py`)
Two independent feature sets:

| Task | Features |
|---|---|
| Categorizer | `description_clean` -> TF-IDF (word 1-2 grams + char 2-5 grams), optional amount bucket |
| Forecaster | Monthly aggregates: lag-1/2/3 spend, rolling mean, month-of-year, rent/fixed-cost flags |

### Stage 4: EDA (`notebooks/01_eda.ipynb`)
Reads the processed CSV only. Produces figures to `reports/figures/`. Findings from here guide feature choices (e.g. seasonality -> month features).

### Stage 5: ML

**A) Categorizer (`src/train_categorizer.py`)**
- Train on labelled rows; predict on unlabelled rows.
- `Pipeline(TfidfVectorizer / FeatureUnion -> LogisticRegression or LinearSVC)`.
- Stratified split, macro-F1, confusion matrix, confidence threshold for manual review.
- Saves `models/categorizer.joblib`.

**B) Forecaster (`src/forecast.py`)**
- Monthly (or monthly-per-category) time series -> supervised learning via lag features.
- Time-ordered split / `TimeSeriesSplit`; compare vs naive baseline.
- Models: Linear/Ridge -> RandomForest / GradientBoosting.
- Saves `models/forecaster.joblib`.

## 3. How the Files Connect

| File | Reads | Writes | Used by |
|---|---|---|---|
| `generate_data.py` | none | `data/raw/` | `preprocess.py` |
| `preprocess.py` | `data/raw/` | `data/processed/` | EDA, `features.py`, training scripts |
| `features.py` | `data/processed/` | (in-memory) | `train_categorizer.py`, `forecast.py` |
| `train_categorizer.py` | `data/processed/` | `models/`, metrics | notebooks, future app |
| `forecast.py` | `data/processed/` | `models/`, metrics | notebooks, future app |
| `utils.py` | none | none | all modules (paths, seeds, helpers) |

**Rule:** notebooks are for exploration and visualization; reusable logic lives in `src/` and is imported into notebooks (`from src.preprocess import clean`).

## 4. Conventions

- **Reproducibility:** fixed `RANDOM_STATE = 42` in `utils.py`.
- **Paths:** use `pathlib.Path`, defined once in `utils.py`.
- **No leakage:** fit vectorizers/scalers on training data only (inside the `Pipeline`).
- **Raw data is immutable:** all cleaning goes through code, never manual edits.
- **Testing:** `tests/` covers cleaning functions (amount parsing, date parsing, text normalization).