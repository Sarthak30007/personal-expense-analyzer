# Personal Expense Analyzer

An end-to-end data science project that cleans messy bank-style transaction data, explores spending behaviour, **auto-categorizes unlabelled transactions with NLP**, and **forecasts next month's spending with regression**.

## Problem Statement

Bank statements contain cryptic merchant strings (`UPI/ZOMATO/40213/PAY`, `AMZN MKTP*2K4`) and no reliable categories. This project:

1. Cleans and standardizes raw transaction exports.
2. Surfaces spending patterns through EDA.
3. Predicts the category of unlabelled transactions from their description text.
4. Forecasts next month's total (and per-category) spend to support budgeting.

## Features

| Module | What it does |
|---|---|
| Data generation | Synthetic, realistically dirty transaction data (UPI-style descriptions, duplicates, nulls, mixed date formats) |
| Preprocessing | Type fixing, deduplication, text normalization, outlier handling |
| EDA | Monthly trends, category breakdown, weekday/weekend patterns, top merchants, anomalies |
| ML 1: Categorizer | TF-IDF (word + char n-grams) + Logistic Regression / Linear SVM |
| ML 2: Forecaster | Lag-feature regression with time-aware validation |

## Tech Stack

- **Language:** Python 3.10+
- **Data:** pandas, numpy
- **Viz:** matplotlib, seaborn
- **ML:** scikit-learn, joblib
- **Synthetic data:** Faker
- **Testing:** pytest
- **Environment:** VS Code + venv + Jupyter

## Project Structure

```
personal-expense-analyzer/
├── data/
│   ├── raw/              # generated / original CSVs (never edited by hand)
│   ├── interim/          # partially processed data
│   └── processed/        # model-ready data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_categorizer.ipynb
│   └── 03_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_categorizer.py
│   ├── forecast.py
│   └── utils.py
├── models/               # serialized .joblib pipelines
├── reports/figures/      # exported charts
├── tests/
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
└── TODO.md
```

## Setup

```bash
# 1. Clone / create the project folder, then:
python -m venv .venv

# 2. Activate
#    Windows:
.venv\Scripts\activate
#    macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the kernel for notebooks (optional)
python -m ipykernel install --user --name expense-analyzer
```

**requirements.txt**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
faker
joblib
jupyter
ipykernel
pytest
```

In VS Code: install the **Python** and **Jupyter** extensions, then select the `.venv` interpreter (`Ctrl+Shift+P` -> *Python: Select Interpreter*).

## Usage

Run the pipeline from the project root:

```bash
python -m src.generate_data        # -> data/raw/transactions_raw.csv
python -m src.preprocess           # -> data/processed/transactions_clean.csv
python -m src.train_categorizer    # -> models/categorizer.joblib + metrics
python -m src.forecast             # -> models/forecaster.joblib + metrics
```

Then open the notebooks in `notebooks/` for EDA and model experiments.

## Results

Fill these in as you finish each phase.

| Task | Baseline | Final model | Metric |
|---|---|---|---|
| Category classification | _e.g. majority class_ | 0.069 | Macro F1 |
| Monthly spend forecast | _e.g. last-month naive_ | 4460.81 | MAE / MAPE |

## Key Design Decisions

- **Time-based split** for forecasting (no random shuffling) to prevent leakage.
- **Stratified split** + macro-F1 for the categorizer, since classes are imbalanced.
- **Confidence threshold** on predictions: low-confidence transactions are flagged for manual review.
- All preprocessing lives inside a scikit-learn `Pipeline` so training and inference stay consistent.

## Future Work

- Anomaly detection (Isolation Forest) for unusual transactions
- Streamlit dashboard for uploading a statement and viewing insights
- Active learning loop: user corrections feed back into retraining
- Support for real bank CSV/PDF statement formats

## License

MIT