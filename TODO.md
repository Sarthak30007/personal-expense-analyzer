# TODO: Personal Expense Analyzer

Work top to bottom. Don't start a phase until the previous one's **Done when** line is true.

---

## Phase 0: Setup

- [ ] Create project folder and folder structure (see README)
- [ ] `git init` and add a `.gitignore` (`.venv/`, `__pycache__/`, `data/raw/`, `*.joblib`, `.ipynb_checkpoints/`)
- [ ] Create and activate virtual environment
- [ ] Create `requirements.txt` and `pip install -r requirements.txt`
- [ ] Select `.venv` interpreter in VS Code
- [ ] Create empty `src/__init__.py` and `src/utils.py` (paths, `RANDOM_STATE = 42`)
- [ ] First commit

**Done when:** `python -c "import pandas, sklearn"` runs without error.

---

## Phase 1: Data Generation

- [ ] Define 8-10 categories (Food, Transport, Shopping, Bills, Rent, Entertainment, Health, Education, Transfers, Misc)
- [ ] Create a merchant dictionary per category (5-10 merchants each)
- [ ] Write description templates (`UPI/{merchant}/{ref}/PAY`, `{MERCHANT}*{ref}`, `POS {merchant}`)
- [ ] Generate dates across 18-24 months
- [ ] Generate realistic amounts per category (e.g. normal / lognormal distributions)
- [ ] Add recurring items (rent, subscriptions, phone bill) on fixed days
- [ ] Add seasonality (festival spikes, exam-month changes, etc.)
- [ ] Inject noise:
  - [ ] ~3% duplicate rows
  - [ ] ~2% missing amounts or descriptions
  - [ ] Mixed date formats
  - [ ] Amounts as strings with symbols/commas
  - [ ] Random casing and extra whitespace
- [ ] Blank out ~25% of `category` values to simulate unlabelled data (keep a hidden copy for evaluation)
- [ ] Save to `data/raw/transactions_raw.csv`
- [ ] Add a `__main__` block so `python -m src.generate_data` works

**Done when:** raw CSV has 2,000+ rows and looks visibly messy in `df.head()` / `df.info()`.

---

## Phase 2: Data Cleaning and Preprocessing

- [ ] Load raw data and inspect (`info()`, `describe()`, `isna().sum()`, `duplicated().sum()`)
- [ ] Write `parse_amount()` to strip symbols/commas and cast to float
- [ ] Write `parse_date()` to handle mixed formats (`pd.to_datetime(..., errors="coerce", dayfirst=True)`)
- [ ] Drop duplicates and rows with unparseable date/amount
- [ ] Decide and document a null-handling strategy per column
- [ ] Write `clean_description()`: lowercase, remove digits/refs, remove punctuation, collapse spaces
- [ ] Split debit vs credit (income) transactions
- [ ] Detect outliers (IQR or z-score) and decide: keep, cap, or flag
- [ ] Add derived columns: `year`, `month`, `day_of_week`, `is_weekend`, `year_month`
- [ ] Save to `data/processed/transactions_clean.csv`
- [ ] Write 3+ pytest tests (amount parsing, date parsing, text cleaning)

**Done when:** processed CSV has correct dtypes, zero duplicates, and `pytest` passes.

---

## Phase 3: Exploratory Data Analysis

Notebook: `notebooks/01_eda.ipynb`

- [ ] Summary statistics: total spend, average per month, number of transactions
- [ ] Monthly spending trend (line chart)
- [ ] Spend by category (bar + pie/donut)
- [ ] Category x month heatmap
- [ ] Weekday vs weekend spending comparison
- [ ] Day-of-month pattern (salary-day effects, bill dates)
- [ ] Top 10 merchants by total spend and by frequency
- [ ] Transaction amount distribution (histogram + boxplot per category)
- [ ] Correlation analysis between category monthly totals
- [ ] Identify recurring transactions
- [ ] Flag anomalous transactions (unusually large for their category)
- [ ] Write a "Key Insights" markdown cell with 5+ findings
- [ ] Export final figures to `reports/figures/`

**Done when:** you can describe the spending behaviour in 5 bullet points, backed by charts.

---

## Phase 4: ML Feature A, Transaction Categorizer (NLP)

Notebook: `notebooks/02_categorizer.ipynb` -> then move code into `src/train_categorizer.py`

- [ ] Separate labelled vs unlabelled rows
- [ ] Check class distribution (imbalance?)
- [ ] Stratified train/test split (80/20)
- [ ] Baseline: majority class and/or keyword-rule classifier
- [ ] Build `TfidfVectorizer` (word 1-2 grams)
- [ ] Add char n-grams (2-5, `analyzer="char_wb"`) and combine with `FeatureUnion`
- [ ] Train Logistic Regression -> record accuracy, macro-F1
- [ ] Train LinearSVC and/or Multinomial Naive Bayes -> compare
- [ ] Use `class_weight="balanced"` if classes are imbalanced
- [ ] Tune with `GridSearchCV` / `RandomizedSearchCV` (C, ngram_range, min_df)
- [ ] Plot confusion matrix and print `classification_report`
- [ ] Error analysis: inspect the 20 most confusing misclassifications
- [ ] Add confidence threshold (`predict_proba` max < 0.6 -> "Needs review")
- [ ] Predict categories for unlabelled rows and compare against the hidden ground truth
- [ ] Save pipeline with `joblib.dump` to `models/categorizer.joblib`
- [ ] Write `predict_category(description: str)` helper and test it on 5 new strings

**Done when:** the final model beats the baseline by a clear margin and the pipeline reloads and predicts from raw text.

---

## Phase 5: ML Feature B, Spend Forecasting (Regression)

Notebook: `notebooks/03_forecasting.ipynb` -> then move code into `src/forecast.py`

- [ ] Aggregate to monthly totals (overall first, per-category later)
- [ ] Plot the series; check trend and seasonality
- [ ] Baselines: last-month value, 3-month moving average
- [ ] Feature engineering:
  - [ ] Lag features (t-1, t-2, t-3)
  - [ ] Rolling mean / std (3-month)
  - [ ] Month-of-year (cyclical sin/cos encoding)
  - [ ] Fixed-cost total (rent, bills)
- [ ] Time-ordered train/test split (last 3-4 months as test)
- [ ] Train Linear / Ridge regression
- [ ] Train RandomForest / GradientBoosting
- [ ] Evaluate with MAE, RMSE, MAPE vs baselines
- [ ] Use `TimeSeriesSplit` for cross-validation
- [ ] Plot actual vs predicted
- [ ] Forecast next month's total spend
- [ ] (Stretch) Per-category forecasts and a budget-overrun alert
- [ ] Save model to `models/forecaster.joblib`

**Done when:** the model beats the naive baseline on MAE and produces a next-month prediction.

---

## Phase 6: Integration and Polish

- [ ] Write a single `run_pipeline.py` (or Makefile) that runs all stages
- [ ] Add docstrings and type hints to all functions in `src/`
- [ ] Add basic logging instead of `print`
- [ ] Format with `black`, lint with `ruff` or `flake8`
- [ ] Re-run everything from a fresh venv to confirm reproducibility
- [ ] Fill in the Results table in `README.md`
- [ ] Add 2-3 screenshots/figures to `README.md`
- [ ] Update `ARCHITECTURE.md` if the structure changed
- [ ] Final commit and push to GitHub

**Done when:** a stranger can clone the repo, follow the README, and reproduce your results.

---

## Phase 7: Stretch Goals

- [ ] Anomaly detection with Isolation Forest
- [ ] Streamlit app: upload CSV -> cleaned data, charts, categories, forecast
- [ ] Active learning: let users correct categories, then retrain
- [ ] Try a sentence-embedding model (e.g. `sentence-transformers`) vs TF-IDF
- [ ] Parse a real bank statement export (your own, anonymized)
- [ ] Write a short blog post or LinkedIn write-up of the project