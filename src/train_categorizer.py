"""Stage 5A: NLP transaction categorizer.

Run:  python -m src.train_categorizer

What it does
  1. Splits debit rows into labelled / unlabelled.
  2. Stratified 80/20 split of the labelled rows.
  3. Baselines: majority class + keyword rules.
  4. Compares Logistic Regression, LinearSVC and Complement Naive Bayes.
  5. Tunes the logistic regression with GridSearchCV (macro-F1).
  6. Reports the confusion matrix, error analysis and confidence-threshold trade-off.
  7. Refits on ALL labelled data, predicts the unlabelled rows, and (because this
     is synthetic data) scores those predictions against the hidden ground truth.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features import AMOUNT_COL, TEXT_COL, build_preprocessor, load_clean, split_labelled
from src.preprocess import clean_description
from src.utils import (CATEGORIZED_CSV, CATEGORIZER_PATH, CONFIDENCE_THRESHOLD, FIGURES_DIR,
                       GROUND_TRUTH_CSV, MODELS_DIR, RANDOM_STATE, ensure_dirs, get_logger)

log = get_logger("categorizer")

KEYWORD_RULES = {
    "Food": ["zomato", "swiggy", "dominos", "mcdonalds", "starbucks", "bigbasket", "blinkit"],
    "Transport": ["uber", "ola", "rapido", "irctc", "redbus", "petrol", "metro"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "decathlon"],
    "Entertainment": ["netflix", "spotify", "bookmyshow", "hotstar", "pvr"],
    "Health": ["apollo", "pharmacy", "practo", "netmeds", "medplus"],
    "Bills": ["jio", "airtel", "electricity", "fibernet"],
    "Rent": ["rent"],
    "Education": ["udemy", "coursera", "unacademy", "bookstore"],
}


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def keyword_rule_predict(texts: pd.Series, default: str) -> np.ndarray:
    """First matching keyword wins; otherwise fall back to `default` (most common class)."""
    out = []
    for t in texts:
        label = default
        for cat, words in KEYWORD_RULES.items():
            if any(w in t for w in words):
                label = cat
                break
        out.append(label)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def make_pipeline(clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logreg": make_pipeline(LogisticRegression(max_iter=2000, C=5, class_weight="balanced")),
        "linear_svc": make_pipeline(LinearSVC(C=1.0, class_weight="balanced")),
        "complement_nb": make_pipeline(ComplementNB(alpha=0.3)),
    }


def _scores(y_true, y_pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4)}


# --------------------------------------------------------------------------- #
# Inference helper (use this from your future app)
# --------------------------------------------------------------------------- #
def predict_category(description: str, amount: float = 500.0, model_path=CATEGORIZER_PATH):
    """Predict (category, confidence) for one raw bank description."""
    model = joblib.load(model_path)
    X = pd.DataFrame({TEXT_COL: [clean_description(description)], AMOUNT_COL: [float(amount)]})
    proba = model.predict_proba(X)[0]
    i = int(proba.argmax())
    return str(model.classes_[i]), float(proba[i])


# --------------------------------------------------------------------------- #
def main() -> None:
    plt.switch_backend("Agg")      # script mode: save figures, never open windows
    ensure_dirs()
    df = load_clean()
    labelled, unlabelled = split_labelled(df)
    log.info("Labelled debits: %d | unlabelled debits: %d", len(labelled), len(unlabelled))
    log.info("Class distribution:\n%s", labelled["category"].value_counts().to_string())

    X, y = labelled[[TEXT_COL, AMOUNT_COL]], labelled["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results: dict = {}

    # ---- baselines ---------------------------------------------------------
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    results["baseline_majority"] = _scores(y_test, dummy.predict(X_test))
    results["baseline_keyword_rules"] = _scores(
        y_test, keyword_rule_predict(X_test[TEXT_COL], default=y_train.mode()[0]))

    # ---- compare models ----------------------------------------------------
    for name, pipe in candidate_models().items():
        pipe.fit(X_train, y_train)
        results[name] = _scores(y_test, pipe.predict(X_test))
    comparison = pd.DataFrame(results).T.sort_values("macro_f1", ascending=False)
    log.info("Model comparison on the held-out test set:\n%s", comparison.to_string())

    # ---- tune logistic regression -----------------------------------------
    grid = {
        "prep__word__ngram_range": [(1, 1), (1, 2)],
        "prep__char__ngram_range": [(2, 4), (2, 5)],
        "clf__C": [1, 5, 20],
    }
    search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
        grid, scoring="f1_macro", n_jobs=-1,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    search.fit(X_train, y_train)
    best = search.best_estimator_
    log.info("Best params: %s (CV macro-F1 = %.3f)", search.best_params_, search.best_score_)

    # ---- final evaluation on the test set ---------------------------------
    y_pred = best.predict(X_test)
    results["logreg_tuned"] = _scores(y_test, y_pred)
    log.info("Tuned model test scores: %s", results["logreg_tuned"])
    print("\n" + classification_report(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45, ax=ax,
                                            colorbar=False)
    ax.set_title("Categorizer: confusion matrix (test set)")
    fig.savefig(FIGURES_DIR / "categorizer_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- error analysis ----------------------------------------------------
    proba = best.predict_proba(X_test)
    conf = proba.max(axis=1)
    errors = X_test.assign(true=y_test.values, pred=y_pred, confidence=conf.round(3))
    errors = errors[errors["true"] != errors["pred"]].sort_values("confidence", ascending=False)
    log.info("Top confusing mistakes (confidently wrong):\n%s", errors.head(20).to_string())

    # ---- confidence threshold trade-off -----------------------------------
    keep = conf >= CONFIDENCE_THRESHOLD
    results["confidence_threshold"] = {
        "threshold": CONFIDENCE_THRESHOLD,
        "coverage": round(float(keep.mean()), 4),
        "accuracy_on_confident": round(float(accuracy_score(y_test[keep], y_pred[keep])), 4),
    }
    log.info("At threshold %.2f: coverage %.1f%%, accuracy on confident predictions %.1f%%",
             CONFIDENCE_THRESHOLD, 100 * keep.mean(),
             100 * results["confidence_threshold"]["accuracy_on_confident"])

    # ---- refit on ALL labelled data, label the unlabelled rows -------------
    final = clone(best).fit(X, y)
    joblib.dump(final, CATEGORIZER_PATH)
    log.info("Saved %s", CATEGORIZER_PATH)

    out = df.copy()
    out["predicted_category"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["confidence"] = np.nan
    if len(unlabelled):
        p = final.predict_proba(unlabelled[[TEXT_COL, AMOUNT_COL]])
        out.loc[unlabelled.index, "predicted_category"] = final.classes_[p.argmax(axis=1)]
        out.loc[unlabelled.index, "confidence"] = p.max(axis=1).round(4)
    out["needs_review"] = out["confidence"] < CONFIDENCE_THRESHOLD
    out["category_filled"] = out["category"].fillna(out["predicted_category"])
    out.to_csv(CATEGORIZED_CSV, index=False, encoding="utf-8")
    log.info("Saved %s (%d rows need manual review)", CATEGORIZED_CSV, int(out["needs_review"].sum()))

    # ---- check against hidden ground truth (only possible with synthetic data)
    if GROUND_TRUTH_CSV.exists() and len(unlabelled):
        truth = pd.read_csv(GROUND_TRUTH_CSV, encoding="utf-8")[["txn_id", "true_category"]]
        merged = out.loc[unlabelled.index].merge(truth, on="txn_id", how="left").dropna(
            subset=["true_category"])
        results["hidden_truth_on_unlabelled"] = _scores(
            merged["true_category"], merged["predicted_category"])
        log.info("Accuracy on the genuinely unlabelled rows (vs hidden truth): %s",
                 results["hidden_truth_on_unlabelled"])

    results["best_params"] = {k: str(v) for k, v in search.best_params_.items()}
    with open(MODELS_DIR / "categorizer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- demo --------------------------------------------------------------
    print("\nDemo predictions:")
    for text, amt in [("UPI/SWIGGY/778899/PAY", 320), ("UBER INDIA*88231", 210),
                      ("NETFLIX.COM 4417", 199), ("Apollo Pharmacy POS Pune", 640),
                      ("UPI/Rahul Verma/5512/PAY", 750)]:
        cat, c = predict_category(text, amt)
        print(f"  {text:<28} -> {cat:<14} ({c:.2f})")


if __name__ == "__main__":
    main()