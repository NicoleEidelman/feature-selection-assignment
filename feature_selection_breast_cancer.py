from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple
RANDOM_STATE = 42
TEST_SIZE = 0.2
K = 10  # number of features to select

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    ds = load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    # The sklearn dataset has no missing values or categoricals, but we show checks:
    assert not X.isnull().any().any(), "Dataset contains missing values."
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

def baseline_logreg(X_train, X_test, y_train, y_test) -> float:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

def filter_anova_kbest(X_train, X_test, y_train, y_test, k: int) -> Tuple[float, List[str]]:
    # scale first (ANOVA f_classif is sensitive to scale differences)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kbest", SelectKBest(score_func=f_classif, k=k)),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # get selected feature mask from fitted SelectKBest
    kbest = pipe.named_steps["kbest"]
    selected_mask = kbest.get_support()
    selected_features = list(X_train.columns[selected_mask])
    return acc, selected_features

def embedded_rf_topk(X_train, X_test, y_train, y_test, k: int) -> Tuple[float, List[str]]:
    # Train RF on raw (unscaled) features to get importances
    rf = RandomForestClassifier(n_estimators=300, random_state=X_train.shape[1] + RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    importances.to_csv("rf_feature_importances.csv", index=True)
    top_features = list(importances.head(k).index)

    # Retrain LR on the selected features (scaled)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train[top_features], y_train)
    y_pred = pipe.predict(X_test[top_features])
    acc = accuracy_score(y_test, y_pred)
    return acc, top_features

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline_acc = baseline_logreg(X_train, X_test, y_train, y_test)
    print(f"[Baseline] Logistic Regression on all {X.shape[1]} features — Test Accuracy: {baseline_acc:.4f}")

    filter_acc, filter_feats = filter_anova_kbest(X_train, X_test, y_train, y_test, K)
    print(f"[Filter-ANOVA] k={K} — Test Accuracy: {filter_acc:.4f}")
    print("Selected features (ANOVA):", filter_feats)

    embedded_acc, embedded_feats = embedded_rf_topk(X_train, X_test, y_train, y_test, K)
    print(f"[Embedded-RF] top-{K} — Test Accuracy: {embedded_acc:.4f}")
    print("Selected features (RF):", embedded_feats)

    # Simple comparison summary
    results = pd.DataFrame({
        "Setting": ["Baseline (All)", f"Filter-ANOVA (k={K})", f"Embedded-RF (top-{K})"],
        "Test Accuracy": [baseline_acc, filter_acc, embedded_acc],
        "Num Features": [X.shape[1], len(filter_feats), len(embedded_feats)]
    })
    print("\n=== Summary ===")
    print(results.to_string(index=False))

    # Save summary
    results.to_csv("results_summary.csv", index=False)
    print("\nSaved: results_summary.csv, rf_feature_importances.csv")

if __name__ == "__main__":
    main()