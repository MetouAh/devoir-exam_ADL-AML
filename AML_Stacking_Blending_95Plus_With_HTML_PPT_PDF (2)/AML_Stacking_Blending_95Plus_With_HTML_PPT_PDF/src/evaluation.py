"""evaluation.py — métriques et visualisations."""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

def evaluate_classifier(model, X_test, y_test, model_name: str, train_time_sec: float | None = None) -> Dict:
    y_pred = model.predict(X_test)
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    if train_time_sec is not None:
        row["train_time_sec"] = train_time_sec
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if proba.shape[1] == 2:
                row["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
        except Exception:
            row["roc_auc"] = np.nan
    return row

def results_table(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)

def summarize_stability(df: pd.DataFrame, metric="f1_macro") -> pd.DataFrame:
    return (
        df.groupby("model")
        .agg(mean=(metric, "mean"), std=(metric, "std"), min=(metric, "min"), max=(metric, "max"),
             mean_accuracy=("accuracy", "mean"), mean_time=("train_time_sec", "mean"))
        .sort_values("mean", ascending=False)
    )

def plot_bar_scores(results: pd.DataFrame, metric="f1_macro", title="Comparaison des modèles") -> None:
    plt.figure(figsize=(12, 5))
    plt.bar(results["model"], results[metric])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_stability_boxplot(stability: pd.DataFrame, metric="f1_macro") -> None:
    order = stability.groupby("model")[metric].mean().sort_values(ascending=False).index.tolist()
    plt.figure(figsize=(12, 5))
    plt.boxplot([stability.loc[stability["model"] == m, metric].values for m in order], labels=order, showmeans=True)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"Distribution de {metric} sur 30 graines")
    plt.tight_layout()
    plt.show()

def plot_confusion(model, X_test, y_test, title="Matrice de confusion") -> None:
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc(model, X_test, y_test, title="Courbe ROC") -> None:
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.tight_layout()
    plt.show()
