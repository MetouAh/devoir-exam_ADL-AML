from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer

UCI_RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def load_wine_quality_binary():
    df = pd.read_csv(UCI_RED_WINE_URL, sep=";")
    X = df.drop(columns=["quality"])
    y = (df["quality"] >= 6).astype(int)
    return X, y, {"name": "Wine Quality UCI", "target": "quality >= 6", "source": UCI_RED_WINE_URL}


def load_pima_or_fallback():
    cols = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
    ]
    try:
        df = pd.read_csv(PIMA_URL, names=cols)
        X = df.drop(columns=["outcome"])
        y = df["outcome"].astype(int)
        meta = {"name": "Pima Indians Diabetes", "source": PIMA_URL}
        return X, y, meta
    except Exception:
        data = load_breast_cancer(as_frame=True)
        X = data.data
        y = data.target.astype(int)
        meta = {"name": "Breast Cancer Wisconsin fallback", "source": "sklearn.datasets.load_breast_cancer"}
        return X, y, meta
