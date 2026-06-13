"""utils.py — fonctions communes pour le chapitre AML Stacking & Blending."""

from __future__ import annotations
import random, time, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

UCI_RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
UCI_WHITE_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    test_size: float = 0.25
    n_splits: int = 5
    n_seeds: int = 30
    output_dir: Path = Path("outputs")

def configure_environment(seed: int = 42) -> None:
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)

def load_wine_quality(color: str = "red", binary: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    url = UCI_RED_WINE_URL if color == "red" else UCI_WHITE_WINE_URL
    df = pd.read_csv(url, sep=";")
    X = df.drop(columns=["quality"])
    y_raw = df["quality"]
    y = (y_raw >= 6).astype(int) if binary else y_raw
    meta = {"name": f"UCI Wine Quality ({color})", "source": url, "target": "quality >= 6" if binary else "quality"}
    return X, y, meta

def load_pima_diabetes() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    cols = ["pregnancies","glucose","blood_pressure","skin_thickness","insulin","bmi","diabetes_pedigree","age","outcome"]
    df = pd.read_csv(PIMA_URL, names=cols)
    return df.drop(columns=["outcome"]), df["outcome"].astype(int), {"name": "Pima Indians Diabetes", "source": PIMA_URL}

def stratified_split(X, y, seed=42, test_size=0.25):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def timed_fit(model, X_train, y_train) -> float:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - start
