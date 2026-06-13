"""stacking_from_scratch.py — CustomStackingClassifier avec Out-of-Fold."""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

class CustomStackingClassifier(BaseEstimator, ClassifierMixin):
    """StackingClassifier pédagogique et robuste.

    Chaque ligne de la matrice Z est prédite par un modèle de base qui n'a
    jamais vu cette observation, grâce à StratifiedKFold.
    """

    def __init__(
        self,
        base_estimators: List[Tuple[str, object]],
        meta_estimator: Optional[object] = None,
        n_splits: int = 5,
        stack_method: str = "predict_proba",
        random_state: int = 42,
    ):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.n_splits = n_splits
        self.stack_method = stack_method
        self.random_state = random_state

    def _base_prediction(self, estimator, X):
        if self.stack_method == "predict_proba":
            proba = estimator.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba.max(axis=1)
        if self.stack_method == "predict":
            return estimator.predict(X)
        raise ValueError("stack_method doit être 'predict_proba' ou 'predict'.")

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.names_ = [name for name, _ in self.base_estimators]
        self.classes_ = np.unique(y)

        Z = np.zeros((len(X), len(self.base_estimators)))
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        self.fold_estimators_ = []
        for fold_id, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            fold_models = []
            X_tr, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr = y.iloc[train_idx]

            for j, (name, estimator) in enumerate(self.base_estimators):
                model = clone(estimator)
                model.fit(X_tr, y_tr)
                Z[valid_idx, j] = self._base_prediction(model, X_valid)
                fold_models.append((name, model))
            self.fold_estimators_.append(fold_models)

        self.oof_meta_features_ = pd.DataFrame(Z, columns=self.names_)
        self.meta_estimator_ = clone(
            self.meta_estimator if self.meta_estimator is not None
            else LogisticRegression(max_iter=5000, random_state=self.random_state)
        )
        self.meta_estimator_.fit(self.oof_meta_features_, y)

        self.fitted_base_estimators_ = []
        for name, estimator in self.base_estimators:
            model = clone(estimator)
            model.fit(X, y)
            self.fitted_base_estimators_.append((name, model))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        Z = np.zeros((len(X), len(self.fitted_base_estimators_)))
        for j, (name, model) in enumerate(self.fitted_base_estimators_):
            Z[:, j] = self._base_prediction(model, X)
        return pd.DataFrame(Z, columns=self.names_)

    def predict(self, X):
        return self.meta_estimator_.predict(self.transform(X))

    def predict_proba(self, X):
        if not hasattr(self.meta_estimator_, "predict_proba"):
            raise AttributeError("Le meta_estimator ne fournit pas predict_proba.")
        return self.meta_estimator_.predict_proba(self.transform(X))
