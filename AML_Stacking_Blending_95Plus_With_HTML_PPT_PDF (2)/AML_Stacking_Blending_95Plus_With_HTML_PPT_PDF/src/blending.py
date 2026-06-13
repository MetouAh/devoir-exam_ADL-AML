"""blending.py — CustomBlendingClassifier avec hold-out blend set."""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class CustomBlendingClassifier(BaseEstimator, ClassifierMixin):
    """BlendingClassifier pédagogique.

    Différence avec stacking :
    - stacking : méta-features par KFold OOF ;
    - blending : méta-features sur un hold-out blend set.
    """

    def __init__(
        self,
        base_estimators: List[Tuple[str, object]],
        meta_estimator: Optional[object] = None,
        blend_size: float = 0.25,
        random_state: int = 42,
    ):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.blend_size = blend_size
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        X_base, X_blend, y_base, y_blend = train_test_split(
            X, y, test_size=self.blend_size, random_state=self.random_state, stratify=y
        )

        self.names_ = [name for name, _ in self.base_estimators]
        self.fitted_base_estimators_ = []
        Z_blend = np.zeros((len(X_blend), len(self.base_estimators)))

        for j, (name, estimator) in enumerate(self.base_estimators):
            model = clone(estimator)
            model.fit(X_base, y_base)
            Z_blend[:, j] = model.predict_proba(X_blend)[:, 1]
            self.fitted_base_estimators_.append((name, model))

        self.blend_meta_features_ = pd.DataFrame(Z_blend, columns=self.names_)
        self.meta_estimator_ = clone(
            self.meta_estimator if self.meta_estimator is not None
            else LogisticRegression(max_iter=5000, random_state=self.random_state)
        )
        self.meta_estimator_.fit(self.blend_meta_features_, y_blend)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        Z = np.zeros((len(X), len(self.fitted_base_estimators_)))
        for j, (name, model) in enumerate(self.fitted_base_estimators_):
            Z[:, j] = model.predict_proba(X)[:, 1]
        return pd.DataFrame(Z, columns=self.names_)

    def predict(self, X):
        return self.meta_estimator_.predict(self.transform(X))

    def predict_proba(self, X):
        return self.meta_estimator_.predict_proba(self.transform(X))
