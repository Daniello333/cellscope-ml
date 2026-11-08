"""Fold-local variance filtering and ANOVA feature ranking.

Every fitted statistic uses training samples only. get_support() indexes the
*original* input gene columns, preserving their names for downstream reports.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.utils.validation import check_is_fitted


class NonconstantKBest(BaseEstimator, TransformerMixin):
    """Remove zero-variance genes within fit(), then select k by ANOVA F-score.

    Raising if fewer than k informative candidates remain prevents silently
    including constant genes and keeps feature-budget comparisons honest.
    """

    def __init__(self, k=50):
        self.k = k

    def fit(self, X, y):
        if not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError("k must be a positive integer")
        self.n_features_in_ = X.shape[1]
        self.variance_ = VarianceThreshold(threshold=0.0).fit(X)
        X_valid = self.variance_.transform(X)
        if X_valid.shape[1] < self.k:
            raise ValueError(f"Only {X_valid.shape[1]} nonconstant genes on this training fold; "
                             f"cannot select {self.k}. Reduce --n-genes.")
        self.selector_ = SelectKBest(score_func=f_classif, k=self.k).fit(X_valid, y)
        original_kept = np.flatnonzero(self.variance_.get_support())
        self.selected_indices_ = original_kept[self.selector_.get_support()]
        self.scores_ = np.full(self.n_features_in_, -np.inf, dtype=float)
        self.scores_[original_kept] = np.nan_to_num(self.selector_.scores_, nan=-np.inf)
        self.pvalues_ = np.full(self.n_features_in_, np.nan, dtype=float)
        self.pvalues_[original_kept] = self.selector_.pvalues_
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction input has a different number of genes")
        return X[:, self.selected_indices_]

    def get_support(self):
        check_is_fitted(self, "selected_indices_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask
