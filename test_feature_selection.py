"""Minimal smoke tests for the selected CellScope ML feature-selection source."""
import numpy as np
import pytest
from feature_selection import NonconstantKBest


def test_excludes_constant_gene_and_keeps_original_column_order():
    X = np.array([[1., 0., 1.], [2., 0., 2.], [3., 0., 3.], [4., 0., 4.]])
    y = np.array([0, 0, 1, 1])
    model = NonconstantKBest(k=1).fit(X, y)
    assert model.get_support().shape == (3,)
    assert not model.get_support()[1]
    assert model.transform(X).shape == (4, 1)


def test_cannot_select_more_than_nonconstant_features():
    X = np.array([[1., 7.], [2., 7.], [3., 7.], [4., 7.]])
    with pytest.raises(ValueError, match="Only 1 nonconstant"):
        NonconstantKBest(k=2).fit(X, np.array([0, 0, 1, 1]))
