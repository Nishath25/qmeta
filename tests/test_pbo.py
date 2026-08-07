import numpy as np
import pytest
from qmeta.selection.trials import probability_of_backtest_overfitting as pbo


def test_pbo_no_overfit_when_one_column_dominates():
    # col 0 has a strong positive mean every period; others are flat noise-free
    T = 400
    t = np.arange(T)
    dominant = np.full(T, 0.02)
    others = np.column_stack([0.0001 * np.sin(t / (k + 2)) for k in range(3)])
    R = np.column_stack([dominant, others])
    out = pbo(R, n_splits=8)
    assert out["pbo"] < 0.05


def test_pbo_perfect_overfit_when_ranking_flips():
    # Two columns whose in-sample winner is always the out-of-sample loser.
    half = 100
    A = np.concatenate([np.full(half, 0.02), np.full(half, -0.02)])
    B = np.concatenate([np.full(half, -0.02), np.full(half, 0.02)])
    R = np.column_stack([A, B])
    out = pbo(R, n_splits=2)
    assert out["pbo"] == 1.0


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pbo(np.zeros((10, 1)), n_splits=2)   # need >=2 columns
    with pytest.raises(ValueError):
        pbo(np.zeros((10, 3)), n_splits=3)   # n_splits must be even
