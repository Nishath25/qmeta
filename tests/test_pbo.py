import numpy as np
import pytest
from qmeta.selection.trials import probability_of_backtest_overfitting as pbo


def test_pbo_no_overfit_when_one_column_dominates():
    # One column has a real positive mean AND real variance in every block;
    # the others are genuinely varying but centered near zero.
    T = 480
    t = np.arange(T)
    dom = 0.010 + 0.003 * np.sin(t / 5.0)      # mean ~0.01, sd>0, best in every block
    o1 = 0.004 * np.sin(t / 2.0)
    o2 = 0.004 * np.cos(t / 2.5)
    o3 = 0.003 * np.sin(t / 1.7 + 1.0)
    R = np.column_stack([dom, o1, o2, o3])
    out = pbo(R, n_splits=8)
    assert out["pbo"] < 0.10


def test_pbo_perfect_overfit_when_ranking_flips():
    # Both columns have real variance (noise term) but the in-sample winner is
    # always the out-of-sample loser -> the IS->OOS rank genuinely flips.
    half = 120
    t = np.arange(half)
    noise = 0.002 * np.sin(t / 2.0)
    A = np.concatenate([0.02 + noise, -0.02 + noise])
    B = np.concatenate([-0.02 + noise, 0.02 + noise])
    R = np.column_stack([A, B])
    out = pbo(R, n_splits=2)
    assert out["pbo"] == 1.0


def test_pbo_constant_column_is_neutral_not_inf():
    # A truly constant column must not crash or become spuriously best/worst.
    T = 240
    t = np.arange(T)
    good = 0.008 + 0.003 * np.sin(t / 4.0)
    flat = np.full(T, 0.02)          # zero variance
    noisy = 0.004 * np.cos(t / 3.0)
    out = pbo(np.column_stack([good, flat, noisy]), n_splits=6)
    assert 0.0 <= out["pbo"] <= 1.0  # runs cleanly, no inf/NaN blowup


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pbo(np.zeros((10, 1)), n_splits=2)   # need >=2 columns
    with pytest.raises(ValueError):
        pbo(np.zeros((10, 3)), n_splits=3)   # n_splits must be even
