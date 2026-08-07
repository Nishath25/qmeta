import numpy as np
from qmeta.portfolio.baselines import (
    cov_to_corr, equal_weights, inverse_variance_weights,
    min_variance_weights, max_sharpe_weights,
)


def test_equal_weights():
    assert list(equal_weights(4)) == [0.25, 0.25, 0.25, 0.25]


def test_inverse_variance_diag():
    w = inverse_variance_weights(np.diag([4.0, 1.0]))
    assert abs(w.iloc[0] - 0.2) < 1e-9 and abs(w.iloc[1] - 0.8) < 1e-9


def test_min_variance_diag_equals_inverse_variance():
    w = min_variance_weights(np.diag([4.0, 1.0]))
    assert abs(w.iloc[0] - 0.2) < 1e-9 and abs(w.iloc[1] - 0.8) < 1e-9


def test_cov_to_corr():
    corr = cov_to_corr(np.array([[4.0, 1.0], [1.0, 1.0]]))
    assert abs(corr[0, 1] - 0.5) < 1e-9 and abs(corr[0, 0] - 1.0) < 1e-9


def test_max_sharpe_sums_to_one():
    w = max_sharpe_weights(np.array([[4.0, 0.0], [0.0, 1.0]]), np.array([0.1, 0.1]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w.iloc[0] - 0.2) < 1e-9 and abs(w.iloc[1] - 0.8) < 1e-9
