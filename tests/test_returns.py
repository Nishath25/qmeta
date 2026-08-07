import numpy as np
from qmeta.selection.returns import clean, sharpe_per_obs, moments


def test_clean_drops_nan():
    out = clean([1.0, np.nan, 2.0])
    assert list(out) == [1.0, 2.0]


def test_sharpe_per_obs_constant_series_is_zero():
    assert sharpe_per_obs([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_per_obs_known_value():
    r = [0.02, -0.01, 0.03, 0.00, 0.01]
    a = np.asarray(r)
    assert abs(sharpe_per_obs(r) - a.mean() / a.std(ddof=1)) < 1e-12


def test_moments_symmetric_zero_skew():
    sk, ku = moments([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(sk) < 1e-9
    assert ku > 0  # non-excess kurtosis is positive
