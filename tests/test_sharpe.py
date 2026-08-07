import math
from scipy.stats import norm
from qmeta.selection.sharpe import (
    annualize_sr, deannualize_sr,
    probabilistic_sharpe_ratio, min_track_record_length,
    psr_from_returns, mintrl_from_returns,
)


def test_annualization_roundtrip():
    assert abs(annualize_sr(deannualize_sr(0.95)) - 0.95) < 1e-12
    assert abs(annualize_sr(0.06) - 0.06 * math.sqrt(252)) < 1e-12


def test_psr_gaussian_reference():
    # sr=0.06 per-obs, T=253, skew=0, kurt=3, sr_star=0 -> ~0.8293
    psr = probabilistic_sharpe_ratio(0.06, 253, 0.0, 3.0, 0.0)
    denom = math.sqrt(1 + 0.5 * 0.06 ** 2)
    expected = norm.cdf(0.06 * math.sqrt(252) / denom)
    assert abs(psr - expected) < 1e-12
    assert abs(psr - 0.8293) < 1e-3


def test_psr_increases_with_n_and_sr():
    base = probabilistic_sharpe_ratio(0.06, 253, 0.0, 3.0)
    assert probabilistic_sharpe_ratio(0.06, 600, 0.0, 3.0) > base
    assert probabilistic_sharpe_ratio(0.09, 253, 0.0, 3.0) > base


def test_mintrl_reference():
    # sr=0.10, skew=0, kurt=3, prob=0.95 -> ~272.9 observations
    val = min_track_record_length(0.10, 0.0, 3.0, 0.0, 0.95)
    z = norm.ppf(0.95)
    expected = 1 + (1 + 0.5 * 0.10 ** 2) * (z / 0.10) ** 2
    assert abs(val - expected) < 1e-9
    assert abs(val - 272.9) < 0.5


def test_mintrl_infinite_when_sr_below_star():
    assert min_track_record_length(0.05, 0.0, 3.0, sr_star=0.05) == float("inf")


def test_from_returns_helpers_run():
    r = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.008]
    assert 0.0 <= psr_from_returns(r) <= 1.0
    assert mintrl_from_returns(r) > 0
