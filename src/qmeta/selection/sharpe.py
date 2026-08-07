"""Probabilistic Sharpe Ratio (PSR) and Minimum Track Record Length (MinTRL).

Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
All formulas use the PER-OBSERVATION Sharpe `sr = mean/std` (not annualized)
and NON-EXCESS kurtosis (Gaussian = 3.0).
"""
import math
from scipy.stats import norm

from qmeta.selection.returns import clean, sharpe_per_obs, moments


def annualize_sr(sr: float, ppy: int = 252) -> float:
    return sr * math.sqrt(ppy)


def deannualize_sr(sr_ann: float, ppy: int = 252) -> float:
    return sr_ann / math.sqrt(ppy)


def probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_star=0.0) -> float:
    """P(true SR > sr_star) given estimation error and higher moments. In [0,1]."""
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    stat = (sr - sr_star) * math.sqrt(max(1.0, n_obs - 1.0)) / denom
    return float(norm.cdf(stat))


def min_track_record_length(sr, skew, kurt, sr_star=0.0, prob=0.95) -> float:
    """Number of observations needed for PSR(sr_star) to reach `prob`."""
    if sr <= sr_star:
        return float("inf")
    z = norm.ppf(prob)
    factor = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    return 1.0 + factor * (z / (sr - sr_star)) ** 2


def psr_from_returns(r, sr_star=0.0) -> float:
    a = clean(r)
    sk, ku = moments(a)
    return probabilistic_sharpe_ratio(sharpe_per_obs(a), len(a), sk, ku, sr_star)


def mintrl_from_returns(r, prob=0.95, sr_star=0.0) -> float:
    a = clean(r)
    sk, ku = moments(a)
    return min_track_record_length(sharpe_per_obs(a), sk, ku, sr_star, prob)
