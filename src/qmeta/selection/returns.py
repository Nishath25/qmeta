"""Shared return-series helpers for the selection toolkit."""
import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurtosis


def clean(r) -> np.ndarray:
    """Return r as a float ndarray with NaNs dropped."""
    a = np.asarray(r, dtype=float)
    return a[~np.isnan(a)]


def sharpe_per_obs(r) -> float:
    """Per-observation Sharpe = mean/std (ddof=1). 0.0 if degenerate."""
    a = clean(r)
    if len(a) < 2:
        return 0.0
    sd = a.std(ddof=1)
    return float(a.mean() / sd) if sd > 0 else 0.0


def moments(r):
    """(skewness, non-excess kurtosis). Gaussian -> (0.0, 3.0)."""
    a = clean(r)
    if len(a) < 2:
        return 0.0, 3.0
    return float(_skew(a)), float(_kurtosis(a, fisher=False))
