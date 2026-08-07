"""Baseline allocators HRP/NCO are judged against."""
import numpy as np
import pandas as pd


def _as_df(cov):
    return cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(np.asarray(cov, dtype=float))


def cov_to_corr(cov):
    cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0)


def equal_weights(n):
    return np.full(int(n), 1.0 / int(n))


def inverse_variance_weights(cov):
    df = _as_df(cov)
    ivp = 1.0 / np.diag(df.values)
    ivp = ivp / ivp.sum()
    return pd.Series(ivp, index=df.index)


def min_variance_weights(cov):
    """Global minimum-variance weights = Σ⁻¹1 / (1ᵀΣ⁻¹1) (pseudo-inverse for stability)."""
    df = _as_df(cov)
    inv = np.linalg.pinv(df.values)
    ones = np.ones(df.shape[0])
    w = inv @ ones
    s = w.sum()
    w = w / s if s != 0 else equal_weights(df.shape[0])
    return pd.Series(w, index=df.index)


def max_sharpe_weights(cov, mu):
    """Unconstrained max-Sharpe (tangency) weights ∝ Σ⁻¹μ, normalized to sum 1."""
    df = _as_df(cov)
    inv = np.linalg.pinv(df.values)
    w = inv @ np.asarray(mu, dtype=float)
    s = w.sum()
    w = w / s if s != 0 else equal_weights(df.shape[0])
    return pd.Series(w, index=df.index)
