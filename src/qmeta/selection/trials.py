"""Multiple-testing corrections: the False Strategy theorem (E[max SR]),
the Deflated Sharpe Ratio (DSR), Minimum Backtest Length (MinBTL), and the
Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric CV.

Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio";
Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of Backtest
Overfitting"; and "Pseudo-Mathematics and Financial Charlatanism" (MinBTL).
"""
import math
from itertools import combinations

import numpy as np
from scipy.stats import norm

from qmeta.selection.sharpe import probabilistic_sharpe_ratio

EULER = 0.5772156649015329  # Euler-Mascheroni


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max Sharpe] across n_trials independent strategies under the null,
    where var_sr is the variance of the trial Sharpe estimates."""
    if n_trials < 2:
        return 0.0
    sd = math.sqrt(var_sr) if var_sr > 0 else 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sd * ((1.0 - EULER) * z1 + EULER * z2)


def deflated_sharpe_ratio(sr, n_obs, skew, kurt, n_trials, var_sr) -> float:
    """PSR with the benchmark set to E[max SR] under multiple testing."""
    sr_star = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_star=sr_star)


def min_backtest_length(n_trials: int, target_sr_ann: float) -> float:
    """Minimum backtest length (in YEARS) for target_sr_ann to be beyond what
    the best of n_trials would produce by luck. Uses E[max SR] with var_sr=1
    (the bracket term) since annualized trial-SR std ~ 1/sqrt(years)."""
    if target_sr_ann <= 0:
        return float("inf")
    bracket = expected_max_sharpe(n_trials, 1.0)
    return (bracket / target_sr_ann) ** 2


def _columns_sharpe(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=1)
    return np.where(sd > 0, mu / sd, -np.inf)


def probability_of_backtest_overfitting(returns_matrix, n_splits: int = 16) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    returns_matrix: 2D array-like (time x trials), one column per configuration.
    Partitions time into n_splits contiguous blocks; over every half/half
    in-sample/out-of-sample split, picks the IS-best column by Sharpe and finds
    its OOS rank omega in (0,1); PBO = P(logit(omega) <= 0)."""
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2D (time x trials)")
    T, N = R.shape
    if N < 2:
        raise ValueError("need >= 2 trial columns")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    logits = []
    for is_combo in combinations(range(n_splits), half):
        is_set = set(is_combo)
        is_rows = np.concatenate([blocks[b] for b in range(n_splits) if b in is_set])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])
        is_sr = _columns_sharpe(R[is_rows])
        oos_sr = _columns_sharpe(R[oos_rows])
        n_star = int(np.argmax(is_sr))
        order = np.argsort(oos_sr, kind="stable")  # ascending; last = best
        rank = int(np.where(order == n_star)[0][0]) + 1  # 1..N
        omega = rank / (N + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))
    logits = np.asarray(logits, dtype=float)
    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "logits": logits.tolist(),
        "n_partitions": int(len(logits)),
    }
