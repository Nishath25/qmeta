"""Strategy Approval Theorem / Sharpe-ratio Indifference Curve.
Lopez de Prado (2012), "The Strategy Approval Decision".

For an approved book (Sharpe S_a) and a candidate (Sharpe S_n) with correlation
rho, the max-Sharpe of the optimally-combined book is
    S_c = sqrt((S_a^2 - 2*rho*S_a*S_n + S_n^2) / (1 - rho^2)),
and adding the candidate raises the aggregate Sharpe iff S_n > rho * S_a.
All Sharpes may be annualized or per-obs as long as consistent.
"""
import math
import numpy as np


def combined_max_sharpe(sr_app: float, sr_new: float, corr: float) -> float:
    den = 1.0 - corr * corr
    if den <= 0:
        return float("inf")
    num = sr_app ** 2 - 2.0 * corr * sr_app * sr_new + sr_new ** 2
    return math.sqrt(max(0.0, num / den))


def improves_max_sharpe(sr_app: float, sr_new: float, corr: float) -> bool:
    return sr_new > corr * sr_app


def max_correlation_for_approval(sr_app: float, sr_new: float) -> float:
    if sr_app <= 0:
        return 1.0
    return min(1.0, sr_new / sr_app)


def indifference_curve(sr_app: float, level: float = None, n: int = 101):
    """Locus of (corr, sr_new) holding the combined max-Sharpe == `level`
    (default level = sr_app, i.e. the approval boundary). Returns
    (corr_array, sr_new_array); sr_new is NaN where no real solution exists."""
    if level is None:
        level = sr_app
    corrs = np.linspace(-0.99, 0.99, n)
    out = np.full(n, np.nan)
    for i, rho in enumerate(corrs):
        # solve sr_new^2 - 2*rho*sr_app*sr_new + (sr_app^2 - level^2*(1-rho^2)) = 0
        b = -2.0 * rho * sr_app
        c = sr_app ** 2 - level ** 2 * (1.0 - rho ** 2)
        disc = b * b - 4.0 * c
        if disc < 0:
            continue
        out[i] = (-b + math.sqrt(disc)) / 2.0  # larger root = candidate SR needed
    return corrs, out
