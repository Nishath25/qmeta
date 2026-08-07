"""Triple-Penance drawdown & time-under-water.

Bailey & Lopez de Prado (2015), "Stop-Outs Under Serial Correlation and the
'Triple Penance' Rule", Journal of Risk 18(2) / SSRN 2201302.

Cumulative PnL is modelled as pi_t ~ N(mu*t, sigma^2 * t) for i.i.d. Gaussian
bets (phi=0), or the AR(1) generalization
    pi_t = (1-phi)*mu + phi*pi_{t-1} + sigma*eps_t.
`mu`, `sigma` are PER-PERIOD; returned `time_to_maxdd` and `max_tuw` are in the
same period units. MaxDD is returned as a POSITIVE loss, floored at 0.
For the i.i.d. Gaussian case the penance ratio (recovery time / descent time)
equals 3 exactly, for any confidence and any Sharpe (TuW = 4 * t*).
"""
import math

import numpy as np
from scipy.stats import norm

from qmeta.selection.returns import clean


def _z(prob: float) -> float:
    # significance alpha = 1 - prob (confidence); z_alpha is negative for prob>0.5
    return norm.ppf(1.0 - prob)


def max_drawdown_quantile(mu, sigma, prob=0.95, phi=0.0) -> float:
    """MaxDD (positive loss) not exceeded with probability `prob`."""
    if mu <= 0:
        return float("inf")
    if phi == 0.0:
        z = _z(prob)
        return max(0.0, (z * sigma) ** 2 / (4.0 * mu))
    return _ar1_triple_penance(mu, sigma, phi, prob)[0]


def max_time_under_water(mu, sigma, prob=0.95, phi=0.0) -> float:
    """Max time under water (periods) at confidence `prob`."""
    if mu <= 0:
        return float("inf")
    if phi == 0.0:
        z = _z(prob)
        return (z * sigma / mu) ** 2
    return _ar1_triple_penance(mu, sigma, phi, prob)[2]


def triple_penance(mu, sigma, prob=0.95, phi=0.0) -> dict:
    """dict(max_dd, time_to_maxdd, max_tuw, penance_ratio). penance_ratio == 3
    exactly for the i.i.d. Gaussian case (TuW = 4 * t*)."""
    if phi == 0.0:
        if mu <= 0:
            return dict(max_dd=float("inf"), time_to_maxdd=float("inf"),
                        max_tuw=float("inf"), penance_ratio=float("nan"))
        z = _z(prob)
        max_dd = max(0.0, (z * sigma) ** 2 / (4.0 * mu))
        t_star = (z * sigma / (2.0 * mu)) ** 2
        tuw = (z * sigma / mu) ** 2
        pen = tuw / t_star - 1.0 if t_star > 0 else float("nan")
        return dict(max_dd=max_dd, time_to_maxdd=t_star, max_tuw=tuw, penance_ratio=pen)
    max_dd, t_star, tuw, pen = _ar1_triple_penance(mu, sigma, phi, prob)
    return dict(max_dd=max_dd, time_to_maxdd=t_star, max_tuw=tuw, penance_ratio=pen)


def _ar1_quantile_path(t, mu, sigma, phi, z, pi0=0.0):
    """The alpha-quantile of cumulative PnL at time t under AR(1) (Eq. 40-41)."""
    t = np.asarray(t, dtype=float)
    Et = (phi ** (t + 1) - phi) / (phi - 1.0) * (pi0 - mu) + mu * t
    Vt = sigma ** 2 / (phi - 1.0) ** 2 * (
        (phi ** (2 * (t + 1)) - 1.0) / (phi ** 2 - 1.0)
        - 2.0 * (phi ** (t + 1) - 1.0) / (phi - 1.0) + t + 1.0
    )
    return Et + z * np.sqrt(np.maximum(Vt, 0.0))


def _ar1_triple_penance(mu, sigma, phi, prob, pi0=0.0):
    """No closed form under AR(1): the alpha-quantile path is unimodal for mu>0,
    phi in (0,1), so locate the trough and the subsequent zero-crossing on a
    fine grid (Bailey & LdP use golden-section; a dense grid matches it)."""
    z = _z(prob)
    tuw_iid = (z * sigma / mu) ** 2
    t_max = max(50.0, 6.0 * tuw_iid)
    ts = np.linspace(1e-3, t_max, 40000)
    path = _ar1_quantile_path(ts, mu, sigma, phi, z, pi0)
    imin = int(np.argmin(path))
    t_star, min_val = float(ts[imin]), float(path[imin])
    max_dd = max(0.0, -min_val)
    tail = path[imin:]
    cross = np.where(tail >= 0.0)[0]
    tuw = float(ts[imin:][cross[0]]) if len(cross) else float("nan")
    pen = tuw / t_star - 1.0 if t_star > 0 else float("nan")
    return max_dd, t_star, tuw, pen


def estimate_ar1(r):
    """(phi, mu, sigma_innovation) fitted to a realized return series. sigma is
    the per-shock innovation std = sqrt(Var * (1 - phi^2)) (Section 9)."""
    a = clean(r)
    mu = float(a.mean())
    v = float(a.var(ddof=1))
    if len(a) < 3:
        return 0.0, mu, math.sqrt(v)
    x0, x1 = a[:-1] - mu, a[1:] - mu
    denom = float(np.sum(x0 * x0))
    phi = float(np.sum(x0 * x1) / denom) if denom > 0 else 0.0
    phi = max(-0.99, min(0.99, phi))
    sig = math.sqrt(max(0.0, v * (1.0 - phi ** 2)))
    return phi, mu, sig


def from_returns(r, prob=0.95, ar1=False, ppy=252) -> dict:
    """Estimate mu, sigma (and phi if ar1) from a realized return series and
    return the triple-penance dict, augmented with mu/sigma/phi and year-scaled
    times (assuming `ppy` periods/year)."""
    a = clean(r)
    if ar1:
        phi, mu, sigma = estimate_ar1(a)
    else:
        phi, mu, sigma = 0.0, float(a.mean()), float(a.std(ddof=1))
    out = triple_penance(mu, sigma, prob=prob, phi=phi)
    out.update(mu=mu, sigma=sigma, phi=phi,
               time_to_maxdd_years=out["time_to_maxdd"] / ppy,
               max_tuw_years=out["max_tuw"] / ppy)
    return out
