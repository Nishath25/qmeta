import math
import numpy as np
from qmeta.selection.drawdown import (
    max_drawdown_quantile, max_time_under_water, triple_penance, from_returns,
)


def test_triple_penance_iid_worked_example():
    # Bailey & LdP PM1: annual mean $10m, annual std $10m, 12 bets/yr, 95% conf.
    mu = 10e6 / 12.0                # per-bet mean in dollars
    sigma = 10e6 / math.sqrt(12.0)  # per-bet std in dollars (annual var = 12 * per-bet var)
    tp = triple_penance(mu, sigma, prob=0.95, phi=0.0)
    assert abs(tp["max_dd"] - 6_763_858.64) / 6_763_858.64 < 2e-3   # $6.76m
    assert abs(tp["penance_ratio"] - 3.0) < 1e-9
    assert abs(tp["max_tuw"] - 32.4665) < 0.05                       # ~32.5 months
    assert abs(tp["time_to_maxdd"] - 8.1166) < 0.05                  # ~8.1 months


def test_penance_exactly_three_any_inputs():
    for mu, sigma, prob in [(0.001, 0.01, 0.95), (0.02, 0.3, 0.99), (0.5, 2.0, 0.90)]:
        tp = triple_penance(mu, sigma, prob=prob, phi=0.0)
        assert abs(tp["penance_ratio"] - 3.0) < 1e-9


def test_maxdd_monotonic():
    base = max_drawdown_quantile(0.001, 0.01, 0.95)
    assert max_drawdown_quantile(0.001, 0.02, 0.95) > base   # more vol -> deeper
    assert max_drawdown_quantile(0.002, 0.01, 0.95) < base   # more drift -> shallower
    assert max_drawdown_quantile(0.001, 0.01, 0.99) > base   # higher conf -> deeper


def test_ar1_reduces_to_iid_as_phi_small():
    mu, sigma = 0.001, 0.01
    iid = triple_penance(mu, sigma, 0.95, phi=0.0)
    near = triple_penance(mu, sigma, 0.95, phi=1e-6)
    assert abs(near["max_dd"] - iid["max_dd"]) / iid["max_dd"] < 0.02
    assert abs(near["max_tuw"] - iid["max_tuw"]) / iid["max_tuw"] < 0.02


def test_ar1_penance_below_three_for_positive_phi():
    tp = triple_penance(0.001, 0.01, 0.95, phi=0.4)
    assert tp["penance_ratio"] < 3.0


def test_from_returns_runs():
    idx = np.arange(2000)
    r = 0.0005 + 0.01 * np.sin(idx / 9.0)
    out = from_returns(r, prob=0.95, ppy=252)
    assert out["max_dd"] > 0 and out["max_tuw_years"] > 0
    assert "phi" in out
