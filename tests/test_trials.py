import math
from scipy.stats import norm
from qmeta.selection.trials import (
    expected_max_sharpe, deflated_sharpe_ratio, min_backtest_length,
)

EULER = 0.5772156649015329


def test_expected_max_sharpe_formula():
    n, var = 19, 0.5
    sd = math.sqrt(var)
    z1 = norm.ppf(1 - 1.0 / n)
    z2 = norm.ppf(1 - 1.0 / (n * math.e))
    expected = sd * ((1 - EULER) * z1 + EULER * z2)
    assert abs(expected_max_sharpe(n, var) - expected) < 1e-12


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0)
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_expected_max_sharpe_literal_anchor():
    # Standalone anchor (independent of gauntlet parity, which may skip):
    # N=19, var=1 -> bracket = (1-g)Z(1-1/19) + g*Z(1-1/(19e)) = 1.87802
    assert abs(expected_max_sharpe(19, 1.0) - 1.87802) < 1e-4


def test_dsr_is_psr_at_deflated_threshold():
    from qmeta.selection.sharpe import probabilistic_sharpe_ratio
    star = expected_max_sharpe(20, 1.0 / 2000)
    assert abs(
        deflated_sharpe_ratio(0.06, 2000, 0.0, 3.0, 20, 1.0 / 2000)
        - probabilistic_sharpe_ratio(0.06, 2000, 0.0, 3.0, star)
    ) < 1e-12


def test_min_backtest_length_reference():
    # 19 trials, target annual SR 0.95 -> ~3.9 years
    y = min_backtest_length(19, 0.95)
    assert 3.5 < y < 4.3
    assert min_backtest_length(100, 0.95) > y  # more trials -> longer needed
