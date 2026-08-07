"""Parity: qmeta's E[maxSR] and DSR must equal orb-strategy/gauntlet/cpcv.py.
Skips cleanly if orb-strategy (or its deps) is not importable."""
import sys
from pathlib import Path
import pytest

ORB = Path(r"C:\Users\madas\orb-strategy")


def _load_gauntlet():
    if str(ORB) not in sys.path:
        sys.path.insert(0, str(ORB))
    from gauntlet import cpcv  # needs strategy.paths + sklearn present
    return cpcv


def test_parity_expected_max_sharpe():
    try:
        cpcv = _load_gauntlet()
    except Exception as e:  # ImportError or missing deps
        pytest.skip(f"gauntlet not importable: {e}")
    from qmeta.selection.trials import expected_max_sharpe
    for n, v in [(10, 1.0), (19, 0.5), (100, 0.25)]:
        assert abs(expected_max_sharpe(n, v) - cpcv.expected_max_sharpe(n, v)) < 1e-12


def test_parity_deflated_sharpe():
    try:
        cpcv = _load_gauntlet()
    except Exception as e:
        pytest.skip(f"gauntlet not importable: {e}")
    from qmeta.selection.trials import deflated_sharpe_ratio
    args = (0.06, 2000, 0.1, 3.5, 20, 1.0 / 2000)
    assert abs(deflated_sharpe_ratio(*args) - cpcv.deflated_sharpe(*args)) < 1e-12
