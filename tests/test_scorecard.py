import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")


def test_grade_stream_keys_and_ranges():
    from scorecard import grade_stream
    idx = pd.bdate_range("2018-01-01", periods=1500)
    # deterministic mild-positive-drift series
    r = pd.Series(0.0004 + 0.01 * np.sin(np.arange(1500) / 7.0), index=idx)
    c = grade_stream(r, "synthetic", n_trials=19)
    assert 0.0 <= c["psr"] <= 1.0
    assert 0.0 <= c["dsr"] <= 1.0
    assert c["mintrl_years"] > 0
    assert {"sharpe_ann", "drawdown", "min_backtest_years"}.issubset(c)
    assert c["drawdown"]["max_dd"] >= 0
