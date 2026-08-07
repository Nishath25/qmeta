import numpy as np
import pandas as pd
from qmeta.metalabel.evaluate import stream_stats, compare


def test_stream_stats_keys():
    r = pd.Series(0.0005 + 0.01 * np.sin(np.arange(800) / 6.0))
    s = stream_stats(r)
    assert {"sharpe", "dsr", "maxdd", "vol", "n"}.issubset(s)
    assert s["maxdd"] <= 0


def test_compare_identical_streams_zero_delta():
    raw = pd.Series(np.full(500, 0.001) + 0.01 * np.sin(np.arange(500) / 5.0))
    c = compare(raw, raw.copy())
    assert abs(c["d_sharpe"]) < 1e-9 and abs(c["d_maxdd"]) < 1e-9
