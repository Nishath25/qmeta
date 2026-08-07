from pathlib import Path
import sys
import pytest
import pandas as pd

pytestmark = pytest.mark.skipif(
    not Path(r"C:\Users\madas\orb-strategy\data\raw_har_trades_atr3.parquet").exists(),
    reason="strategy data files not present",
)
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")


def test_orb_stream_shape():
    from load_streams import load_orb
    s = load_orb()
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) > 1000
    assert abs(s.mean()) < 0.05  # sane daily return scale


def test_orb_matrix_columns():
    from load_streams import load_orb_ticker_matrix
    m = load_orb_ticker_matrix()
    assert m.shape[1] >= 2 and m.shape[0] > 500
