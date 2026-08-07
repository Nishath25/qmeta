import numpy as np
from qmeta.labeling.triple_barrier import triple_barrier, meta_label, meta_labels_from_R


def test_pt_touch_first():
    prices = np.array([100, 101, 103.5, 99])  # long: +3.5% at idx 2
    out, idx, r = triple_barrier(prices, 0, pt=0.03, sl=0.05, max_h=3, side=1)
    assert out == "pt" and idx == 2 and r >= 0.03


def test_sl_touch_first():
    prices = np.array([100, 99, 94, 110])  # long: -6% by idx 2
    out, idx, r = triple_barrier(prices, 0, pt=0.10, sl=0.05, max_h=3, side=1)
    assert out == "sl" and idx == 2 and r <= -0.05


def test_time_barrier():
    prices = np.array([100, 100.5, 100.2, 100.8])
    out, idx, r = triple_barrier(prices, 0, pt=0.10, sl=0.10, max_h=3, side=1)
    assert out == "time" and idx == 3


def test_short_side_wins_on_drop():
    prices = np.array([100, 97, 110])  # short profits as price falls
    out, idx, r = triple_barrier(prices, 0, pt=0.02, sl=0.10, max_h=2, side=-1)
    assert out == "pt" and r >= 0.02


def test_meta_labels():
    assert meta_label(0.05, 1) == 1 and meta_label(-0.05, 1) == 0
    assert meta_label(-0.05, -1) == 1  # short + price down = win
    assert list(meta_labels_from_R([1.0, -1.0, 0.0, 3.0])) == [1, 0, 0, 1]
