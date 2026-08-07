import numpy as np
import pandas as pd
import pytest
from qmeta.cv.purged_kfold import PurgedKFold


def test_no_leakage_overlapping_labels():
    n = 20
    start = np.arange(n)
    end = start + 2                      # each label spans 2 bars -> overlaps neighbors
    t1 = pd.Series(end, index=start)
    pk = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0)
    seen_any = False
    for tr, te in pk.split(np.zeros((n, 1))):
        seen_any = True
        assert len(set(tr) & set(te)) == 0
        t0 = t1.index[te[0]]
        test_end = t1.iloc[te].max()
        for k in tr:
            s, e = t1.index[k], t1.iloc[k]
            assert e <= t0 or s >= test_end   # train span never overlaps the test span
    assert seen_any


def test_embargo_does_not_increase_train():
    n = 30
    t1 = pd.Series(np.arange(n) + 1, index=np.arange(n))
    a = sum(len(tr) for tr, _ in PurgedKFold(5, t1, 0.0).split(np.zeros((n, 1))))
    b = sum(len(tr) for tr, _ in PurgedKFold(5, t1, 0.2).split(np.zeros((n, 1))))
    assert b <= a


def test_requires_sorted_t1():
    with pytest.raises(ValueError):
        PurgedKFold(3, pd.Series([1, 2], index=[5, 1]))
