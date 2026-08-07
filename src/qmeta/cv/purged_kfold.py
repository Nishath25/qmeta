"""Purged K-Fold cross-validation with embargo (Lopez de Prado, AFML ch. 7).

Standard k-fold leaks when labels span time: a training label whose horizon
overlaps the test window shares information with it. PurgedKFold removes
(purges) such training observations and embargoes a buffer after each test
block, so the out-of-sample estimate is honest.
"""
import numpy as np
import pandas as pd


class PurgedKFold:
    """t1 : pd.Series indexed by event START time (sorted ascending), values =
    event END time. Positionally aligned with the design matrix X."""

    def __init__(self, n_splits=5, t1=None, embargo_pct=0.0):
        if t1 is None:
            raise ValueError("t1 (event end times, indexed by start time) is required")
        if not isinstance(t1, pd.Series):
            raise TypeError("t1 must be a pandas Series")
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 must be sorted by its index (event start time)")
        self.n_splits = int(n_splits)
        self.t1 = t1
        self.embargo_pct = float(embargo_pct)

    def split(self, X=None):
        n = self.t1.shape[0]
        indices = np.arange(n)
        mbrg = int(n * self.embargo_pct)
        test_ranges = [(g[0], g[-1] + 1) for g in np.array_split(indices, self.n_splits)]
        starts = self.t1.index
        ends = self.t1.values
        for i, j in test_ranges:
            test_indices = indices[i:j]
            t0 = starts[i]                              # test window begins here
            test_end = self.t1.iloc[test_indices].max()  # latest label-end in test
            # train part 1: events that END on or before the test window starts
            left = indices[ends <= t0]
            # train part 2: events that START after the test's latest label-end (+embargo)
            max_end_pos = int(starts.searchsorted(test_end))
            right = indices[max_end_pos + mbrg:] if max_end_pos + mbrg < n else np.array([], dtype=int)
            # never let test indices sneak into train
            train_indices = np.setdiff1d(np.concatenate([left, right]), test_indices)
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
