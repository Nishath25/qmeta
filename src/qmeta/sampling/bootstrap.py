"""Uniqueness weighting & Sequential Bootstrap (Lopez de Prado, AFML ch. 4).

Overlapping labels are not independent draws. Average uniqueness down-weights
observations whose label spans overlap others; the sequential bootstrap draws a
training set that favors low-overlap (more unique) samples, reducing redundancy.
"""
import numpy as np
import pandas as pd


def get_indicator_matrix(bar_index, t1):
    """Binary (bars x events) matrix: 1 where an event is live at a bar.
    bar_index : iterable of bar ids. t1 : pd.Series index=event start bar,
    value=event end bar (inclusive)."""
    ind = pd.DataFrame(0.0, index=list(bar_index), columns=range(len(t1)))
    for c, (t0, t1v) in enumerate(t1.items()):
        ind.loc[t0:t1v, c] = 1.0
    return ind


def average_uniqueness(ind):
    """Per-event average uniqueness = mean over its live bars of 1/concurrency."""
    concurrency = ind.sum(axis=1)
    avg = pd.Series(0.0, index=ind.columns)
    for col in ind.columns:
        live = ind[col] > 0
        if live.any():
            avg[col] = float((1.0 / concurrency[live]).mean())
    return avg


def sequential_bootstrap(ind, n=None, rng=None):
    """Draw `n` event indices sequentially, each step favoring the event that is
    most unique given those already drawn. Returns a list of column labels.
    Works positionally so repeated draws are handled correctly."""
    if n is None:
        n = ind.shape[1]
    if rng is None:
        rng = np.random.default_rng(0)
    M = ind.to_numpy(dtype=float)          # bars x events
    cols = list(ind.columns)
    n_ev = M.shape[1]
    conc = np.zeros(M.shape[0])            # running concurrency of the drawn set
    phi = []
    while len(phi) < n:
        avg = np.zeros(n_ev)
        for k in range(n_ev):
            c = conc + M[:, k]             # concurrency if event k were added
            live = M[:, k] > 0
            if live.any():
                avg[k] = float((1.0 / c[live]).mean())
        total = avg.sum()
        prob = avg / total if total > 0 else np.full(n_ev, 1.0 / n_ev)
        pick = int(rng.choice(n_ev, p=prob))
        phi.append(pick)
        conc = conc + M[:, pick]
    return [cols[i] for i in phi]
