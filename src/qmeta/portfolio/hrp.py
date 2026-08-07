"""Hierarchical Risk Parity (Lopez de Prado 2016, AFML ch. 16).

HRP avoids inverting the covariance matrix: it clusters assets by correlation,
reorders the covariance so similar assets are adjacent (quasi-diagonalization),
then splits risk top-down by inverse cluster variance (recursive bisection).
This delivers lower out-of-sample variance than Markowitz, with no matrix inversion.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from qmeta.portfolio.baselines import cov_to_corr


def _quasi_diag(link):
    """Order the original leaves by following the linkage tree (AFML)."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    n = link[-1, 3]
    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i = df0.index
        j = df0.values - n
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(C, items):
    sub = C[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def _rec_bipart(C, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while len(clusters) > 0:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(C, c0), _cluster_var(C, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def hrp_weights(cov):
    """Hierarchical Risk Parity weights (sum to 1, all non-negative)."""
    if isinstance(cov, pd.DataFrame):
        labels = list(cov.index)
        C = cov.values.astype(float)
    else:
        C = np.asarray(cov, dtype=float)
        labels = list(range(C.shape[0]))
    n = C.shape[0]
    if n == 1:
        return pd.Series([1.0], index=labels)
    corr = cov_to_corr(C)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = _quasi_diag(link)                 # positional order
    w = _rec_bipart(C, sort_ix).reindex(range(n)).values
    return pd.Series(w, index=labels)
