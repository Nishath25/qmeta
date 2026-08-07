"""Nested-Clustered Optimization (Lopez de Prado 2019).

NCO splits the optimization into robust sub-problems: cluster correlated assets,
solve each cluster on its own (small, well-conditioned) covariance, reduce each
cluster to one synthetic asset, solve the between-cluster problem, then combine.
This controls the instability that wrecks a single large Markowitz optimization.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from qmeta.portfolio.baselines import cov_to_corr, min_variance_weights, max_sharpe_weights


def _optimal(cov, mu=None):
    return min_variance_weights(cov) if mu is None else max_sharpe_weights(cov, mu)


def cluster_corr(corr, max_k=None, n_init=10, seed=0):
    """KMeans-cluster assets on the correlation-distance; silhouette picks k."""
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    n = corr.shape[0]
    if n <= 2:
        return np.arange(n)
    if max_k is None:
        max_k = max(2, n // 2)
    max_k = min(max_k, n - 1)
    best_labels, best_score = np.zeros(n, dtype=int), -np.inf
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(dist)
        if len(set(km.labels_)) < 2:
            continue
        sil = silhouette_samples(dist, km.labels_)
        score = sil.mean() / (sil.std() + 1e-9)   # AFML: mean/std of silhouette
        if score > best_score:
            best_score, best_labels = score, km.labels_
    return best_labels


def nco_weights(cov, mu=None, max_k=None, seed=0):
    """Nested-Clustered Optimization weights (sum to 1). Falls back to the base
    optimizer for n<4, where clustering is not meaningful."""
    cov = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(np.asarray(cov, dtype=float))
    n = cov.shape[0]
    mu_s = None if mu is None else pd.Series(np.asarray(mu, dtype=float), index=cov.index)
    if n < 4:
        return _optimal(cov, None if mu_s is None else mu_s.values)
    corr = cov_to_corr(cov.values)
    labels = cluster_corr(corr, max_k=max_k, seed=seed)
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    keys = sorted(clusters)
    w_intra = pd.DataFrame(0.0, index=range(n), columns=keys)
    for lab in keys:
        items = clusters[lab]
        sub = cov.iloc[items, items]
        wi = _optimal(sub, None if mu_s is None else mu_s.iloc[items].values).values
        for pos, it in enumerate(items):
            w_intra.iloc[it, keys.index(lab)] = wi[pos]
    cov_red = w_intra.values.T @ cov.values @ w_intra.values
    mu_red = None if mu_s is None else (w_intra.values.T @ mu_s.values)
    w_inter = _optimal(pd.DataFrame(cov_red), mu_red).values
    w = (w_intra.values * w_inter).sum(axis=1)
    return pd.Series(w, index=cov.index)
