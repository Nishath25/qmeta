import numpy as np
from qmeta.portfolio.nco import nco_weights, cluster_corr
from qmeta.portfolio.baselines import cov_to_corr


def _block_cov():
    corr = np.full((6, 6), 0.1)
    for i in range(6):
        for j in range(6):
            if i == j:
                corr[i, j] = 1.0
            elif (i < 3) == (j < 3):
                corr[i, j] = 0.7
    std = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    return corr * np.outer(std, std)


def test_nco_small_n_fallback_sums_to_one():
    w = nco_weights(np.diag([4.0, 1.0]))
    assert abs(w.sum() - 1.0) < 1e-9


def test_nco_block_runs_and_sums():
    w = nco_weights(_block_cov(), max_k=3, seed=0)
    assert abs(w.sum() - 1.0) < 1e-9 and np.isfinite(w.values).all()


def test_nco_symmetric_blocks_give_equal_weight():
    # two symmetric correlated blocks (unit variances) -> NCO allocates equally to all four
    corr = np.array([[1, .9, 0, 0], [.9, 1, 0, 0], [0, 0, 1, .9], [0, 0, .9, 1]], dtype=float)
    w = nco_weights(corr, max_k=3, seed=0)          # cov == corr (unit variances)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w.values, 0.25, atol=0.03)   # intra 0.5/0.5, inter 0.5/0.5 -> 0.25 each


def test_cluster_corr_recovers_two_blocks():
    labels = cluster_corr(cov_to_corr(_block_cov()), max_k=4, seed=0)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
