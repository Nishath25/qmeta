import numpy as np
from qmeta.portfolio.hrp import hrp_weights
from qmeta.portfolio.baselines import min_variance_weights


def _block_cov():
    # 2 clusters of 3; high intra-corr (0.7), low inter (0.1); cluster A vol 0.2, B vol 0.1
    corr = np.full((6, 6), 0.1)
    for i in range(6):
        for j in range(6):
            if i == j:
                corr[i, j] = 1.0
            elif (i < 3) == (j < 3):
                corr[i, j] = 0.7
    std = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    return corr * np.outer(std, std)


def test_hrp_two_asset_reduces_to_inverse_variance():
    w = hrp_weights(np.diag([4.0, 1.0]))
    assert abs(w.iloc[0] - 0.2) < 1e-9 and abs(w.iloc[1] - 0.8) < 1e-9


def test_hrp_weights_sum_and_nonneg():
    rng = np.random.default_rng(0)
    cov = np.cov(rng.normal(size=(300, 6)), rowvar=False)
    w = hrp_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-9 and (w >= -1e-12).all()


def test_hrp_risk_parity_favours_low_vol_cluster():
    w = hrp_weights(_block_cov())
    assert w.iloc[3:].sum() > w.iloc[:3].sum()   # low-vol cluster gets more weight
    assert (w >= -1e-12).all() and abs(w.sum() - 1.0) < 1e-9
