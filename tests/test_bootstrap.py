import numpy as np
import pandas as pd
from qmeta.sampling.bootstrap import (
    get_indicator_matrix, average_uniqueness, sequential_bootstrap,
)


def test_indicator_and_uniqueness_afml_example():
    # AFML ch.4 example: events over bars {0-2, 2-3, 4-5}
    t1 = pd.Series({0: 2, 2: 3, 4: 5})
    ind = get_indicator_matrix(range(6), t1)
    assert ind.shape == (6, 3)
    au = average_uniqueness(ind).values
    assert abs(au[0] - 0.8333333) < 1e-6   # bars 0,1,2 with concurrency 1,1,2
    assert abs(au[1] - 0.75) < 1e-6        # bars 2,3 with concurrency 2,1
    assert abs(au[2] - 1.0) < 1e-9         # bars 4,5 alone


def test_sequential_bootstrap_deterministic_and_valid():
    t1 = pd.Series({0: 2, 2: 3, 4: 5})
    ind = get_indicator_matrix(range(6), t1)
    a = sequential_bootstrap(ind, n=6, rng=np.random.default_rng(7))
    b = sequential_bootstrap(ind, n=6, rng=np.random.default_rng(7))
    assert a == b                 # same seed -> reproducible
    assert len(a) == 6 and set(a) <= {0, 1, 2}


def test_most_unique_event_has_highest_selection_prob():
    # On the first draw the pick probability is proportional to average uniqueness,
    # so the most-unique event (2, uniqueness 1.0) must have the largest share.
    t1 = pd.Series({0: 2, 2: 3, 4: 5})
    ind = get_indicator_matrix(range(6), t1)
    au = average_uniqueness(ind).values
    prob = au / au.sum()
    assert prob.argmax() == 2
