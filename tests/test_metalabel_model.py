import numpy as np
import pandas as pd
from qmeta.metalabel.model import train_meta_model, bet_size, oos_walk_forward_proba


def test_bet_size_modes():
    p = np.array([0.2, 0.5, 0.8, 1.0])
    assert list(bet_size(p, "threshold")) == [0, 1, 1, 1]
    lin = bet_size(p, "linear")
    assert abs(lin[0] - 0.0) < 1e-9 and abs(lin[2] - 0.6) < 1e-9 and lin[3] == 1.0
    assert list(bet_size(p, "prob")) == [0.2, 0.5, 0.8, 1.0]
    assert bet_size(np.array([0.55]), "linear", floor=0.2)[0] == 0.0  # size 0.1 < floor


def test_model_learns_separable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = ((x[:, 0] + x[:, 1]) > 0).astype(int)
    m = train_meta_model(x, y, seed=0)
    assert (m.predict(x) == y).mean() > 0.8


def test_walk_forward_no_lookahead():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2018-01-01", periods=600)
    rows = []
    for d in dates:
        for _ in range(3):
            f1, f2 = rng.normal(), rng.normal()
            rows.append(dict(date=d, f1=f1, f2=f2, y=int(f1 + f2 + 0.2 * rng.normal() > 0)))
    df = pd.DataFrame(rows)
    out = oos_walk_forward_proba(df, ["f1", "f2"], "y", "date", min_train=252, refit=63)
    early = out[out["date"] < dates[252]]
    assert early["proba"].isna().all()                      # nothing predicted in-sample
    late = out[out["date"] >= dates[252]].dropna(subset=["proba"])
    assert len(late) > 0.9 * len(out[out["date"] >= dates[252]])
    acc = ((late["proba"] > 0.5).astype(int) == late["y"]).mean()
    assert acc > 0.6                                        # OOS beats chance
