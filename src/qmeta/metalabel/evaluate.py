"""Honest OOS comparison of a raw vs meta-labeled return stream."""
import numpy as np
import pandas as pd

from qmeta.selection import (
    sharpe_per_obs, annualize_sr, moments, deflated_sharpe_ratio,
)


def stream_stats(daily, n_trials=19, ppy=252) -> dict:
    r = np.asarray(pd.Series(daily).dropna(), dtype=float)
    n = len(r)
    if n < 2 or r.std() == 0:
        return dict(sharpe=0.0, dsr=0.0, maxdd=0.0, vol=0.0, n=int(n))
    sk, ku = moments(r)
    sr = sharpe_per_obs(r)
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return dict(
        sharpe=annualize_sr(sr, ppy),
        dsr=deflated_sharpe_ratio(sr, n, sk, ku, n_trials, 1.0 / n),
        maxdd=mdd,
        vol=float(r.std() * np.sqrt(ppy)),
        n=int(n),
    )


def compare(raw_daily, meta_daily, n_trials=19) -> dict:
    a = stream_stats(raw_daily, n_trials)
    b = stream_stats(meta_daily, n_trials)
    return dict(
        raw=a, meta=b,
        d_sharpe=b["sharpe"] - a["sharpe"],
        d_dsr=b["dsr"] - a["dsr"],
        d_maxdd=b["maxdd"] - a["maxdd"],
    )
