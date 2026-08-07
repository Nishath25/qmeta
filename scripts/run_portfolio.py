"""HRP / NCO portfolio construction on the user's real streams.

Panel A (headline): the principled ORB-vs-dip split vs the hand-tuned 50/50.
Panel B: a rolling OUT-OF-SAMPLE contest across the 8 ORB instrument sleeves + dip
-- equal / inverse-variance / Markowitz min-variance / HRP / NCO -- estimating the
covariance on a trailing window, holding the next month, rolling. The HRP thesis:
lower out-of-sample variance than Markowitz, with no matrix inversion.
Run: python scripts/run_portfolio.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.portfolio import (
    equal_weights, inverse_variance_weights, min_variance_weights,
    hrp_weights, nco_weights,
)
from qmeta.metalabel.evaluate import stream_stats
from load_streams import load_orb, load_orb_ticker_matrix, load_dip

OUT = Path(r"C:\Users\madas\qmeta\scratch\portfolio.json")
ANN = 252
LOOK, STEP = 252, 21   # 1y trailing covariance, monthly rebalance


def _eff_n(w):
    w = np.asarray(w, dtype=float)
    return float(1.0 / np.sum(w ** 2))


def panel_a():
    """ORB vs dip, evaluated the SAME rolling-OOS way as Panel B (weights fit on a
    trailing window, applied forward) -- no in-sample fitting."""
    orb, dip = load_orb(), load_dip()
    df = pd.concat([orb.rename("ORB"), dip.rename("dip")], axis=1, join="inner", sort=False).sort_index()
    cols = list(df.columns)
    vol_full = df.std() * np.sqrt(ANN)
    methods = ["50/50", "inverse-vol (risk parity)", "inverse-variance / HRP"]
    port = {m: np.full(len(df), np.nan) for m in methods}
    wsum = {m: np.zeros(len(cols)) for m in methods}
    nreb = 0
    t = LOOK
    while t < len(df):
        win = df.iloc[t - LOOK:t]
        inv_vol = (1.0 / win.std()) / (1.0 / win.std()).sum()
        wts = {
            "50/50": pd.Series([0.5, 0.5], index=cols),
            "inverse-vol (risk parity)": inv_vol,
            "inverse-variance / HRP": inverse_variance_weights(win.cov()),  # HRP == inverse-variance for 2 assets
        }
        hi = min(t + STEP, len(df))
        seg = df.iloc[t:hi].values
        for m in methods:
            w = wts[m].reindex(cols).values
            port[m][t:hi] = (seg * w).sum(axis=1)
            wsum[m] += w
        nreb += 1
        t = hi
    idx = df.index
    out = {}
    for m in methods:
        r = pd.Series(port[m], index=idx).dropna()
        aw = wsum[m] / nreb
        out[m] = dict(weights={c: round(float(aw[i]), 3) for i, c in enumerate(cols)},
                      sharpe=round(stream_stats(r)["sharpe"], 3))
    oos_start = str(pd.Series(port["50/50"], index=idx).dropna().index[0].date())
    return dict(cols=cols, vol={c: round(float(vol_full[c]), 3) for c in cols},
                oos_start=oos_start, splits=out)


def panel_b():
    m = load_orb_ticker_matrix()
    dip = load_dip()
    panel = m.join(dip.rename("dip"), how="inner").fillna(0.0)
    cols = list(panel.columns)
    methods = ["equal", "inverse-variance", "min-variance", "HRP", "NCO"]
    port = {me: np.full(len(panel), np.nan) for me in methods}
    effn = {me: [] for me in methods}
    t = LOOK
    while t < len(panel):
        win = panel.iloc[t - LOOK:t]
        covm = win.cov().values.copy()
        covm[np.diag_indices_from(covm)] += 1e-12   # guard singular
        cov = pd.DataFrame(covm, index=cols, columns=cols)
        wts = {
            "equal": pd.Series(equal_weights(len(cols)), index=cols),
            "inverse-variance": inverse_variance_weights(cov),
            "min-variance": min_variance_weights(cov),
            "HRP": hrp_weights(cov),
            "NCO": nco_weights(cov),
        }
        hi = min(t + STEP, len(panel))
        seg = panel.iloc[t:hi].values
        for me in methods:
            w = wts[me].reindex(cols).values
            port[me][t:hi] = (seg * w).sum(axis=1)
            effn[me].append(_eff_n(w))
        t = hi
    idx = panel.index
    res, equity = {}, {}
    for me in methods:
        r = pd.Series(port[me], index=idx).dropna()
        s = stream_stats(r)
        res[me] = dict(sharpe=round(s["sharpe"], 3), vol=round(s["vol"], 4),
                       maxdd=round(s["maxdd"], 4), eff_n=round(float(np.mean(effn[me])), 2))
        eq = (1.0 + r).cumprod()
        equity[me] = [round(float(v), 4) for v in eq.values]
    r0 = pd.Series(port["equal"], index=idx).dropna()
    return dict(cols=cols, n_rebal=len(effn["equal"]),
                oos_start=str(r0.index[0].date()), oos_end=str(r0.index[-1].date()),
                methods=res, equity=equity, dates=[str(d.date()) for d in r0.index])


def main():
    out = dict(panel_a=panel_a(), panel_b=panel_b())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    a = out["panel_a"]
    print(f"\n=== Panel A: ORB vs dip -- rolling OOS from {a['oos_start']} (avg weights) ===")
    print(f"  annualized vol: " + ", ".join(f"{c} {a['vol'][c]*100:.0f}%" for c in a["cols"]))
    for k, v in a["splits"].items():
        ws = ", ".join(f"{c} {v['weights'][c]*100:.0f}%" for c in a["cols"])
        print(f"  {k:26s} [{ws}]  OOS Sharpe {v['sharpe']:.2f}")
    b = out["panel_b"]
    print(f"\n=== Panel B: {len(b['cols'])} sleeves, rolling OOS {b['oos_start']}..{b['oos_end']} ({b['n_rebal']} rebalances) ===")
    print(f"  {'method':18s} {'Sharpe':>7} {'vol':>7} {'maxDD':>8} {'effN':>6}")
    for me, v in b["methods"].items():
        print(f"  {me:18s} {v['sharpe']:>7.2f} {v['vol']*100:>6.1f}% {v['maxdd']*100:>7.1f}% {v['eff_n']:>6.1f}")
    hrp, mv = b["methods"]["HRP"], b["methods"]["min-variance"]
    print(f"\n  HRP vs Markowitz min-variance (OOS): Sharpe {hrp['sharpe']:.2f} vs {mv['sharpe']:.2f}, "
          f"vol {hrp['vol']*100:.1f}% vs {mv['vol']*100:.1f}%, effN {hrp['eff_n']:.1f} vs {mv['eff_n']:.1f}")
    return out


if __name__ == "__main__":
    main()
