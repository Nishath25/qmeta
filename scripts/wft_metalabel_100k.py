"""$100k walk-forward of the ORB fund WITH the meta-label breakout filter (skip the
breakouts the P(win) model flags as losers) vs the raw ORB, over the out-of-sample
window. Full performance metrics + per-year + monthly equity for the dashboard.
Run: python scripts/wft_metalabel_100k.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.selection import sharpe_per_obs, annualize_sr, moments, deflated_sharpe_ratio
from run_metalabel import eval_config, daily, FEAT
from build_orb_features import build

START = 100_000.0
ANN = np.sqrt(252)
N_TRIALS = 19
OUT = Path(r"C:\Users\madas\qmeta\scratch\wft_metalabel_100k.json")


def dsr(r):
    a = np.asarray(r.dropna(), dtype=float)
    sk, ku = moments(a)
    return deflated_sharpe_ratio(sharpe_per_obs(a), len(a), sk, ku, N_TRIALS, 1.0 / len(a))


def metrics(r, start=START):
    r = r.dropna()
    eq = start * (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / start) ** (1 / yrs) - 1
    dn = r[r < 0].std()
    mo = eq.resample("ME").last()
    mor = mo.pct_change().dropna()
    yr = eq.resample("YE").last()
    peryr, prev = [], start
    for t, v in yr.items():
        peryr.append([int(t.year), round(float(v - prev)), round(float(v / prev - 1) * 100, 1)])
        prev = v
    me = eq.resample("ME").last()
    mdd = me / eq.cummax().resample("ME").last() - 1
    return dict(
        final=float(eq.iloc[-1]), total=float(eq.iloc[-1] / start - 1), cagr=float(cagr),
        sharpe=float(r.mean() / r.std() * ANN), sortino=float(r.mean() / dn * ANN) if dn > 0 else None,
        maxdd=float(dd.min()), maxdd_dollar=float((eq.cummax() - eq).max()),
        calmar=float(cagr / abs(dd.min())) if dd.min() < 0 else None,
        vol=float(r.std() * ANN), win_mo=float((mor > 0).mean()), dsr=float(dsr(r)),
        per_year=peryr,
        monthly=[{"m": t.strftime("%Y-%m"), "e": round(float(v)), "dd": round(float(mdd[t]) * 100, 1)}
                 for t, v in me.items()],
    )


def main():
    df = pd.read_parquet(FEAT) if FEAT.exists() else build()
    oos, raw_d, modes = eval_config(df, "uw")            # walk-forward with uniqueness weighting
    meta_d = modes["threshold"]["daily"]                 # threshold skip-gate, vol-matched to raw
    m_raw, m_meta = metrics(raw_d), metrics(meta_d)
    th = modes["threshold"]
    thn = modes["threshold"]  # for taken/hit
    no_uw = eval_config(df, None)[2]["threshold"]["compare"]["d_sharpe"]

    out = dict(
        start=START,
        oos_start=str(oos["date"].min().date()), oos_end=str(oos["date"].max().date()),
        n_trades=int(len(oos)),
        raw=m_raw, meta=m_meta,
        taken_frac=float(th["taken_frac"]), hit_taken=float(th["hit_taken"]),
        raw_hit=float((oos["R"] > 0).mean()),
        d_sharpe_with_uw=float(th["compare"]["d_sharpe"]), d_sharpe_without_uw=float(no_uw),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    def line(name, m):
        so = "n/a" if m["sortino"] is None else f"{m['sortino']:.2f}"
        ca = "n/a" if m["calmar"] is None else f"{m['calmar']:.2f}"
        print(f"  {name:22s} ${m['final']:>10,.0f}  {m['total']*100:>+6.0f}%  CAGR {m['cagr']*100:>4.1f}%  "
              f"Sharpe {m['sharpe']:.2f}  Sortino {so}  maxDD {m['maxdd']*100:>5.1f}% (${m['maxdd_dollar']:>8,.0f})  "
              f"Calmar {ca}  DSR {m['dsr']*100:.0f}%")

    print(f"\n$100k WFT -- ORB with meta-label filter vs raw  (OOS {out['oos_start']}..{out['oos_end']}, {out['n_trades']} trades)")
    line("RAW ORB", m_raw)
    line("ORB + meta-filter", m_meta)
    print(f"\n  filter takes {out['taken_frac']*100:.0f}% of breakouts; hit rate {out['raw_hit']*100:.0f}% -> {out['hit_taken']*100:.0f}%")
    print(f"  delta Sharpe {out['d_sharpe_with_uw']:+.2f} (with uniqueness weighting) | {out['d_sharpe_without_uw']:+.2f} (without -- the gain is contingent on uw)")
    print(f"  wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
