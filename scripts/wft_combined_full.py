"""FULL-WINDOW (2018-2026) $100k WFT of meta-ORB + dip. Honest live experience:
the meta-filter needs a 2y training warmup, so the ORB runs RAW during 2018-09..2020-12
(no proba yet) and META-FILTERED after. Combined 50/50 equal-risk @15% vol vs each sleeve
and vs the raw-ORB+dip blend. Run: python scripts/wft_combined_full.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.metalabel import oos_walk_forward_proba, bet_size
from run_metalabel import daily, FEAT, MIN_TRAIN, REFIT
from build_orb_features import FEATURES, build
from load_streams import load_dip
from wft_metalabel_100k import metrics, START
from wft_combined_100k import to_vol, blend

OUT = Path(r"C:\Users\madas\qmeta\scratch\wft_combined_full.json")


def main():
    df = pd.read_parquet(FEAT) if FEAT.exists() else build()
    d = oos_walk_forward_proba(df, FEATURES, "y", "date", MIN_TRAIN, REFIT, sw_col="uw")
    proba = d["proba"].to_numpy()
    # filter engages only where a walk-forward proba exists; warmup takes every breakout (bet=1)
    bet = np.where(np.isnan(proba), 1.0, (proba >= 0.5).astype(float))
    meta_full = daily(d, bet)
    raw_full = daily(d, np.ones(len(d)))
    active_from = str(pd.to_datetime(d.loc[d["proba"].notna(), "date"]).min().date())
    dip = load_dip()

    j = pd.concat([meta_full.rename("orbM"), raw_full.rename("orbR"), dip.rename("dip")],
                  axis=1, join="inner", sort=False).sort_index()
    corr = float(j[["orbM", "dip"]].corr().iloc[0, 1])
    streams = {
        "ORB + meta-filter": to_vol(j["orbM"]),
        "Dip diversifier": to_vol(j["dip"]),
        "Combined (meta-ORB + dip)": blend(j["orbM"], j["dip"]),
        "Combined (raw-ORB + dip)": blend(j["orbR"], j["dip"]),
    }
    res = {k: metrics(v) for k, v in streams.items()}
    out = dict(start=START, win_start=str(j.index[0].date()), win_end=str(j.index[-1].date()),
               n_days=int(len(j)), filter_active_from=active_from, correlation=corr, streams=res)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n$100k FULL-WINDOW WFT  ({out['win_start']}..{out['win_end']}, {out['n_days']} days, @15% vol)")
    print(f"  meta-filter engages from {active_from} (raw before); corr meta-ORB vs dip {corr:+.2f}")
    for k, m in res.items():
        so = "n/a" if m["sortino"] is None else f"{m['sortino']:.2f}"
        ca = "n/a" if m["calmar"] is None else f"{m['calmar']:.2f}"
        print(f"  {k:28s} ${m['final']:>10,.0f}  CAGR {m['cagr']*100:>4.1f}%  Sharpe {m['sharpe']:.2f}  "
              f"Sortino {so}  maxDD {m['maxdd']*100:>5.1f}%  Calmar {ca}  DSR {m['dsr']*100:.0f}%")
    cm, cr = res["Combined (meta-ORB + dip)"], res["Combined (raw-ORB + dip)"]
    print(f"\n  full-window combined: meta {cm['sharpe']:.2f}/DSR {cm['dsr']*100:.0f}% vs raw {cr['sharpe']:.2f}/DSR {cr['dsr']*100:.0f}%")
    print(f"  wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
