"""$100k walk-forward of the deployable ORB + dip book (raw ORB, 50/50 equal-risk,
15% vol) over the full ORB window, with ORB / dip sleeves and SPY buy&hold for context.
Run: python scripts/wft_orbdip_100k.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from wft_metalabel_100k import metrics, START
from load_streams import load_orb, load_dip
from stress_test import load_spy
from wft_combined_100k import to_vol, blend

OUT = Path(r"C:\Users\madas\qmeta\scratch\wft_orbdip_100k.json")


def main():
    orb, dip, spy = load_orb(), load_dip(), load_spy()
    j = pd.concat([orb.rename("orb"), dip.rename("dip")], axis=1, join="inner", sort=False).sort_index()
    corr = float(j.corr().iloc[0, 1])
    comb = blend(j["orb"], j["dip"])
    spy_al = spy.reindex(j.index).dropna()
    streams = {
        "Combined (ORB + dip)": metrics(comb),
        "ORB fund": metrics(to_vol(j["orb"])),
        "Dip diversifier": metrics(to_vol(j["dip"])),
        "SPY buy & hold": metrics(spy_al),
    }
    out = dict(start=START, win_start=str(j.index[0].date()), win_end=str(j.index[-1].date()),
               n_days=int(len(j)), correlation=round(corr, 3), streams=streams)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n$100k WFT -- ORB + dip combined  ({out['win_start']}..{out['win_end']}, {out['n_days']} days) corr {corr:+.2f}")
    for k, m in streams.items():
        so = "n/a" if m["sortino"] is None else f"{m['sortino']:.2f}"
        print(f"  {k:22s} ${m['final']:>10,.0f}  {(m['final']/START-1)*100:>+5.0f}%  CAGR {m['cagr']*100:>4.1f}%  "
              f"Sharpe {m['sharpe']:.2f}  Sortino {so}  maxDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:.2f}  DSR {m['dsr']*100:.0f}%")
    print("\n  combined per-year P&L:")
    for yr, pnl, pct in streams["Combined (ORB + dip)"]["per_year"]:
        print(f"    {yr}:  ${pnl:>+10,.0f}  ({pct:>+6.1f}%)")
    print(f"  wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
