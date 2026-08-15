"""$100k walk-forward of the DIP strategy on its own (natural sizing), 2016-2026,
benchmarked against SPY buy-and-hold. The dip rule is fixed a priori, so every year
is out-of-sample. Full metrics + per-year + monthly equity/underwater for the dashboard.
Run: python scripts/wft_dip_100k.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from wft_metalabel_100k import metrics, START
from load_streams import load_dip
from stress_test import load_spy

OUT = Path(r"C:\Users\madas\qmeta\scratch\wft_dip_100k.json")


def main():
    dip = load_dip()
    spy = load_spy()
    j = pd.concat([dip.rename("dip"), spy.rename("spy")], axis=1, join="inner", sort=False).sort_index()
    m_dip, m_spy = metrics(j["dip"]), metrics(j["spy"])
    out = dict(start=START, win_start=str(j.index[0].date()), win_end=str(j.index[-1].date()),
               n_days=int(len(j)), dip=m_dip, spy=m_spy)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    def line(name, m):
        so = "n/a" if m["sortino"] is None else f"{m['sortino']:.2f}"
        print(f"  {name:14s} ${m['final']:>10,.0f}  {(m['final']/START-1)*100:>+5.0f}%  CAGR {m['cagr']*100:>4.1f}%  "
              f"Sharpe {m['sharpe']:.2f}  Sortino {so}  vol {m['vol']*100:.0f}%  maxDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:.2f}")

    print(f"\n$100k DIP walk-forward  ({out['win_start']}..{out['win_end']}, {out['n_days']} days, natural sizing)")
    line("DIP strategy", m_dip)
    line("SPY buy&hold", m_spy)
    print("\n  per-year P&L (dip):")
    for yr, pnl, pct in m_dip["per_year"]:
        print(f"    {yr}:  ${pnl:>+10,.0f}  ({pct:>+6.1f}%)")
    print(f"  wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
