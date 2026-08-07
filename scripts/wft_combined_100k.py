"""$100k walk-forward of the COMBINED book: the meta-filtered ORB + the dip
diversifier, 50/50 equal-risk, vs each sleeve alone -- and vs the raw-ORB+dip
blend, to show how much the meta-label filter lifts the combination.
All streams scaled to 15% annual vol (equal risk), $100k start, common OOS window.
Run: python scripts/wft_combined_100k.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from run_metalabel import eval_config, FEAT
from build_orb_features import build
from load_streams import load_dip
from wft_metalabel_100k import metrics, START

TARGET_VOL_D = 0.15 / np.sqrt(252)   # 15% annualized


def to_vol(r, target=TARGET_VOL_D):
    s = r.std()
    return r / s * target if s > 0 else r


def blend(a, b):
    """50/50 equal-risk: each scaled to unit vol, averaged, then to 15% vol."""
    return to_vol(0.5 * (a / a.std()) + 0.5 * (b / b.std()))


def main():
    df = pd.read_parquet(FEAT) if FEAT.exists() else build()
    oos, raw_d, modes = eval_config(df, "uw")
    meta = modes["threshold"]["daily"]
    dip = load_dip()

    j = pd.concat([meta.rename("orbM"), raw_d.rename("orbR"), dip.rename("dip")],
                  axis=1, join="inner", sort=False).sort_index()
    corr_meta = float(j[["orbM", "dip"]].corr().iloc[0, 1])

    streams = {
        "ORB + meta-filter": to_vol(j["orbM"]),
        "Dip diversifier": to_vol(j["dip"]),
        "Combined (meta-ORB + dip)": blend(j["orbM"], j["dip"]),
        "Combined (raw-ORB + dip)": blend(j["orbR"], j["dip"]),
    }
    res = {k: metrics(v) for k, v in streams.items()}

    out = dict(start=START, oos_start=str(j.index[0].date()), oos_end=str(j.index[-1].date()),
               n_days=int(len(j)), correlation=corr_meta, streams=res)
    OUT = Path(r"C:\Users\madas\qmeta\scratch\wft_combined_100k.json")
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n$100k COMBINED WFT  (OOS {out['oos_start']}..{out['oos_end']}, {out['n_days']} days, all @15% vol)")
    print(f"  correlation meta-ORB vs dip: {corr_meta:+.2f}")
    for k, m in res.items():
        so = "n/a" if m["sortino"] is None else f"{m['sortino']:.2f}"
        ca = "n/a" if m["calmar"] is None else f"{m['calmar']:.2f}"
        print(f"  {k:28s} ${m['final']:>10,.0f}  CAGR {m['cagr']*100:>4.1f}%  Sharpe {m['sharpe']:.2f}  "
              f"Sortino {so}  maxDD {m['maxdd']*100:>5.1f}%  Calmar {ca}  DSR {m['dsr']*100:.0f}%")
    cm, cr = res["Combined (meta-ORB + dip)"], res["Combined (raw-ORB + dip)"]
    print(f"\n  meta-filter lift to the blend: Sharpe {cr['sharpe']:.2f} -> {cm['sharpe']:.2f}, "
          f"${cr['final']:,.0f} -> ${cm['final']:,.0f}, maxDD {cr['maxdd']*100:.1f}% -> {cm['maxdd']*100:.1f}%")
    print(f"  wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
