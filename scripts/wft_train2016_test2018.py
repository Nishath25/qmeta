"""Train the meta-filter on 2016-2018, TEST from 2018 -> the filter is active across
the whole 2018-2026 book (no unfiltered warmup). Feature blotter comes from
blotter_atr3 (trades back to 2016-07); it drops the HAR weight `w` (unavailable before
2018), so the model uses 5 entry-time features. Fund P&L still uses the HAR-sized
raw_har_trades stream, with the 2016-trained proba merged on.
Run: python scripts/wft_train2016_test2018.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.metalabel import oos_walk_forward_proba
from run_metalabel import REFIT
from load_streams import load_dip
from wft_metalabel_100k import metrics, START
from wft_combined_100k import to_vol, blend

DATA = Path(r"C:\Users\madas\orb-strategy\data")
FEATS = ["direction", "atr_ratio", "risk_pct", "dow", "month"]     # no `w` (not available pre-2018)
COST, R_TO_RET = 0.02, 0.0025
TEST_FROM = pd.Timestamp("2018-09-21")
OUT = Path(r"C:\Users\madas\qmeta\scratch\wft_train2016.json")


def feature_blotter():
    bl = pd.read_parquet(DATA / "blotter_atr3.parquet")          # 2016-07 .. 2026-07
    bl["date"] = pd.to_datetime(bl["date"])
    bl = bl.dropna(subset=["atr_ratio", "entry", "risk", "R"])
    bl = bl[bl["risk"] > 0].copy()
    bl["risk_pct"] = bl["risk"] / bl["entry"]
    bl["dow"] = bl["date"].dt.dayofweek
    bl["month"] = bl["date"].dt.month
    bl["y"] = (bl["R"] > 0).astype(int)
    bl["uw"] = 1.0 / bl.groupby("date")["R"].transform("size")
    return bl.sort_values("date").reset_index(drop=True)


def daily(mdf, size):
    x = np.asarray(size) * mdf["w"].to_numpy() * (mdf["R"].to_numpy() - COST) * R_TO_RET
    return pd.Series(x, index=mdf["date"]).groupby(level=0).sum()


def main():
    bl = feature_blotter()
    dates = np.array(sorted(bl["date"].unique()))
    min_train = int((dates < TEST_FROM).sum())                    # first OOS == first trade in 2018-09
    d = oos_walk_forward_proba(bl, FEATS, "y", "date", min_train, REFIT, sw_col="uw")

    raw = pd.read_parquet(DATA / "raw_har_trades_atr3.parquet")    # HAR-sized fund trades (2018+)
    raw["date"] = pd.to_datetime(raw["date"])
    keys = ["date", "ticker", "entry", "risk"]
    pm = d[keys + ["proba"]].drop_duplicates(keys)
    m = raw.merge(pm, on=keys, how="left")
    covered = float(m["proba"].notna().mean())
    bet = np.where(m["proba"].isna(), 1.0, (m["proba"] >= 0.5).astype(float))
    meta_orb = daily(m, bet)
    raw_orb = daily(m, np.ones(len(m)))
    taken = float((bet > 0).mean())
    dip = load_dip()

    j = pd.concat([meta_orb.rename("orbM"), raw_orb.rename("orbR"), dip.rename("dip")],
                  axis=1, join="inner", sort=False).sort_index()
    corr = float(j[["orbM", "dip"]].corr().iloc[0, 1])
    streams = {
        "ORB + meta-filter (2016-trained)": to_vol(j["orbM"]),
        "Dip diversifier": to_vol(j["dip"]),
        "Combined (meta-ORB + dip)": blend(j["orbM"], j["dip"]),
        "Combined (raw-ORB + dip)": blend(j["orbR"], j["dip"]),
    }
    res = {k: metrics(v) for k, v in streams.items()}
    out = dict(start=START, win_start=str(j.index[0].date()), win_end=str(j.index[-1].date()),
               n_days=int(len(j)), train_from="2016-07", test_from="2018-09",
               features=FEATS, proba_coverage=covered, taken_frac=taken, correlation=corr, streams=res)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n$100k WFT -- filter TRAINED 2016-2018, TESTED {out['win_start']}..{out['win_end']} ({out['n_days']} days, @15% vol)")
    print(f"  5 features (no w); proba covers {covered*100:.0f}% of fund trades; filter takes {taken*100:.0f}%; corr {corr:+.2f}")
    for k, mm in res.items():
        so = "n/a" if mm["sortino"] is None else f"{mm['sortino']:.2f}"
        ca = "n/a" if mm["calmar"] is None else f"{mm['calmar']:.2f}"
        print(f"  {k:34s} ${mm['final']:>10,.0f}  CAGR {mm['cagr']*100:>4.1f}%  Sharpe {mm['sharpe']:.2f}  "
              f"Sortino {so}  maxDD {mm['maxdd']*100:>5.1f}%  Calmar {ca}  DSR {mm['dsr']*100:.0f}%")
    print(f"  wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
