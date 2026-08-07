"""Walk-forward meta-labeling on the ORB fund: does a "will this breakout win?"
model improve the fund's OUT-OF-SAMPLE risk-adjusted return? Honest verdict.
Run: python scripts/run_metalabel.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.metalabel import oos_walk_forward_proba, bet_size, compare, train_meta_model
from build_orb_features import build, FEATURES, OUT as FEAT

R_TO_RET, COST = 0.0025, 0.02
MIN_TRAIN, REFIT = 504, 63           # ~2y warmup, quarterly refit (in trading-day units)
OUTJSON = Path(r"C:\Users\madas\qmeta\scratch\metalabel.json")


def daily(df, size) -> pd.Series:
    x = np.asarray(size) * df["w"].to_numpy() * (df["R"].to_numpy() - COST) * R_TO_RET
    return pd.Series(x, index=df["date"]).groupby(level=0).sum()


def vol_match(s: pd.Series, target: pd.Series) -> pd.Series:
    sd = s.std()
    return s * (target.std() / sd) if sd > 0 else s


def main():
    df = pd.read_parquet(FEAT) if FEAT.exists() else build()
    df = oos_walk_forward_proba(df, FEATURES, "y", "date", MIN_TRAIN, REFIT, sw_col="uw")
    oos = df.dropna(subset=["proba"]).copy()
    raw_d = daily(oos, np.ones(len(oos)))

    modes = {}
    for mode in ["linear", "threshold", "prob"]:
        bs = bet_size(oos["proba"].to_numpy(), mode=mode)
        md = vol_match(daily(oos, bs), raw_d)        # vol-match so we compare SHAPE, not leverage
        taken = bs > 1e-9
        modes[mode] = dict(
            compare=compare(raw_d, md),
            taken_frac=float(taken.mean()),
            hit_taken=float((oos.loc[taken, "R"] > 0).mean()) if taken.any() else None,
            daily=md,
        )

    # descriptive feature importances (full-sample fit; the OOS proba came from walk-forward)
    m = train_meta_model(oos[FEATURES].to_numpy(), oos["y"].to_numpy(), sample_weight=oos["uw"].to_numpy())
    importances = dict(sorted(zip(FEATURES, [round(float(v), 4) for v in m.feature_importances_]),
                              key=lambda t: -t[1]))

    best = max(modes, key=lambda k: modes[k]["compare"]["meta"]["sharpe"])
    best_d = modes[best]["daily"]
    eq_raw = (1 + raw_d).cumprod()
    eq_meta = (1 + best_d).cumprod()

    out = dict(
        oos_start=str(oos["date"].min().date()), oos_end=str(oos["date"].max().date()),
        n_trades=int(len(oos)), raw_hit=float((oos["R"] > 0).mean()),
        raw=modes["linear"]["compare"]["raw"],
        modes={k: dict(sharpe=v["compare"]["meta"]["sharpe"], dsr=v["compare"]["meta"]["dsr"],
                       maxdd=v["compare"]["meta"]["maxdd"], d_sharpe=v["compare"]["d_sharpe"],
                       taken_frac=v["taken_frac"], hit_taken=v["hit_taken"]) for k, v in modes.items()},
        best_mode=best, importances=importances,
        equity=dict(dates=[str(d.date()) for d in eq_raw.index],
                    raw=[round(float(v), 4) for v in eq_raw.values],
                    meta=[round(float(v), 4) for v in eq_meta.values]),
    )
    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps(out, indent=2, default=float))

    r = modes["linear"]["compare"]["raw"]
    print(f"\nMeta-labeling the ORB fund  (OOS {out['oos_start']}..{out['oos_end']}, {out['n_trades']} trades)")
    print(f"  RAW ORB (OOS):  Sharpe {r['sharpe']:.2f}   DSR {r['dsr']*100:.0f}%   maxDD {r['maxdd']*100:.1f}%   hit {out['raw_hit']*100:.0f}%")
    for k, v in modes.items():
        c = v["compare"]["meta"]
        tag = "  <-- best" if k == best else ""
        ht = "n/a" if v["hit_taken"] is None else f"{v['hit_taken']*100:.0f}%"
        print(f"  meta[{k:9s}] Sharpe {c['sharpe']:.2f} ({v['compare']['d_sharpe']:+.2f})  DSR {c['dsr']*100:.0f}%  "
              f"maxDD {c['maxdd']*100:.1f}%  taken {v['taken_frac']*100:.0f}%  hit {ht}{tag}")
    print("  feature importances:", ", ".join(f"{k} {v:.2f}" for k, v in importances.items()))
    verdict = "HELPS" if modes[best]["compare"]["d_sharpe"] > 0.05 else ("NEUTRAL" if modes[best]["compare"]["d_sharpe"] > -0.05 else "HURTS")
    print(f"  VERDICT: meta-labeling {verdict} (best delta Sharpe {modes[best]['compare']['d_sharpe']:+.2f})")
    return out


if __name__ == "__main__":
    main()
