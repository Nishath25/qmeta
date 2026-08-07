"""Grade the ORB fund and the dip diversifier with the full selection toolkit,
emit scorecard.json + a console table. Run: python scripts/scorecard.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.selection import (
    sharpe_per_obs, moments, annualize_sr,
    probabilistic_sharpe_ratio, min_track_record_length,
    expected_max_sharpe, deflated_sharpe_ratio, min_backtest_length,
    probability_of_backtest_overfitting,
    combined_max_sharpe, improves_max_sharpe, max_correlation_for_approval,
    indifference_curve,
)
from qmeta.selection.drawdown import from_returns as dd_from_returns

OUT = Path(r"C:\Users\madas\qmeta\scratch\scorecard.json")
N_TRIALS = 19  # documented gauntlet trial count; adjustable
EMPIRICAL_MAXDD = {"orb": 0.18, "dip": 0.40}  # from the $50k WFTs, for the dashboard


def grade_stream(returns: pd.Series, name: str, n_trials: int = N_TRIALS, ppy: int = 252) -> dict:
    r = np.asarray(returns.dropna().values, dtype=float)
    sr = sharpe_per_obs(r)
    sk, ku = moments(r)
    n = len(r)
    var_sr = 1.0 / n
    dd = dd_from_returns(r, prob=0.95, ppy=ppy)
    mintrl = min_track_record_length(sr, sk, ku, 0.0, 0.95)
    return {
        "name": name,
        "n_obs": n,
        "track_years": n / ppy,
        "sharpe_ann": annualize_sr(sr, ppy),
        "skew": sk, "kurt": ku,
        "psr": probabilistic_sharpe_ratio(sr, n, sk, ku, 0.0),
        "mintrl_obs": mintrl,
        "mintrl_years": mintrl / ppy,
        "expected_max_sr_ann": annualize_sr(expected_max_sharpe(n_trials, var_sr), ppy),
        "dsr": deflated_sharpe_ratio(sr, n, sk, ku, n_trials, var_sr),
        "min_backtest_years": min_backtest_length(n_trials, annualize_sr(sr, ppy)),
        "drawdown": dd,
        "pbo": None,
    }


def build_scorecard() -> dict:
    from load_streams import load_orb, load_orb_ticker_matrix, load_dip
    orb, dip = load_orb(), load_dip()
    cards = {"orb": grade_stream(orb, "ORB fund"), "dip": grade_stream(dip, "Dip diversifier")}
    for key in cards:
        cards[key]["empirical_maxdd"] = EMPIRICAL_MAXDD[key]
    # PBO on ORB per-ticker columns: does the best in-sample instrument survive OOS?
    mat = load_orb_ticker_matrix()
    cards["orb"]["pbo"] = probability_of_backtest_overfitting(mat.values, n_splits=10)["pbo"]
    cards["orb"]["pbo_note"] = f"across {mat.shape[1]} per-ticker return columns"
    # Strategy approval: ORB approved, dip candidate (annualized Sharpes, aligned corr)
    sa, sn = cards["orb"]["sharpe_ann"], cards["dip"]["sharpe_ann"]
    a0, a1 = orb.align(dip, join="inner")
    corr = float(pd.concat([a0, a1], axis=1).corr().iloc[0, 1])
    xs, ys = indifference_curve(sa, level=sa, n=81)
    approval = {
        "sr_approved": sa, "sr_candidate": sn, "correlation": corr,
        "improves": bool(improves_max_sharpe(sa, sn, corr)),
        "combined_max_sharpe": combined_max_sharpe(sa, sn, corr),
        "max_corr_for_approval": max_correlation_for_approval(sa, sn),
        "indifference_curve": {"corr": [float(x) for x in xs],
                               "sr_new": [None if np.isnan(v) else float(v) for v in ys]},
    }
    scorecard = {"streams": cards, "approval": approval, "n_trials": N_TRIALS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scorecard, indent=2, default=float))
    return scorecard


def _print(sc: dict) -> None:
    for c in sc["streams"].values():
        dd = c["drawdown"]
        print(f"\n=== {c['name']}  ({c['track_years']:.1f}y, {c['n_obs']} days) ===")
        print(f"  Sharpe (ann)     {c['sharpe_ann']:.2f}")
        print(f"  PSR  (SR>0)      {c['psr']*100:.1f}%")
        print(f"  DSR  (deflated)  {c['dsr']*100:.1f}%   [E[maxSR]={c['expected_max_sr_ann']:.2f} over K={sc['n_trials']} trials]")
        print(f"  MinTRL           {c['mintrl_years']:.1f}y   (track record {c['track_years']:.1f}y)")
        print(f"  MinBTL           {c['min_backtest_years']:.1f}y   (need this much history for K={sc['n_trials']})")
        pbo_s = "n/a" if c["pbo"] is None else f"{c['pbo']*100:.0f}%"
        print(f"  PBO              {pbo_s}")
        print(f"  Proj. MaxDD(95%) {dd['max_dd']*100:.1f}%   vs empirical {c['empirical_maxdd']*100:.0f}%")
        print(f"  Proj. underwater {dd['max_tuw_years']:.1f}y   (bottom at {dd['time_to_maxdd_years']:.1f}y, penance {dd['penance_ratio']:.2f})")
    a = sc["approval"]
    print(f"\n=== Strategy Approval: add dip to the ORB book? ===")
    print(f"  ORB SR {a['sr_approved']:.2f}  |  dip SR {a['sr_candidate']:.2f}  |  corr {a['correlation']:+.2f}")
    print(f"  improves aggregate Sharpe: {a['improves']}   ->   combined max-Sharpe {a['combined_max_sharpe']:.2f}")
    print(f"  (dip would help up to correlation {a['max_corr_for_approval']:.2f})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    _print(build_scorecard())
