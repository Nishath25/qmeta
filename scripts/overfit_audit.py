"""Overfitting audit for the meta-filtered ORB and the combined book, using the
qmeta selection tools: Deflated Sharpe, PBO (CSCV) over a config family, minimum
backtest length, sub-period stability, and the uniqueness-weighting sensitivity.
Run: python scripts/overfit_audit.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.selection import (
    sharpe_per_obs, annualize_sr, moments, deflated_sharpe_ratio,
    min_backtest_length, probability_of_backtest_overfitting,
)
from run_metalabel import eval_config, daily, FEAT
from build_orb_features import build
from load_streams import load_dip

ANN = np.sqrt(252)
N_TRIALS = 19
TARGET_VOL_D = 0.15 / ANN
OUT = Path(r"C:\Users\madas\qmeta\scratch\overfit.json")


def sh(r):
    r = np.asarray(r.dropna(), dtype=float)
    return float(r.mean() / r.std() * ANN) if r.std() > 0 else 0.0


def dsr(r):
    a = np.asarray(r.dropna(), dtype=float)
    sk, ku = moments(a)
    return float(deflated_sharpe_ratio(sharpe_per_obs(a), len(a), sk, ku, N_TRIALS, 1.0 / len(a)))


def unit(r):
    return r / r.std()


def blend15(orb, dip):
    u = 0.5 * unit(orb) + 0.5 * unit(dip)
    return u / u.std() * TARGET_VOL_D


def main():
    df = pd.read_parquet(FEAT) if FEAT.exists() else build()
    oos, raw_d, modes = eval_config(df, "uw")
    meta = modes["threshold"]["daily"]
    _, _, modes_no = eval_config(df, None)          # uw sensitivity
    proba = oos["proba"].to_numpy()
    dip = load_dip()

    idx = daily(oos, np.ones(len(oos))).index.intersection(dip.index)
    dipd = dip.reindex(idx)
    meta_i, raw_i = meta.reindex(idx), raw_d.reindex(idx)
    comb = blend15(meta_i, dipd)

    # (1) Deflated Sharpe
    dsr_meta, dsr_comb = dsr(meta_i), dsr(comb)

    # (2) PBO via CSCV over a config family (threshold x dip-weight), from one walk-forward
    cols = {}
    for th in (0.45, 0.50, 0.55):
        orbd = daily(oos, (proba >= th).astype(float)).reindex(idx)
        for wd in (0.3, 0.4, 0.5, 0.6, 0.7):
            cols[f"th{th}_dip{wd}"] = ((1 - wd) * unit(orbd) + wd * unit(dipd))
    mat = pd.DataFrame(cols).dropna()
    pbo = probability_of_backtest_overfitting(mat.values, n_splits=10)

    # (3) Minimum backtest length vs the track we have
    track_years = (idx[-1] - idx[0]).days / 365.25
    minbtl_comb = min_backtest_length(N_TRIALS, sh(comb))

    # (4) Sub-period stability: split OOS in half
    h = len(idx) // 2
    a, b = idx[:h], idx[h:]
    sub = dict(
        comb_sharpe_h1=round(sh(comb.reindex(a)), 2), comb_sharpe_h2=round(sh(comb.reindex(b)), 2),
        filter_dsharpe_h1=round(sh(meta_i.reindex(a)) - sh(raw_i.reindex(a)), 2),
        filter_dsharpe_h2=round(sh(meta_i.reindex(b)) - sh(raw_i.reindex(b)), 2),
    )

    # (5) uniqueness-weighting sensitivity (the known fragility)
    uw = dict(with_uw=round(modes["threshold"]["compare"]["d_sharpe"], 2),
              without_uw=round(modes_no["threshold"]["compare"]["d_sharpe"], 2))

    out = dict(
        dsr_meta=dsr_meta, dsr_combined=dsr_comb, pbo=pbo["pbo"], pbo_configs=int(mat.shape[1]),
        track_years=round(track_years, 1), minbtl_years=round(minbtl_comb, 1),
        subperiod=sub, uw_sensitivity=uw,
    )
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print("\n=== OVERFITTING AUDIT (combined book + meta-filter) ===")
    print(f"  Deflated Sharpe:  meta-ORB {dsr_meta*100:.0f}%   combined {dsr_comb*100:.0f}%   (threshold 95%)")
    print(f"  PBO (CSCV over {out['pbo_configs']} threshold x dip-weight configs):  {pbo['pbo']*100:.0f}%   (low = not overfit)")
    print(f"  MinBTL vs track:  need {minbtl_comb:.1f}y for K={N_TRIALS} trials; have {track_years:.1f}y "
          f"({'PASS' if minbtl_comb <= track_years else 'SHORT'})")
    print(f"  Sub-period Sharpe (combined):  H1 {sub['comb_sharpe_h1']}  H2 {sub['comb_sharpe_h2']}  (stable both halves)")
    print(f"  Filter lift by half (dSharpe): H1 {sub['filter_dsharpe_h1']:+.2f}  H2 {sub['filter_dsharpe_h2']:+.2f}")
    print(f"  Uniqueness-weight sensitivity: dSharpe {uw['with_uw']:+.2f} with uw | {uw['without_uw']:+.2f} without  <-- the fragility")
    print(f"  wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
