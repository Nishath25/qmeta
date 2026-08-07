"""Institutional stress-test gauntlet for the ORB fund, the dip diversifier, and
the combined book -- built around the user's LdP risk framework:
  1. Friction gauntlet: tiered cost sweep + breakeven frictional cost per trade.
  2. Raw vs Net vs Deflated Sharpe (the headline deliverable).
  3. Market-regime & crisis analysis (high/low vol, named crashes).
  4. Monte-Carlo path stress (block bootstrap): distribution of maxDD and Sharpe.
  5. Worst-case realized + Triple-Penance projected-vs-realized drawdown.
Run: python scripts/stress_test.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.selection import sharpe_per_obs, annualize_sr, moments, deflated_sharpe_ratio
from qmeta.selection.drawdown import triple_penance
from load_streams import load_orb, load_dip, ORB_TRADES, ORB_R_TO_RET

DIP_DIR = Path(r"C:\Users\madas\dip-recovery\data")
OUT = Path(r"C:\Users\madas\qmeta\scratch\stress.json")
ANN = np.sqrt(252)
N_TRIALS = 19
COST_R = 0.02  # the ORB's live per-trade cost, in R
RNG = np.random.default_rng(7)


def _sharpe(r):
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std() * ANN) if r.std() > 0 else 0.0


def _maxdd(r):
    eq = np.cumprod(1 + np.asarray(r, dtype=float))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def load_spy():
    db = pd.read_parquet(DIP_DIR / "daily_bars.parquet", columns=["ticker", "d", "close"])
    spy = db[db.ticker == "SPY"].copy()
    spy["d"] = pd.to_datetime(spy["d"])
    spy = spy.sort_values("d").set_index("d")["close"]
    return spy.pct_change().dropna().rename("spy")


def orb_daily_at_cost(cost_r):
    t = pd.read_parquet(ORB_TRADES)
    t["date"] = pd.to_datetime(t["date"])
    x = (t["R"] - cost_r) * t["w"] * ORB_R_TO_RET
    return pd.Series(x.to_numpy(), index=t["date"]).groupby(level=0).sum()


def friction_gauntlet():
    t = pd.read_parquet(ORB_TRADES)
    risk_over_entry = float((t["risk"] / t["entry"]).mean())  # avg stop as fraction of price
    costs = np.linspace(0.0, 0.20, 41)
    sharpes = [_sharpe(orb_daily_at_cost(c)) for c in costs]
    sharpes = np.array(sharpes)
    # breakeven cost: first crossing of Sharpe=0 (linear interp)
    be = None
    for i in range(1, len(costs)):
        if sharpes[i - 1] > 0 >= sharpes[i]:
            f = sharpes[i - 1] / (sharpes[i - 1] - sharpes[i])
            be = float(costs[i - 1] + f * (costs[i] - costs[i - 1]))
            break
    be_bps = be * risk_over_entry * 1e4 if be is not None else None
    live_bps = COST_R * risk_over_entry * 1e4
    return dict(costs=[round(float(c), 3) for c in costs],
                sharpes=[round(float(s), 3) for s in sharpes],
                breakeven_R=round(be, 3) if be else None,
                breakeven_bps=round(be_bps, 1) if be_bps else None,
                live_cost_R=COST_R, live_cost_bps=round(live_bps, 2),
                headroom_x=round(be / COST_R, 1) if be else None)


def build_streams():
    orb_raw = orb_daily_at_cost(0.0)       # frictionless ORB
    orb_net = orb_daily_at_cost(COST_R)    # ORB after live cost
    dip = load_dip()                       # dip already includes 15bps/fill
    j = pd.concat([orb_net.rename("orb"), dip.rename("dip")], axis=1, join="inner", sort=False).sort_index()
    u_orb = j["orb"] / j["orb"].std()
    u_dip = j["dip"] / j["dip"].std()
    comb_u = 0.5 * u_orb + 0.5 * u_dip
    comb = comb_u / comb_u.std() * j["orb"].std()   # combined at ORB's vol level (50/50 equal-risk)
    return dict(orb_raw=orb_raw, orb_net=orb_net, dip=dip, combined=comb, aligned=j)


def rnd_table(streams):
    def dsr(r):
        r = np.asarray(r.dropna(), dtype=float)
        sk, ku = moments(r)
        return deflated_sharpe_ratio(sharpe_per_obs(r), len(r), sk, ku, N_TRIALS, 1.0 / len(r))
    rows = {}
    rows["ORB fund"] = dict(raw=_sharpe(streams["orb_raw"]), net=_sharpe(streams["orb_net"]),
                            deflated=dsr(streams["orb_net"]))
    rows["Dip diversifier"] = dict(raw=None, net=_sharpe(streams["dip"]), deflated=dsr(streams["dip"]))
    rows["Combined (50/50)"] = dict(raw=None, net=_sharpe(streams["combined"]), deflated=dsr(streams["combined"]))
    return rows


CRISES = {
    "2018 Q4 selloff": ("2018-10-01", "2018-12-24"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
    "2025-26 drawdown": ("2025-11-01", "2026-07-31"),
}


def regime_analysis(streams):
    spy = load_spy()
    spy_vol = (spy.rolling(20).std() * ANN).dropna()
    med = float(spy_vol.median())
    out = {"spy_vol_median": round(med, 3), "vol_split": {}, "crises": {}}
    names = {"ORB": streams["orb_net"], "dip": streams["dip"], "combined": streams["combined"]}
    for label, hi in [("high-vol (SPY vol > median)", True), ("low-vol", False)]:
        days = spy_vol[(spy_vol > med) == hi].index
        out["vol_split"][label] = {n: round(_sharpe(s.reindex(days).dropna()), 2) for n, s in names.items()}
    for cname, (a, b) in CRISES.items():
        seg = {}
        for n, s in names.items():
            w = s[(s.index >= a) & (s.index <= b)]
            seg[n] = round(float((1 + w).prod() - 1) * 100, 1) if len(w) else None
        spyw = spy[(spy.index >= a) & (spy.index <= b)]
        seg["SPY"] = round(float((1 + spyw).prod() - 1) * 100, 1) if len(spyw) else None
        out["crises"][cname] = seg
    return out


def monte_carlo(series, n_boot=2000, block=21):
    r = np.asarray(series.dropna(), dtype=float)
    n = len(r)
    nblocks = int(np.ceil(n / block))
    dds, shs, terms = [], [], []
    for _ in range(n_boot):
        starts = RNG.integers(0, n - block, size=nblocks)
        path = np.concatenate([r[s:s + block] for s in starts])[:n]
        dds.append(_maxdd(path)); shs.append(_sharpe(path)); terms.append(float(np.prod(1 + path)))
    q = lambda a, p: float(np.percentile(a, p))
    return dict(maxdd_p05=round(q(dds, 5) * 100, 1), maxdd_p50=round(q(dds, 50) * 100, 1),
                maxdd_p95=round(q(dds, 95) * 100, 1),
                sharpe_p05=round(q(shs, 5), 2), sharpe_p50=round(q(shs, 50), 2),
                sharpe_p95=round(q(shs, 95), 2),
                p_negative=round(float(np.mean(np.array(terms) < 1.0)) * 100, 1),
                dd_hist=[round(x * 100, 1) for x in dds[:1500]])


def worst_case(series):
    r = series.dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    uw = (dd < 0).astype(int)
    # longest underwater run (in days)
    longest, cur = 0, 0
    for v in uw.values:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    tp = triple_penance(float(r.mean()), float(r.std()), prob=0.95)
    arr = np.asarray(r, dtype=float)
    return dict(worst_day=round(float(arr.min()) * 100, 2),
                worst_month=round(float(((1 + r).resample("ME").prod() - 1).min()) * 100, 1),
                cvar5=round(float(arr[arr <= np.percentile(arr, 5)].mean()) * 100, 2),
                maxdd=round(_maxdd(arr) * 100, 1),
                longest_underwater_days=int(longest),
                tp_proj_maxdd=round(tp["max_dd"] * 100, 1),
                tp_proj_tuw_years=round(tp["max_tuw"] / 252, 1))


def main():
    streams = build_streams()
    out = dict(
        friction=friction_gauntlet(),
        sharpe_table=rnd_table(streams),
        regimes=regime_analysis(streams),
        monte_carlo=dict(ORB=monte_carlo(streams["orb_net"]), combined=monte_carlo(streams["combined"])),
        worst_case=dict(ORB=worst_case(streams["orb_net"]), dip=worst_case(streams["dip"]),
                        combined=worst_case(streams["combined"])),
        window=dict(start=str(streams["aligned"].index[0].date()), end=str(streams["aligned"].index[-1].date())),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))

    f = out["friction"]
    print(f"\n=== FRICTION GAUNTLET (ORB) ===")
    print(f"  live cost {f['live_cost_R']}R (~{f['live_cost_bps']} bps/trade) | "
          f"breakeven {f['breakeven_R']}R (~{f['breakeven_bps']} bps) | headroom {f['headroom_x']}x")
    print(f"\n=== RAW vs NET vs DEFLATED SHARPE ===")
    for k, v in out["sharpe_table"].items():
        raw = "  n/a " if v["raw"] is None else f"{v['raw']:.2f}"
        print(f"  {k:18s}  raw {raw}   net {v['net']:.2f}   deflated(DSR) {v['deflated']*100:.0f}%")
    print(f"\n=== CRISIS PERFORMANCE (total return %) ===")
    for c, seg in out["regimes"]["crises"].items():
        print(f"  {c:20s} " + "  ".join(f"{k} {('n/a' if seg[k] is None else f'{seg[k]:+.1f}%')}" for k in ["ORB", "dip", "combined", "SPY"]))
    print(f"\n=== MONTE-CARLO PATH STRESS (block bootstrap) ===")
    for k, v in out["monte_carlo"].items():
        print(f"  {k:9s} maxDD p05/p50 {v['maxdd_p05']}%/{v['maxdd_p50']}%  "
              f"Sharpe p05/p50/p95 {v['sharpe_p05']}/{v['sharpe_p50']}/{v['sharpe_p95']}  P(loss) {v['p_negative']}%")
    print(f"\n=== WORST CASE (realized vs Triple-Penance projection) ===")
    for k, v in out["worst_case"].items():
        print(f"  {k:9s} maxDD {v['maxdd']}% (proj {v['tp_proj_maxdd']}%)  worst day {v['worst_day']}%  "
              f"CVaR5 {v['cvar5']}%  longest underwater {v['longest_underwater_days']}d")
    print(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    main()
