"""Build the two live return streams the scorecard grades: the ORB fund and
the dip diversifier. Reads the strategy repos by absolute path; writes only a
git-ignored cache. No lookahead - realized history only."""
from pathlib import Path
import numpy as np
import pandas as pd

ORB_TRADES = Path(r"C:\Users\madas\orb-strategy\data\raw_har_trades_atr3.parquet")
DIP_DIR = Path(r"C:\Users\madas\dip-recovery\data")
CACHE = Path(r"C:\Users\madas\qmeta\scratch")
ORB_R_TO_RET = 0.0025  # 0.25%/net-R -> ~15% annual vol (matches the WFT)

# dip sim params (from the settled $50k WFT)
VOL_MAX, VSPIKE_MAX, ADD_STEP, MAX_TR, MAX_HOLD, CBPS, BASE = 0.03, 3.0, 0.10, 4, 252, 15 / 10_000, 0.02


def load_orb() -> pd.Series:
    t = pd.read_parquet(ORB_TRADES)
    t["date"] = pd.to_datetime(t["date"])
    net_R = t.assign(x=(t["R"] - 0.02) * t["w"]).groupby("date")["x"].sum().sort_index()
    return (net_R * ORB_R_TO_RET).rename("orb")


def load_orb_ticker_matrix(min_days: int = 60) -> pd.DataFrame:
    """date x ticker matrix of daily net-R*scale (0 when no trade). Columns
    restricted to tickers active on >= min_days days (for a stable PBO)."""
    t = pd.read_parquet(ORB_TRADES)
    t["date"] = pd.to_datetime(t["date"])
    t["x"] = (t["R"] - 0.02) * t["w"] * ORB_R_TO_RET
    piv = t.pivot_table(index="date", columns="ticker", values="x", aggfunc="sum").sort_index()
    active = (piv.abs() > 0).sum(axis=0)
    keep = active[active >= min_days].index
    return piv[keep].fillna(0.0)


def load_orb_config_variants(wcaps=(1.0, 1.5, 2.0, 3.0, 1e9), costs=(0.0, 0.02, 0.04)) -> pd.DataFrame:
    """A family of ORB strategy CONFIGURATIONS as daily-return columns, for a
    genuine strategy-overfitting PBO. Each column caps the signal weight w at
    `wcap` and deducts `cost` R/trade (both are real tuned hyperparameters, per
    the gauntlet trial audit). Capping w changes which trades dominate, so the
    columns have genuinely different time profiles -- the right input for CSCV."""
    t = pd.read_parquet(ORB_TRADES)
    t["date"] = pd.to_datetime(t["date"])
    cols = {}
    for cap in wcaps:
        wc = np.minimum(t["w"].to_numpy(), cap)
        for c in costs:
            x = (t["R"].to_numpy() - c) * wc * ORB_R_TO_RET
            s = pd.Series(x, index=t["date"]).groupby(level=0).sum()
            cols[f"wcap{cap:g}_cost{c:g}"] = s
    return pd.DataFrame(cols).sort_index().fillna(0.0)


def _dip_portfolio_equity() -> pd.Series:
    """Capital-constrained dip sim -> equity series (fraction of start capital)."""
    panel = pd.read_parquet(DIP_DIR / "panel.parquet")
    panel["d"] = pd.to_datetime(panel["d"])
    panel = panel.sort_values(["ticker", "d"])
    db = pd.read_parquet(DIP_DIR / "daily_bars.parquet", columns=["ticker", "d", "dollar_volume"])
    db["d"] = pd.to_datetime(db["d"])
    DTS, OPEN, CLOSE = {}, {}, {}
    for tk, g in panel.groupby("ticker"):
        DTS[tk] = g["d"].to_numpy(); OPEN[tk] = g["open"].to_numpy(); CLOSE[tk] = g["close"].to_numpy()
    base = panel[panel.eligible_tier].merge(db, on=["ticker", "d"], how="left")
    base["vspike"] = base["dollar_volume"] / base["adv20"]
    base = base[(base.sigma20 <= VOL_MAX) & (base.vspike <= VSPIKE_MAX)]
    sig = base[base.ret_1d.between(-.08, -.04)].copy()
    sig["tgt"] = sig["prev_close"]; sig = sig.sort_values("d")
    PX = {tk: {d: k for k, d in enumerate(DTS[tk])} for tk in set(sig.ticker) | {"SPY"}}
    cal = list(np.sort(DTS["SPY"]))
    sbd = {}
    for tk, d, tg in zip(sig.ticker, sig.d, sig.tgt):
        sbd.setdefault(pd.Timestamp(d), []).append((tk, tg))
    cash = 1.0; pos = {}; pend = []; cid = 0; eq = []; onm = set()

    def px(tk, dt, w):
        j = PX.get(tk, {}).get(dt)
        return None if j is None else (OPEN[tk] if w == "o" else CLOSE[tk])[j]

    for ci, dt in enumerate(cal):
        for o in pend:
            p = px(o["tk"], dt, "o")
            if p is None or p <= 0:
                continue
            if o["k"] == "b":
                if cash >= o["a"] * (1 + CBPS):
                    cash -= o["a"] * (1 + CBPS); q = pos.get(o["id"])
                    if q:
                        q["sh"] += o["a"] / p; q["u"] += o["a"]
                    else:
                        pos[o["id"]] = dict(tk=o["tk"], sh=o["a"] / p, u=o["a"], first=p, tgt=o["t"], st=ci, last=p); onm.add(o["tk"])
            else:
                q = pos.pop(o["id"], None)
                if q:
                    cash += q["sh"] * p * (1 - CBPS); onm.discard(q["tk"])
        pend = []; mv = 0.0
        for pid, q in list(pos.items()):
            c = px(q["tk"], dt, "c"); c = q["last"] if c is None else c; q["last"] = c; mv += q["sh"] * c
            if c >= q["tgt"] or (ci - q["st"]) >= MAX_HOLD:
                pend.append(dict(k="s", id=pid, tk=q["tk"]))
            elif q["u"] < BASE * MAX_TR - 1e-9 and c <= q["first"] * (1 - ADD_STEP * round(q["u"] / BASE)):
                pend.append(dict(k="b", id=pid, tk=q["tk"], a=BASE, t=q["tgt"]))
        eq.append(cash + mv)
        for tk, tg in sbd.get(pd.Timestamp(dt), []):
            if tk in onm or pd.isna(tg):
                continue
            j = PX[tk].get(dt)
            if j is None or j < 1:
                continue
            cid += 1; pend.append(dict(k="b", id=cid, tk=tk, a=BASE, t=tg))
    return pd.Series(eq, index=pd.DatetimeIndex(cal), name="dip_equity")


def load_dip(cache: bool = True) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "dip_equity.parquet"
    if cache and f.exists():
        eq = pd.read_parquet(f)["dip_equity"]
    else:
        eq = _dip_portfolio_equity()
        if cache:
            eq.to_frame().to_parquet(f)
    return eq.pct_change().dropna().rename("dip")


if __name__ == "__main__":
    o = load_orb()
    print(f"ORB: {len(o)} days {o.index[0].date()}..{o.index[-1].date()} mean={o.mean():.2e} vol_ann={o.std()*(252**.5):.1%}")
    m = load_orb_ticker_matrix()
    print(f"ORB matrix: {m.shape[0]} days x {m.shape[1]} tickers")
    d = load_dip()
    print(f"DIP: {len(d)} days {d.index[0].date()}..{d.index[-1].date()} mean={d.mean():.2e} vol_ann={d.std()*(252**.5):.1%}")
