"""Build the ORB meta-labeling feature blotter from the ATR_MAX=3.0 build.
Every feature is known at entry; label y = 1 if the trade won (R>0). No lookahead."""
from pathlib import Path
import pandas as pd

DATA = Path(r"C:\Users\madas\orb-strategy\data")
OUT = Path(r"C:\Users\madas\qmeta\scratch\orb_features.parquet")
FEATURES = ["direction", "atr_ratio", "risk_pct", "w", "dow", "month"]


def build() -> pd.DataFrame:
    raw = pd.read_parquet(DATA / "raw_har_trades_atr3.parquet")   # 3.0 build + HAR weight w
    raw["date"] = pd.to_datetime(raw["date"])
    bl = pd.read_parquet(DATA / "blotter_atr3.parquet")           # entry-time features (direction, atr_ratio)
    bl["date"] = pd.to_datetime(bl["date"])
    keys = ["date", "ticker", "entry", "risk"]
    blk = bl[keys + ["direction", "atr_ratio"]].drop_duplicates(keys)
    df = raw.merge(blk, on=keys, how="left").dropna(subset=["direction", "atr_ratio"]).reset_index(drop=True)
    df["risk_pct"] = df["risk"] / df["entry"]
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["y"] = (df["R"] > 0).astype(int)
    # same-day trades overlap in information -> uniqueness weight = 1/(trades that day)
    df["uw"] = 1.0 / df.groupby("date")["R"].transform("size")
    out = df[["date", "ticker", "R"] + FEATURES + ["y", "uw"]].sort_values("date").reset_index(drop=True)  # w is in FEATURES
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    return out


if __name__ == "__main__":
    o = build()
    print(f"features: {len(o)} trades, {o['date'].nunique()} days, win rate {o['y'].mean():.1%}")
    print(o[FEATURES + ["y"]].describe().round(3).to_string())
