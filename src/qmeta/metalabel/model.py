"""Meta-labeling model + bet sizing (Lopez de Prado, AFML ch. 3 & 10).

A secondary classifier predicts P(the primary bet wins); the bet is then sized
by that probability. Training and out-of-sample prediction are strictly
walk-forward (train only on dates before the test window) so there is no lookahead.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_meta_model(X, y, sample_weight=None, n_estimators=200, max_depth=4,
                     min_samples_leaf=20, seed=0):
    """Balanced random forest meta-classifier."""
    m = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1,
    )
    m.fit(np.asarray(X), np.asarray(y), sample_weight=sample_weight)
    return m


def bet_size(p, mode="linear", floor=0.0):
    """Map P(win) -> bet fraction in [0,1].
    threshold: 1 if p>=0.5 else 0. linear: clip((p-0.5)*2, 0, 1). prob: clip(p,0,1).
    `floor`: sizes below it are set to 0 (skip low-confidence bets)."""
    p = np.asarray(p, dtype=float)
    if mode == "threshold":
        s = (p >= 0.5).astype(float)
    elif mode == "linear":
        s = np.clip((p - 0.5) * 2.0, 0.0, 1.0)
    elif mode == "prob":
        s = np.clip(p, 0.0, 1.0)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return np.where(s < floor, 0.0, s)


def oos_walk_forward_proba(df, feat_cols, y_col, date_col, min_train, refit,
                           sw_col=None, seed=0):
    """Expanding-window walk-forward OOS P(win). At each step, train on all rows
    with date-index < `start` and predict the next `refit` dates. Rows before the
    first `min_train` dates keep NaN (never predicted out-of-sample). No lookahead."""
    d = df.sort_values(date_col).reset_index(drop=True)
    dates = np.array(sorted(d[date_col].unique()))
    pos = {v: i for i, v in enumerate(dates)}
    dpos = d[date_col].map(pos).to_numpy()
    proba = np.full(len(d), np.nan)
    start = min_train
    while start < len(dates):
        tr = dpos < start
        te = (dpos >= start) & (dpos < start + refit)
        if te.any() and tr.sum() > 50 and d.loc[tr, y_col].nunique() > 1:
            Xtr = d.loc[tr, feat_cols].to_numpy()
            ytr = d.loc[tr, y_col].to_numpy()
            sw = d.loc[tr, sw_col].to_numpy() if sw_col else None
            m = train_meta_model(Xtr, ytr, sample_weight=sw, seed=seed)
            proba[te] = m.predict_proba(d.loc[te, feat_cols].to_numpy())[:, 1]
        start += refit
    d["proba"] = proba
    return d
