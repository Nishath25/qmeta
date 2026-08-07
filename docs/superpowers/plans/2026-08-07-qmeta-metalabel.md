# qmeta Phase 2 — ML Alpha Layer (meta-labeling on ORB) — Plan

> Executed inline (TDD, commit per module). Spec: `docs/superpowers/specs/2026-08-07-qmeta-metalabel-design.md`.

**Global constraints:** zero lookahead (features known at entry; walk-forward trains on PAST dates only);
overlap-aware (uniqueness weights); honest OOS verdict (report the null if meta-labeling doesn't help);
ORB features come from `blotter_atr3.parquet` (the ATR_MAX=3.0 build) + `w` from `raw_har_trades_atr3.parquet`.

### Task 1 — `qmeta/labeling/triple_barrier.py` (+ tests)
- `triple_barrier(prices, entry_idx, pt, sl, max_h, side=1) -> (outcome, exit_idx, ret)` — first of
  profit-take / stop / time barrier (AFML ch.3). Reference test: synthetic up path → 'pt'; down → 'sl'; flat → 'time'.
- `meta_label(realized_return, side=1) -> int` (1 if the bet made money); `meta_labels_from_R(R)` vectorized.

### Task 2 — `qmeta/cv/purged_kfold.py` (+ tests)
- `PurgedKFold(n_splits, t1, embargo_pct).split(X)` — AFML snippet 7.3: train = events ending before the
  test window starts, or starting after (test-max-t1 + embargo). Leakage test: no train label span overlaps
  the test span; embargo removes the right count.

### Task 3 — `qmeta/sampling/bootstrap.py` (+ tests)
- `get_indicator_matrix(bar_index, t1)`, `average_uniqueness(ind)`, `sequential_bootstrap(ind, n, rng)`.
- Reference: AFML ch.4 example `t1={0:2,2:3,4:5}` → avg uniqueness ≈ [0.833, 0.75, 1.0].

### Task 4 — `qmeta/metalabel/model.py` (+ tests, needs scikit-learn)
- `train_meta_model(X, y, sample_weight, ...) -> RandomForestClassifier` (balanced).
- `bet_size(p, mode in {threshold,linear,prob}, floor) -> [0,1]`.
- `oos_walk_forward_proba(df, feat_cols, y_col, date_col, min_train, refit, sw_col) -> df+proba` —
  expanding window, trains only on dates < test window (no lookahead). Test on synthetic separable data:
  OOS accuracy > 0.6; bet_size monotone in p.

### Task 5 — `qmeta/metalabel/evaluate.py` (+ tests)
- `stream_stats(daily, n_trials) -> dict(sharpe, dsr, maxdd, hit, n)` (reuses `qmeta.selection`).
- `compare(raw_daily, meta_daily) -> dict` with deltas.

### Task 6 — `scripts/build_orb_features.py`
- Merge `blotter_atr3` (direction, atr_ratio, entry, risk, R) + `raw_har_trades_atr3` (w) on
  (date,ticker,entry,risk). Features: `direction, atr_ratio, risk_pct=risk/entry, w, dow`. Label `y=(R>0)`.
  Keep `date, ticker, R, w` for reconstruction. Cache to `scratch/orb_features.parquet`.

### Task 7 — `scripts/run_metalabel.py`
- Walk-forward OOS P(win) (min_train≈504d/2y, refit≈63d). Build meta-labeled daily returns
  `Σ bet_size(p)·w·(R−0.02)·0.0025` vs raw `Σ w·(R−0.02)·0.0025`. Compare Sharpe/DSR/maxDD/hit + per-year;
  feature importances. Write `scratch/metalabel.json` + console verdict.

### Task 8 — `scripts/make_metalabel_dashboard.py`
- Self-contained HTML: raw-vs-metalabeled equity + metric deltas + P(win) calibration + feature importances.

Final: full suite green, gauntlet parity intact, whole-branch review, honest verdict.
