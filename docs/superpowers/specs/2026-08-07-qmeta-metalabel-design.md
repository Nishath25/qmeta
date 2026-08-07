# qmeta — Phase 2: ML Alpha Layer (`qmeta.{labeling,cv,sampling,metalabel}`) — Design

**Date:** 2026-08-07
**Status:** Design (awaiting user review)

## Goal

Build the machine-learning layer of `qmeta` — Triple-Barrier labeling, Purged K-Fold CV with
embargo, Sequential Bootstrap uniqueness weighting, and Meta-labeling/bet-sizing — and apply it to
the **ORB fund** to answer one honest question: **does a secondary model that predicts "will this
breakout win?" improve the fund's out-of-sample net Sharpe, or not?** (Same no-nonsense standard as
the dip study: if it doesn't beat the raw fund OOS, we report the null.)

## Why this is the "improve" phase

The ORB rule is the *primary model* — it decides direction (long/short breakout). Meta-labeling adds
a *secondary* classifier that predicts P(win) for each signaled trade, then sizes the bet by that
probability (or skips it). Per López de Prado, this raises the F1 score by trading recall for
precision — killing false breakouts, which is where an opening-range strategy bleeds.

## Constraints (carried from Phase 1 + the user's standard)

- **Zero lookahead.** Features use only information available AT the trade's entry timestamp. The
  meta-model is trained walk-forward (only past trades) and validated with **Purged K-Fold + embargo**
  so no same-day/adjacent information leaks from test into train.
- **Overlap-aware.** ORB fires multiple trades per day; those share day-level information. Training
  samples get **uniqueness weights** and are drawn via **Sequential Bootstrap** so the model isn't
  fooled by redundant, non-independent samples.
- **Honest evaluation.** The verdict is OOS net Sharpe / DSR of meta-labeled sizing vs the raw fund,
  not in-sample fit. Report the null if it doesn't help.

## Architecture

```
src/qmeta/
  labeling/triple_barrier.py   # triple-barrier labels + meta-labels
  cv/purged_kfold.py           # PurgedKFold splitter (embargo) + purged walk-forward
  sampling/bootstrap.py        # indicator matrix, average uniqueness, sequential bootstrap
  metalabel/model.py           # train secondary classifier -> P(win); bet_size(p)
  metalabel/evaluate.py        # OOS raw-vs-metalabeled comparison (Sharpe/DSR/hit/DD)
scripts/
  build_orb_features.py        # entry-time feature blotter for the ORB trades (see below)
  run_metalabel.py             # end-to-end: features -> purged-CV meta-model -> scorecard + dashboard
tests/  (one per module, reference-anchored + leakage tests)
```

### `labeling/triple_barrier.py`
- `triple_barrier_labels(close, events, pt, sl, vertical)` — for each event time, the first of
  {upper barrier (pt·σ), lower barrier (sl·σ), vertical (time) barrier} that is touched; returns the
  touch time and the realized sign. (AFML ch. 3.)
- `meta_labels(primary_side, realized_return)` — the binary meta-label: 1 if the primary model's bet
  made money, else 0. For ORB the primary outcome is already realized as `R`, so the meta-label is
  `1[R > 0]`; the triple-barrier module is still provided/tested for general use and to document that
  the ORB's PT/SL/EOD exits ARE a triple barrier.

### `cv/purged_kfold.py`
- `PurgedKFold(n_splits, t1, embargo_pct)` — sklearn-style splitter that purges training observations
  whose label span (`t1`) overlaps the test set, plus an embargo after each test block. (AFML ch. 7.)
- `purged_walk_forward(...)` — expanding-window OOS folds by date for the final evaluation.
- Reference/leakage tests: assert no train index's `t1` falls inside any test interval; embargo removes
  the right count.

### `sampling/bootstrap.py`
- `indicator_matrix(bar_index, t1)` — which bars each label spans.
- `average_uniqueness(ind_matrix)` — per-label average uniqueness (concurrency-adjusted weight).
- `sequential_bootstrap(ind_matrix, n)` — draw samples favoring low-overlap labels. (AFML ch. 4.)
- Reference tests against the small worked example in AFML (a 3-bar indicator matrix with known
  uniqueness).

### `metalabel/model.py` and `evaluate.py`
- `train_meta_model(X, y, sample_weight, cv)` — a gradient-boosted / logistic classifier with
  `PurgedKFold`; returns fitted model + OOS P(win) via cross-val-predict.
- `bet_size(p, mode)` — size in [0,1] from P(win): thresholded, linear, or the AFML
  `bet_size = (p - 0.5)/sqrt(p(1-p))` → CDF form. Skips when p below a floor.
- `evaluate.py` — build the meta-labeled daily return stream (each ORB trade's R × bet_size(p_oos)),
  compare to the raw fund on Sharpe, DSR (via `qmeta.selection`), hit rate, and drawdown, all OOS.

## Data & the one real decision: the feature set

The trade file `raw_har_trades_atr3.parquet` has `date,ticker,R,entry,risk,w` — enough for the
meta-LABEL (`1[R>0]`) but thin on entry-time FEATURES. Two options:

- **(A, recommended) Enriched feature blotter.** `build_orb_features.py` re-runs the ORB blotter
  logic and dumps the entry-time features the strategy already computes internally: `atr_ratio`,
  opening-range width / entry (%), `risk/entry`, direction (long/short), HAR σ forecast, entry
  time-of-day, day-of-week, consecutive-stop state, and `w`. This is the honest, powerful version.
  Needs the local minute data the ORB uses (`basket_minute.parquet` / `basket_hist.parquet`).
- **(B, fallback) Thin features.** Use only the four columns already in the trade file
  (`atr_ratio` if present, `risk`, `entry`, `w`). Weaker, but works with zero new data and still a
  valid meta-labeling demonstration.

Plan: attempt (A); if the minute data isn't reachable, fall back to (B) and say so. Either way the
evaluation is OOS and leakage-controlled.

## Success criteria

1. All four modules implemented, tested (reference-anchored + explicit leakage tests), suite green.
2. `run_metalabel.py` produces an OOS comparison: raw ORB vs meta-labeled ORB (Sharpe, DSR, hit, DD).
3. A visual dashboard of the comparison (same style as the Phase-1 scorecard).
4. An honest verdict: meta-labeling helps (by how much, OOS) or it doesn't.

## Out of scope

- Changing the ORB primary rule itself (we only add a sizing/filter layer on top).
- HRP/NCO portfolio construction (Phase 3).
