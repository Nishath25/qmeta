# qmeta — Phase 1: Strategy-Selection Toolkit (`qmeta.selection`) — Design

**Date:** 2026-08-07
**Status:** Design (awaiting user review)
**Author:** brainstormed with Claude

## Goal

Build the first phase of `qmeta`, a standalone quantitative meta-strategy library, implementing
the López de Prado **strategy-selection statistics** that grade whether a trading edge is real
(not luck, not overfit) — and run every one of them on the user's two live return streams (the
ORB fund and the dip diversifier), ending in a visual scorecard.

## Context: the whole library (for orientation only — phases 2 & 3 are out of scope here)

`qmeta` will be built in three phases, each its own spec → plan → build:

1. **Phase 1 (THIS SPEC) — `qmeta.selection`:** grade the edge (post-hoc statistics on realized returns).
2. **Phase 2 — `qmeta.{labeling,cv,sampling,metalabel}`:** improve the edge (Triple-Barrier labels,
   Purged CV, Sequential Bootstrap, Meta-labeling/bet-sizing on ORB).
3. **Phase 3 — `qmeta.portfolio`:** combine robustly (HRP, NCO across ORB + dip).

Phase 1 is fully usable on its own the moment it lands.

## Non-negotiable constraints (from the user, carried project-wide)

- **Zero lookahead / zero bias.** Phase 1 is entirely post-hoc statistics on *already-realized*
  return series, so lookahead is structurally impossible here; the heavier no-leakage machinery
  (purged CV, next-bar labels) is Phase 2. This is stated so reviewers hold the line in later phases.
- **Canonical + verified.** Where a formula already exists in the user's `orb-strategy/gauntlet/cpcv.py`
  (`expected_max_sharpe`, `deflated_sharpe`), `qmeta`'s implementation must produce **numerically
  identical** results — verified by a parity test — so the canonical version can be trusted as a drop-in.
- **No dependency on strategy repos.** `qmeta` imports nothing from `orb-strategy`/`dip-recovery`.
  The application scripts read the strategy data files by path; the library itself is pure.

## Tech stack

- Python ≥ 3.10, `numpy`, `scipy`, `pandas`. `pytest` for tests. No sklearn in Phase 1.
- Packaged with `pyproject.toml` (`src/` layout, installable via `pip install -e .`).
- Dashboard is a self-contained HTML file (built per the `dataviz` skill: validated palette,
  light/dark, hover tooltips) — no external assets.

## Architecture

```
src/qmeta/
  __init__.py                 # version, top-level re-exports
  selection/
    __init__.py               # re-exports the public functions below
    sharpe.py                 # PSR, MinTRL, annualization converters
    trials.py                 # E[maxSR] (False-Strategy), DSR, MinBTL, PBO (CSCV)
    approval.py               # Strategy-Approval / Sharpe-indifference curve
    drawdown.py               # Triple-Penance: closed-form MaxDD & Time-under-Water
    returns.py                # small shared helpers: per-obs Sharpe, moments, cleaning
tests/
  test_sharpe.py  test_trials.py  test_approval.py  test_drawdown.py  test_returns.py
scripts/
  load_streams.py             # build ORB + dip daily return series from local files
  scorecard.py                # run every tool on both streams -> scorecard.json + console table
  make_dashboard.py           # render scorecard.json -> self-contained HTML dashboard
```

### Convention: Sharpe frequency

All core formulas operate on the **per-observation** Sharpe `sr = mean(r)/std(r)` (NOT annualized),
matching the LdP papers and the existing `gauntlet` convention. `sharpe.py` provides
`annualize_sr` / `deannualize_sr` (× or ÷ `sqrt(ppy)`, `ppy=252`) so the user's reported annualized
numbers (e.g. ORB 0.95) convert cleanly. Every public function documents which frequency it expects.

## Components

### `returns.py` (shared helpers)
- `clean(r) -> np.ndarray` — drop NaNs, return float array.
- `sharpe_per_obs(r) -> float` — `mean/std` (ddof=1), 0.0 if degenerate.
- `moments(r) -> (skew, kurt)` — sample skew and **non-excess** kurtosis (Gaussian = 3.0), via scipy.

### `sharpe.py`
- `annualize_sr(sr, ppy=252)`, `deannualize_sr(sr_ann, ppy=252)`.
- **`probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_star=0.0) -> float`** — PSR ∈ [0,1]:
  `Φ( (sr - sr_star)·√(n_obs-1) / √(1 - skew·sr + ((kurt-1)/4)·sr²) )`.
  Reference test: with `skew=0, kurt=3`, equals `Φ((sr)·√(n_obs-1) / √(1 + 0.5·sr²))` exactly;
  PSR is increasing in `n_obs` and in `sr`.
- **`min_track_record_length(sr, skew, kurt, sr_star=0.0, prob=0.95) -> float`** — MinTRL (in observations):
  `1 + (1 - skew·sr + ((kurt-1)/4)·sr²)·(Z_prob/(sr - sr_star))²`, `Z_prob = Φ⁻¹(prob)`.
  Returns `inf` if `sr <= sr_star`. Reference: monotonic — larger `sr` ⇒ smaller MinTRL.
- Convenience: `psr_from_returns(r, sr_star=0.0)` and `mintrl_from_returns(r, prob=0.95)` that
  compute `sr`, `skew`, `kurt`, `n_obs` from a return series.

### `trials.py`
- **`expected_max_sharpe(n_trials, var_sr) -> float`** — the "False Strategy" theorem:
  `√(var_sr)·[(1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))]`, `γ` = Euler–Mascheroni.
  **Parity test against `gauntlet.cpcv.expected_max_sharpe`** (identical to ~1e-12).
- **`deflated_sharpe_ratio(sr, n_obs, skew, kurt, n_trials, var_sr) -> float`** — DSR = PSR with
  `sr_star = expected_max_sharpe(n_trials, var_sr)`. **Parity test against `gauntlet.cpcv.deflated_sharpe`.**
- **`min_backtest_length(n_trials, target_sr_ann, ppy=252) -> float`** — MinBTL in **years**:
  `(expected_max_sharpe(n_trials, 1.0) / target_sr_ann)²` (the LdP approximation: expected max
  annualized Sharpe under the null falls like `1/√years`, so solve for years). Reference: increasing
  in `n_trials`, decreasing in `target_sr_ann`.
- **`probability_of_backtest_overfitting(returns_matrix, n_splits=16) -> dict`** — PBO via
  Combinatorially-Symmetric CV (CSCV). Input: a `pandas.DataFrame` shaped (time × trials), one
  column per strategy configuration. Partition time into `n_splits` equal contiguous blocks; over
  all `C(n_splits, n_splits/2)` in/out partitions, pick the in-sample best column by Sharpe, find its
  out-of-sample **rank** `ω ∈ (0,1)` among all columns, form the logit `λ = ln(ω/(1-ω))`;
  `PBO = P(λ ≤ 0)` (in-sample winner lands below the OOS median). Returns
  `dict(pbo, logits, n_partitions)`. Reference tests: (a) one column that dominates every block →
  PBO ≈ 0; (b) i.i.d.-noise columns (constructed deterministically, no RNG-in-test) → PBO ≈ 0.5.
  *Application note:* needs a **matrix of config variants**; the scorecard builds a small honest
  family of ORB sizing/cost variants from the trade file to feed it (documented, not a black box).

### `approval.py` (Strategy-Approval Theorem / Sharpe-Indifference Curve)
Two-strategy result: for an approved book (Sharpe `S_a`) and a candidate (Sharpe `S_n`) with
correlation `ρ`, the max-Sharpe of the optimally-combined book is
`S_c = √((S_a² - 2ρ·S_a·S_n + S_n²)/(1 - ρ²))`. Adding the candidate raises the aggregate Sharpe
**iff `S_n > ρ·S_a`**.
- **`combined_max_sharpe(sr_app, sr_new, corr) -> float`** — the `S_c` formula above.
- **`improves_max_sharpe(sr_app, sr_new, corr) -> bool`** — `sr_new > corr·sr_app`.
- **`max_correlation_for_approval(sr_app, sr_new) -> float`** — `min(1, sr_new/sr_app)`: the highest
  correlation at which the candidate still helps.
- **`indifference_curve(sr_app, level=None, n=101) -> (corr[], sr_new[])`** — for a target combined
  Sharpe `level` (default `sr_app` = the approval boundary `sr_new = ρ·sr_app`), the locus of
  `(ρ, sr_new)` pairs holding `S_c = level`. Used to plot "how good must a candidate be, at each
  correlation, to be worth adding."
  Reference test: `S_a=1, ρ=0.5, S_n=1` ⇒ `S_c = √((1-1+1)/0.75)=√(1.3333)=1.1547`; boundary at
  `ρ=0.5` is `sr_new=0.5`.

### `drawdown.py` (Triple-Penance rule; Bailey & López de Prado, 2014)
Models cumulative performance as an i.i.d. Gaussian process (`phi=0`) or first-order
serially-correlated AR(1) process (`phi≠0`), with per-period mean `mu` and std `sigma`.
- **`max_drawdown_quantile(mu, sigma, prob=0.95, phi=0.0) -> float`** — the MaxDD (positive fraction)
  not exceeded with probability `prob`.
- **`max_time_under_water(mu, sigma, prob=0.95, phi=0.0) -> float`** — the max periods underwater at
  confidence `prob`.
- **`triple_penance(mu, sigma, prob=0.95, phi=0.0) -> dict`** —
  `dict(time_to_maxdd, max_tuw, penance_ratio, max_dd)`, where `penance_ratio` = (recovery time) /
  (time to reach max DD). The rule's headline: for the i.i.d. Gaussian case this ratio ≈ 3
  ("triple penance").
- Convenience `from_returns(r, prob=0.95)` estimating `mu, sigma` (and `phi` for the AR(1) variant)
  from a realized series.
- **Exact closed-form constants** for MaxDD/TuW/penance are pinned in the implementation plan against
  the paper's own worked example; this spec fixes the signatures, the modeling assumptions, and the
  reference anchors (penance_ratio → ≈3 in the symmetric i.i.d. Gaussian limit; MaxDD increasing in
  `sigma`, decreasing in `mu`).

## Data & application (the scripts)

`scripts/load_streams.py` builds two daily return series (no lookahead — all realized history):
- **ORB fund:** `C:\Users\madas\orb-strategy\data\raw_har_trades_atr3.parquet` (13,262 trades,
  cols `date,ticker,R,entry,risk,w`). Daily net R `= Σ w·(R − 0.02)` per day, scaled to the
  0.25%/R (≈15% annual vol) sizing used in the walk-forward. → `orb` daily returns.
- **Dip diversifier:** regenerate the daily equity from the existing portfolio sim
  (`scratchpad/dip_wft_50k.py` logic over `dip-recovery/data/panel.parquet`), then `pct_change`.
  → `dip` daily returns. (If the sim is slow, cache the equity series to a local file that the
  script reuses; the file is git-ignored.)

`scripts/scorecard.py` runs, for **each** stream:
`PSR`, `MinTRL` (obs → also expressed in years), `DSR` + `E[maxSR]` (using the gauntlet's audited
trial count `K` as the default `n_trials`, adjustable), `MinBTL` vs the stream's own annual Sharpe,
`PBO` (over the ORB variant family; dip over its own small filter-variant family), and
`Triple-Penance` MaxDD/TuW. It also runs `approval.*` on the **ORB (approved) + dip (candidate)**
pair. Output: `scorecard.json` + a printed console table.

`scripts/make_dashboard.py` renders `scorecard.json` into a self-contained HTML dashboard: a
per-strategy card (Sharpe, PSR, DSR, MinTRL-in-years, PBO), the Sharpe-indifference curve with the
ORB+dip point plotted, and the projected-vs-empirical drawdown comparison. Published as a claude.ai
Artifact.

## Testing strategy

- **Every formula** has a unit test pinned to a hand-computable reference value or a documented
  limiting case (listed per-function above).
- **Parity tests** for `expected_max_sharpe` and `deflated_sharpe_ratio` against
  `orb-strategy/gauntlet/cpcv.py` (import by path in the test; skip-with-reason if that repo is
  absent, so `qmeta` tests still pass standalone).
- **Determinism:** any test needing a return matrix constructs it arithmetically (no RNG), so
  results are reproducible (`Math.random`/seed-free).
- Target: 100% of public functions covered; `pytest` green before the phase is called done.

## Out of scope (Phase 1)

- Meta-labeling, Triple-Barrier, purged CV, sequential bootstrap (Phase 2).
- HRP / NCO portfolio construction (Phase 3).
- Microstructure (VPIN/BVC/OIB), quantum optimization, SFD/KCA — not planned.
- Re-pointing `gauntlet`/`risk` to import from `qmeta` — a later, optional cleanup once `qmeta` is trusted.

## Success criteria

1. `pip install -e .` works; `pytest` green.
2. Parity with the existing gauntlet DSR / E[maxSR] confirmed.
3. `scripts/scorecard.py` prints a table grading ORB and dip on PSR/DSR/MinTRL/MinBTL/PBO/Triple-Penance,
   plus the ORB+dip approval verdict.
4. A published visual dashboard of the scorecard.
5. Every number is explainable — no black boxes; trial counts and variant families documented.
