# qmeta Phase 3 — Portfolio Construction (HRP + NCO) — Design + Plan

> Executed inline (TDD). Replaces the hand-tuned 50/50 ORB+dip blend with robust allocators.

**Goal:** implement Hierarchical Risk Parity (HRP, Lopez de Prado 2016) and Nested-Clustered
Optimization (NCO, Lopez de Prado 2019), plus the baselines they're judged against, and test whether
they beat naive equal-risk and (overfit-prone) Markowitz min-variance OUT-OF-SAMPLE on the user's
real return streams.

**Global constraints:** OOS/rolling evaluation only (estimate covariance on a trailing window,
allocate, hold, roll — never fit and score on the same window); weights sum to 1; report the null if
HRP/NCO don't beat equal-weight.

### Task 1 — `qmeta/portfolio/baselines.py` (+ tests)
- `cov_to_corr(cov)`, `equal_weights(n)`, `inverse_variance_weights(cov)`,
  `min_variance_weights(cov)` (= Σ⁻¹1 / 1ᵀΣ⁻¹1, pinv), `max_sharpe_weights(cov, mu)` (∝ Σ⁻¹μ).
- Reference: `diag([4,1])` → inverse-variance and min-variance both = `[0.2, 0.8]`.

### Task 2 — `qmeta/portfolio/hrp.py` (+ tests)
- `hrp_weights(cov)`: correlation→distance `sqrt((1-corr)/2)`, single-linkage tree,
  quasi-diagonalization, recursive bisection (AFML ch.16).
- Reference: `hrp_weights(diag([4,1]))` = `[0.2, 0.8]` (reduces to inverse-variance for 2 assets);
  block-correlated 4-asset matrix → positive weights summing to 1, more diversified (higher effective-N)
  than min-variance.

### Task 3 — `qmeta/portfolio/nco.py` (+ tests)
- `nco_weights(cov, mu=None, max_k=None)`: KMeans-cluster the correlation (silhouette picks k),
  intra-cluster optimal weights, reduce to a super-covariance, inter-cluster weights, combine
  (AFML / "A Robust Estimator of the Efficient Frontier" 2019). Falls back to min-variance for n<4.
- Reference: 2-block covariance → weights sum to 1, finite, respects the block structure.

### Task 4 — `scripts/run_portfolio.py`
- Panel A (headline): `{ORB fund, dip}` daily returns → HRP vs 50/50 vs inverse-var → the principled split.
- Panel B (multi-asset): the 8 ORB per-instrument streams + dip (9 sleeves) → rolling monthly OOS
  (252d trailing cov) comparison of {equal, inverse-var, min-variance, HRP, NCO} on realized
  Sharpe / vol / maxDD / effective-N. The HRP thesis: lower OOS variance than Markowitz min-variance.
- Write `scratch/portfolio.json`.

### Task 5 — `scripts/make_portfolio_dashboard.py`
- Self-contained HTML: OOS equity by method, a metrics table, the cluster dendrogram/heatmap, and the
  ORB-vs-dip principled split vs 50/50.

Final: suite green, gauntlet parity intact, whole-branch review, honest verdict.
