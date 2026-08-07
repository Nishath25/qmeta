# qmeta

A reusable **quantitative meta-strategy** library — implementations of the strategy-selection,
machine-learning, and portfolio-construction techniques from the work of Marcos López de Prado
(*Advances in Financial Machine Learning* and related papers), packaged as tested pure functions
and applied to real trading strategies.

`qmeta` is deliberately standalone: it has **no dependency on any strategy project**. The
strategy repos (`orb-strategy`, `dip-recovery`, `lev-etf-ensemble`) import *from* `qmeta`, not the
other way around.

## Layout

```
src/qmeta/
  selection/   Phase 1 — grade whether an edge is real
               PSR, MinTRL, False-Strategy E[maxSR], DSR, MinBTL, PBO,
               Strategy-Approval / Sharpe-indifference, Triple-Penance drawdown & TuW
  labeling/    Phase 2 — Triple-Barrier labels
  cv/          Phase 2 — Purged K-Fold CV + Embargo
  sampling/    Phase 2 — Sequential Bootstrap + uniqueness weights
  metalabel/   Phase 2 — Meta-labeling / bet sizing (Corrective AI)
  portfolio/   Phase 3 — Hierarchical Risk Parity (HRP), Nested-Clustered Optimization (NCO)
```

Design docs live in `docs/superpowers/specs/`; implementation plans in `docs/superpowers/plans/`.

## Status

Phase 1 in design. See `docs/superpowers/specs/2026-08-07-qmeta-selection-design.md`.
