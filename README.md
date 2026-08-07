# qmeta

A reusable **quantitative meta-strategy** library — tested implementations of the strategy-selection,
machine-learning, and portfolio-construction techniques from the work of **Marcos López de Prado**
(*Advances in Financial Machine Learning* and related papers), packaged as pure functions and
**applied to real trading strategies** with honest, out-of-sample verdicts.

`qmeta` is deliberately standalone: it has **no dependency on any strategy project**. Strategy repos
import *from* `qmeta`, never the other way around.

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                 # 64 tests
```

Requires Python ≥ 3.10 (numpy, scipy, pandas; scikit-learn for the meta-labeling layer).

## What's inside

```
src/qmeta/
  selection/   grade whether an edge is real
  labeling/  cv/  sampling/  metalabel/   improve the edge (meta-labeling)
  portfolio/   combine robustly (HRP / NCO)
```

### `qmeta.selection` — is the edge real?
| function | paper |
|---|---|
| `probabilistic_sharpe_ratio`, `min_track_record_length` | PSR / MinTRL (Bailey & LdP 2012) |
| `expected_max_sharpe` | the "False Strategy" theorem (Bailey & LdP 2014) |
| `deflated_sharpe_ratio`, `min_backtest_length` | DSR / MinBTL |
| `probability_of_backtest_overfitting` | PBO via CSCV (Bailey et al. 2015) |
| `combined_max_sharpe`, `improves_max_sharpe`, `indifference_curve` | Strategy-Approval theorem |
| `triple_penance`, `max_drawdown_quantile`, `max_time_under_water` | Triple-Penance (Bailey & LdP 2015) |

### `qmeta.{labeling, cv, sampling, metalabel}` — improve the edge
Triple-Barrier labels, `PurgedKFold` + embargo (AFML ch. 7), Sequential Bootstrap + average
uniqueness (AFML ch. 4), and a walk-forward meta-labeling model (`oos_walk_forward_proba`,
`bet_size`) that predicts *P(a signal wins)* and sizes/skips bets accordingly.

### `qmeta.portfolio` — combine robustly
Hierarchical Risk Parity (`hrp_weights`, LdP 2016), Nested-Clustered Optimization (`nco_weights`,
LdP 2019), and the baselines they're judged against (equal / inverse-variance / min-variance / max-Sharpe).

## Findings on a real book (ORB fund + dip diversifier)

Every result below is computed by the `scripts/` on realized returns, out-of-sample where it matters:

- **Grading:** the ORB fund (Sharpe 0.95) and dip (0.75) are >99% likely to have positive edge (PSR),
  but neither clears the 95% bar once deflated for the number of configurations tried (DSR 74% / 70%).
  Track records are long enough (MinTRL passes). Strategy-Approval confirms the dip is a worthwhile
  diversifier (combined max-Sharpe 1.22).
- **Improving:** meta-labeling the ORB — skipping the breakouts a model flags as losers — lifts
  out-of-sample Sharpe 0.99 → 1.19 and cuts drawdown ~40%. **Caveat (disclosed):** the gain is
  leakage-free but contingent on uniqueness weighting; without it, it's roughly neutral.
- **Combining:** out-of-sample, **naive allocation wins** — flat 50/50 (ORB/dip) and equal-weight
  (instrument sleeves) beat HRP/NCO/Markowitz. A clean reproduction of the "1/N beats optimization
  out-of-sample" result. HRP/NCO are correct; their edge is ill-conditioned covariances, absent here.

## Design notes

- **No lookahead:** selection stats are post-hoc; the ML layer trains strictly walk-forward; the
  portfolio contest fits weights on a trailing window and applies them forward.
- **Conventions:** per-observation Sharpe, non-excess kurtosis. `deflated_sharpe_ratio` /
  `expected_max_sharpe` are numerically identical to a hand-verified reference; `triple_penance`
  reproduces the source paper's worked example to 8 decimals.
- Docs: design specs in `docs/superpowers/specs/`, implementation plans in `docs/superpowers/plans/`.

## License

MIT — see [LICENSE](LICENSE).
