# qmeta Phase 1 — `qmeta.selection` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `qmeta.selection` — tested pure-function implementations of the López de Prado strategy-selection statistics — and run them on the user's real ORB and dip return streams, ending in a visual scorecard.

**Architecture:** A standalone `src/`-layout Python package (`qmeta`) with a `selection` subpackage of pure `numpy/scipy` functions, a `pytest` suite pinning each formula to a reference value, and three application scripts (`load_streams`, `scorecard`, `make_dashboard`) that read the strategy data files by path.

**Tech Stack:** Python ≥3.10, numpy, scipy, pandas, pytest. Dashboard is a self-contained HTML file (no external assets).

## Global Constraints

- **Sharpe frequency:** all core formulas take the **per-observation** Sharpe `sr = mean/std` (ddof=1), NOT annualized. `ppy=252` for conversions. Every function's docstring states its frequency.
- **Non-excess kurtosis:** kurtosis is the raw 4th standardized moment (Gaussian = 3.0), via `scipy.stats.kurtosis(..., fisher=False)`.
- **Parity:** `expected_max_sharpe` and `deflated_sharpe_ratio` must equal `orb-strategy/gauntlet/cpcv.py`'s versions to ~1e-12 (verified by a skip-if-absent parity test).
- **No dependency on strategy repos** inside the library; only the `scripts/` read strategy data by absolute path.
- **Determinism in tests:** construct any test matrix arithmetically — no RNG, no `Math.random`, no time-seeded values.
- **No lookahead:** Phase 1 is post-hoc statistics on realized returns; nothing peeks forward.

---

### Task 0: Package scaffold + shared helpers (`returns.py`)

**Files:**
- Create: `C:\Users\madas\qmeta\pyproject.toml`
- Create: `C:\Users\madas\qmeta\src\qmeta\__init__.py`
- Create: `C:\Users\madas\qmeta\src\qmeta\selection\__init__.py`
- Create: `C:\Users\madas\qmeta\src\qmeta\selection\returns.py`
- Test: `C:\Users\madas\qmeta\tests\test_returns.py`

**Interfaces:**
- Produces: `clean(r)->np.ndarray`, `sharpe_per_obs(r)->float`, `moments(r)->(skew, kurt)`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "qmeta"
version = "0.1.0"
description = "Quantitative meta-strategy library (Lopez de Prado techniques)"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "scipy>=1.10", "pandas>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write `src/qmeta/__init__.py`**

```python
"""qmeta — quantitative meta-strategy library (Lopez de Prado techniques)."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `src/qmeta/selection/returns.py`**

```python
"""Shared return-series helpers for the selection toolkit."""
import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurtosis


def clean(r) -> np.ndarray:
    """Return r as a float ndarray with NaNs dropped."""
    a = np.asarray(r, dtype=float)
    return a[~np.isnan(a)]


def sharpe_per_obs(r) -> float:
    """Per-observation Sharpe = mean/std (ddof=1). 0.0 if degenerate."""
    a = clean(r)
    if len(a) < 2:
        return 0.0
    sd = a.std(ddof=1)
    return float(a.mean() / sd) if sd > 0 else 0.0


def moments(r):
    """(skewness, non-excess kurtosis). Gaussian -> (0.0, 3.0)."""
    a = clean(r)
    if len(a) < 2:
        return 0.0, 3.0
    return float(_skew(a)), float(_kurtosis(a, fisher=False))
```

- [ ] **Step 4: Write `src/qmeta/selection/__init__.py`** (re-exports grow as tasks land; start minimal)

```python
from qmeta.selection.returns import clean, sharpe_per_obs, moments

__all__ = ["clean", "sharpe_per_obs", "moments"]
```

- [ ] **Step 5: Write the failing test `tests/test_returns.py`**

```python
import numpy as np
from qmeta.selection.returns import clean, sharpe_per_obs, moments


def test_clean_drops_nan():
    out = clean([1.0, np.nan, 2.0])
    assert list(out) == [1.0, 2.0]


def test_sharpe_per_obs_constant_series_is_zero():
    assert sharpe_per_obs([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_per_obs_known_value():
    r = [0.02, -0.01, 0.03, 0.00, 0.01]
    a = np.asarray(r)
    assert abs(sharpe_per_obs(r) - a.mean() / a.std(ddof=1)) < 1e-12


def test_moments_symmetric_zero_skew():
    sk, ku = moments([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(sk) < 1e-9
    assert ku > 0  # non-excess kurtosis is positive
```

- [ ] **Step 6: Install editable + run tests**

Run: `cd C:\Users\madas\qmeta && python -m pip install -e ".[dev]" && python -m pytest tests/test_returns.py -q`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(selection): package scaffold + returns helpers"
```

---

### Task 1: `sharpe.py` — PSR, MinTRL, annualization

**Files:**
- Create: `src/qmeta/selection/sharpe.py`
- Modify: `src/qmeta/selection/__init__.py` (add re-exports)
- Test: `tests/test_sharpe.py`

**Interfaces:**
- Consumes: `returns.clean`, `returns.sharpe_per_obs`, `returns.moments`.
- Produces: `annualize_sr`, `deannualize_sr`, `probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_star=0.0)`, `min_track_record_length(sr, skew, kurt, sr_star=0.0, prob=0.95)`, `psr_from_returns(r, sr_star=0.0)`, `mintrl_from_returns(r, prob=0.95, sr_star=0.0)`.

- [ ] **Step 1: Write the failing test `tests/test_sharpe.py`**

```python
import math
from scipy.stats import norm
from qmeta.selection.sharpe import (
    annualize_sr, deannualize_sr,
    probabilistic_sharpe_ratio, min_track_record_length,
    psr_from_returns, mintrl_from_returns,
)


def test_annualization_roundtrip():
    assert abs(annualize_sr(deannualize_sr(0.95)) - 0.95) < 1e-12
    assert abs(annualize_sr(0.06) - 0.06 * math.sqrt(252)) < 1e-12


def test_psr_gaussian_reference():
    # sr=0.06 per-obs, T=253, skew=0, kurt=3, sr_star=0 -> ~0.8293
    psr = probabilistic_sharpe_ratio(0.06, 253, 0.0, 3.0, 0.0)
    denom = math.sqrt(1 + 0.5 * 0.06**2)
    expected = norm.cdf(0.06 * math.sqrt(252) / denom)
    assert abs(psr - expected) < 1e-12
    assert abs(psr - 0.8293) < 1e-3


def test_psr_increases_with_n_and_sr():
    base = probabilistic_sharpe_ratio(0.06, 253, 0.0, 3.0)
    assert probabilistic_sharpe_ratio(0.06, 600, 0.0, 3.0) > base
    assert probabilistic_sharpe_ratio(0.09, 253, 0.0, 3.0) > base


def test_mintrl_reference():
    # sr=0.10, skew=0, kurt=3, prob=0.95 -> ~272.9 observations
    val = min_track_record_length(0.10, 0.0, 3.0, 0.0, 0.95)
    z = norm.ppf(0.95)
    expected = 1 + (1 + 0.5 * 0.10**2) * (z / 0.10) ** 2
    assert abs(val - expected) < 1e-9
    assert abs(val - 272.9) < 0.5


def test_mintrl_infinite_when_sr_below_star():
    assert min_track_record_length(0.05, 0.0, 3.0, sr_star=0.05) == float("inf")


def test_from_returns_helpers_run():
    r = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.008]
    assert 0.0 <= psr_from_returns(r) <= 1.0
    assert mintrl_from_returns(r) > 0
```

- [ ] **Step 2: Run test to verify it fails** — `python -m pytest tests/test_sharpe.py -q` → FAIL (module missing).

- [ ] **Step 3: Write `src/qmeta/selection/sharpe.py`**

```python
"""Probabilistic Sharpe Ratio (PSR) and Minimum Track Record Length (MinTRL).

Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
All formulas use the PER-OBSERVATION Sharpe `sr = mean/std` (not annualized)
and NON-EXCESS kurtosis (Gaussian = 3.0).
"""
import math
from scipy.stats import norm

from qmeta.selection.returns import clean, sharpe_per_obs, moments


def annualize_sr(sr: float, ppy: int = 252) -> float:
    return sr * math.sqrt(ppy)


def deannualize_sr(sr_ann: float, ppy: int = 252) -> float:
    return sr_ann / math.sqrt(ppy)


def probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_star=0.0) -> float:
    """P(true SR > sr_star) given estimation error and higher moments. In [0,1]."""
    denom = math.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr)
    stat = (sr - sr_star) * math.sqrt(n_obs - 1.0) / denom
    return float(norm.cdf(stat))


def min_track_record_length(sr, skew, kurt, sr_star=0.0, prob=0.95) -> float:
    """Number of observations needed for PSR(sr_star) to reach `prob`."""
    if sr <= sr_star:
        return float("inf")
    z = norm.ppf(prob)
    factor = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    return 1.0 + factor * (z / (sr - sr_star)) ** 2


def psr_from_returns(r, sr_star=0.0) -> float:
    a = clean(r)
    sk, ku = moments(a)
    return probabilistic_sharpe_ratio(sharpe_per_obs(a), len(a), sk, ku, sr_star)


def mintrl_from_returns(r, prob=0.95, sr_star=0.0) -> float:
    a = clean(r)
    sk, ku = moments(a)
    return min_track_record_length(sharpe_per_obs(a), sk, ku, sr_star, prob)
```

- [ ] **Step 4: Add re-exports to `src/qmeta/selection/__init__.py`**

```python
from qmeta.selection.returns import clean, sharpe_per_obs, moments
from qmeta.selection.sharpe import (
    annualize_sr, deannualize_sr,
    probabilistic_sharpe_ratio, min_track_record_length,
    psr_from_returns, mintrl_from_returns,
)

__all__ = [
    "clean", "sharpe_per_obs", "moments",
    "annualize_sr", "deannualize_sr",
    "probabilistic_sharpe_ratio", "min_track_record_length",
    "psr_from_returns", "mintrl_from_returns",
]
```

- [ ] **Step 5: Run tests** — `python -m pytest tests/test_sharpe.py -q` → PASS.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(selection): PSR + MinTRL"`

---

### Task 2: `trials.py` — E[maxSR] (False Strategy), DSR, MinBTL

**Files:**
- Create: `src/qmeta/selection/trials.py`
- Modify: `src/qmeta/selection/__init__.py`
- Test: `tests/test_trials.py`, `tests/test_parity_gauntlet.py`

**Interfaces:**
- Consumes: `sharpe.probabilistic_sharpe_ratio`.
- Produces: `expected_max_sharpe(n_trials, var_sr)`, `deflated_sharpe_ratio(sr, n_obs, skew, kurt, n_trials, var_sr)`, `min_backtest_length(n_trials, target_sr_ann)`.

- [ ] **Step 1: Write failing test `tests/test_trials.py`**

```python
import math
from scipy.stats import norm
from qmeta.selection.trials import (
    expected_max_sharpe, deflated_sharpe_ratio, min_backtest_length,
)

EULER = 0.5772156649015329


def test_expected_max_sharpe_formula():
    n, var = 19, 0.5
    sd = math.sqrt(var)
    z1 = norm.ppf(1 - 1.0 / n)
    z2 = norm.ppf(1 - 1.0 / (n * math.e))
    expected = sd * ((1 - EULER) * z1 + EULER * z2)
    assert abs(expected_max_sharpe(n, var) - expected) < 1e-12


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 1.0) > expected_max_sharpe(10, 1.0)
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_dsr_is_psr_at_deflated_threshold():
    from qmeta.selection.sharpe import probabilistic_sharpe_ratio
    star = expected_max_sharpe(20, 1.0 / 2000)
    assert abs(
        deflated_sharpe_ratio(0.06, 2000, 0.0, 3.0, 20, 1.0 / 2000)
        - probabilistic_sharpe_ratio(0.06, 2000, 0.0, 3.0, star)
    ) < 1e-12


def test_min_backtest_length_reference():
    # 19 trials, target annual SR 0.95 -> ~3.9 years
    y = min_backtest_length(19, 0.95)
    assert 3.5 < y < 4.3
    assert min_backtest_length(100, 0.95) > y  # more trials -> longer needed
```

- [ ] **Step 2: Run test** → FAIL.

- [ ] **Step 3: Write `src/qmeta/selection/trials.py`**

```python
"""Multiple-testing corrections: the False Strategy theorem (E[max SR]),
the Deflated Sharpe Ratio (DSR), and Minimum Backtest Length (MinBTL).

Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" and
"Pseudo-Mathematics and Financial Charlatanism" (MinBTL).
"""
import math
from scipy.stats import norm

from qmeta.selection.sharpe import probabilistic_sharpe_ratio

EULER = 0.5772156649015329  # Euler-Mascheroni


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max Sharpe] across n_trials independent strategies under the null,
    where var_sr is the variance of the trial Sharpe estimates."""
    if n_trials < 2:
        return 0.0
    sd = math.sqrt(var_sr) if var_sr > 0 else 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sd * ((1.0 - EULER) * z1 + EULER * z2)


def deflated_sharpe_ratio(sr, n_obs, skew, kurt, n_trials, var_sr) -> float:
    """PSR with the benchmark set to E[max SR] under multiple testing."""
    sr_star = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_star=sr_star)


def min_backtest_length(n_trials: int, target_sr_ann: float) -> float:
    """Minimum backtest length (in YEARS) for target_sr_ann to be beyond what
    the best of n_trials would produce by luck. Uses E[max SR] with var_sr=1
    (the bracket term) since annualized trial-SR std ~ 1/sqrt(years)."""
    if target_sr_ann <= 0:
        return float("inf")
    bracket = expected_max_sharpe(n_trials, 1.0)
    return (bracket / target_sr_ann) ** 2
```

- [ ] **Step 4: Write parity test `tests/test_parity_gauntlet.py`**

```python
"""Parity: qmeta's E[maxSR] and DSR must equal orb-strategy/gauntlet/cpcv.py.
Skips cleanly if orb-strategy (or its deps) is not importable."""
import math
import sys
from pathlib import Path
import pytest

ORB = Path(r"C:\Users\madas\orb-strategy")


def _load_gauntlet():
    if str(ORB) not in sys.path:
        sys.path.insert(0, str(ORB))
    from gauntlet import cpcv  # noqa: needs strategy.paths + sklearn present
    return cpcv


def test_parity_expected_max_sharpe():
    try:
        cpcv = _load_gauntlet()
    except Exception as e:  # ImportError or missing deps
        pytest.skip(f"gauntlet not importable: {e}")
    from qmeta.selection.trials import expected_max_sharpe
    for n, v in [(10, 1.0), (19, 0.5), (100, 0.25)]:
        assert abs(expected_max_sharpe(n, v) - cpcv.expected_max_sharpe(n, v)) < 1e-12


def test_parity_deflated_sharpe():
    try:
        cpcv = _load_gauntlet()
    except Exception as e:
        pytest.skip(f"gauntlet not importable: {e}")
    from qmeta.selection.trials import deflated_sharpe_ratio
    args = (0.06, 2000, 0.1, 3.5, 20, 1.0 / 2000)
    assert abs(deflated_sharpe_ratio(*args) - cpcv.deflated_sharpe(*args)) < 1e-12
```

- [ ] **Step 5: Add re-exports** to `__init__.py` (`expected_max_sharpe`, `deflated_sharpe_ratio`, `min_backtest_length`).
- [ ] **Step 6: Run tests** — `python -m pytest tests/test_trials.py tests/test_parity_gauntlet.py -q` → PASS (parity may SKIP if gauntlet deps absent; that's acceptable).
- [ ] **Step 7: Commit** — `git commit -m "feat(selection): E[maxSR] + DSR + MinBTL (+gauntlet parity)"`

---

### Task 3: `trials.py` — PBO via CSCV

**Files:**
- Modify: `src/qmeta/selection/trials.py` (append), `src/qmeta/selection/__init__.py`
- Test: `tests/test_pbo.py`

**Interfaces:**
- Produces: `probability_of_backtest_overfitting(returns_matrix, n_splits=16) -> dict(pbo, logits, n_partitions)`.

- [ ] **Step 1: Write failing test `tests/test_pbo.py`**

```python
import numpy as np
from qmeta.selection.trials import probability_of_backtest_overfitting as pbo


def test_pbo_no_overfit_when_one_column_dominates():
    # col 0 has a strong positive mean every period; others are flat noise-free
    T = 400
    t = np.arange(T)
    dominant = np.full(T, 0.02)
    others = np.column_stack([0.0001 * np.sin(t / (k + 2)) for k in range(3)])
    R = np.column_stack([dominant, others])
    out = pbo(R, n_splits=8)
    assert out["pbo"] < 0.05


def test_pbo_perfect_overfit_when_ranking_flips():
    # Two columns whose in-sample winner is always the out-of-sample loser.
    # First half: A up, B down. Second half: A down, B up.
    half = 100
    A = np.concatenate([np.full(half, 0.02), np.full(half, -0.02)])
    B = np.concatenate([np.full(half, -0.02), np.full(half, 0.02)])
    R = np.column_stack([A, B])
    out = pbo(R, n_splits=2)
    assert out["pbo"] == 1.0


def test_pbo_rejects_bad_shapes():
    import pytest
    with pytest.raises(ValueError):
        pbo(np.zeros((10, 1)), n_splits=2)   # need >=2 columns
    with pytest.raises(ValueError):
        pbo(np.zeros((10, 3)), n_splits=3)   # n_splits must be even
```

- [ ] **Step 2: Run test** → FAIL.

- [ ] **Step 3: Append to `src/qmeta/selection/trials.py`**

```python
from itertools import combinations
import numpy as np


def _columns_sharpe(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=1)
    return np.where(sd > 0, mu / sd, -np.inf)


def probability_of_backtest_overfitting(returns_matrix, n_splits: int = 16) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV
    (Bailey, Borwein, Lopez de Prado, Zhu 2015).

    returns_matrix: 2D array-like (time x trials), one column per configuration.
    Partitions time into n_splits contiguous blocks; over every half/half
    in-sample/out-of-sample split, picks the IS-best column by Sharpe and finds
    its OOS rank omega in (0,1); PBO = P(logit(omega) <= 0)."""
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2D (time x trials)")
    T, N = R.shape
    if N < 2:
        raise ValueError("need >= 2 trial columns")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    logits = []
    for is_combo in combinations(range(n_splits), half):
        is_set = set(is_combo)
        is_rows = np.concatenate([blocks[b] for b in range(n_splits) if b in is_set])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_set])
        is_sr = _columns_sharpe(R[is_rows])
        oos_sr = _columns_sharpe(R[oos_rows])
        n_star = int(np.argmax(is_sr))
        order = np.argsort(oos_sr, kind="stable")  # ascending; last = best
        rank = int(np.where(order == n_star)[0][0]) + 1  # 1..N
        omega = rank / (N + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))
    logits = np.asarray(logits, dtype=float)
    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "logits": logits.tolist(),
        "n_partitions": int(len(logits)),
    }
```

- [ ] **Step 4: Re-export** `probability_of_backtest_overfitting` in `__init__.py`.
- [ ] **Step 5: Run tests** → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(selection): PBO via CSCV"`

---

### Task 4: `approval.py` — Strategy-Approval / Sharpe-indifference

**Files:**
- Create: `src/qmeta/selection/approval.py`
- Modify: `src/qmeta/selection/__init__.py`
- Test: `tests/test_approval.py`

**Interfaces:**
- Produces: `combined_max_sharpe(sr_app, sr_new, corr)`, `improves_max_sharpe(sr_app, sr_new, corr)`, `max_correlation_for_approval(sr_app, sr_new)`, `indifference_curve(sr_app, level=None, n=101) -> (corr_array, sr_new_array)`.

- [ ] **Step 1: Write failing test `tests/test_approval.py`**

```python
import math
import numpy as np
from qmeta.selection.approval import (
    combined_max_sharpe, improves_max_sharpe,
    max_correlation_for_approval, indifference_curve,
)


def test_combined_max_sharpe_reference():
    # S_a=1, S_n=1, rho=0.5 -> sqrt((1-1+1)/0.75) = 1.15470
    assert abs(combined_max_sharpe(1.0, 1.0, 0.5) - math.sqrt(1 / 0.75)) < 1e-12


def test_improves_condition():
    assert improves_max_sharpe(0.95, 0.76, -0.02) is True    # ORB + dip
    assert improves_max_sharpe(1.0, 0.4, 0.5) is False       # 0.4 < 0.5*1.0
    assert improves_max_sharpe(1.0, 0.6, 0.5) is True


def test_max_correlation_for_approval():
    assert abs(max_correlation_for_approval(0.95, 0.76) - 0.76 / 0.95) < 1e-12
    assert max_correlation_for_approval(0.5, 1.0) == 1.0     # capped


def test_indifference_boundary_is_rho_times_Sa():
    # At level == sr_app the boundary is sr_new = rho * sr_app
    corr, srn = indifference_curve(1.0, level=1.0, n=41)
    ok = ~np.isnan(srn)
    assert np.allclose(srn[ok], corr[ok], atol=1e-9)


def test_orb_dip_combined_beats_orb():
    # ORB approved 0.95, dip 0.76, corr -0.02 -> combined clearly > 0.95
    assert combined_max_sharpe(0.95, 0.76, -0.02) > 0.95
```

- [ ] **Step 2: Run test** → FAIL.

- [ ] **Step 3: Write `src/qmeta/selection/approval.py`**

```python
"""Strategy Approval Theorem / Sharpe-ratio Indifference Curve.
Lopez de Prado (2012), "The Strategy Approval Decision".

For an approved book (Sharpe S_a) and a candidate (Sharpe S_n) with correlation
rho, the max-Sharpe of the optimally-combined book is
    S_c = sqrt((S_a^2 - 2*rho*S_a*S_n + S_n^2) / (1 - rho^2)),
and adding the candidate raises the aggregate Sharpe iff S_n > rho * S_a.
All Sharpes may be annualized or per-obs as long as consistent."""
import math
import numpy as np


def combined_max_sharpe(sr_app: float, sr_new: float, corr: float) -> float:
    den = 1.0 - corr * corr
    if den <= 0:
        return float("inf")
    num = sr_app ** 2 - 2.0 * corr * sr_app * sr_new + sr_new ** 2
    return math.sqrt(max(0.0, num / den))


def improves_max_sharpe(sr_app: float, sr_new: float, corr: float) -> bool:
    return sr_new > corr * sr_app


def max_correlation_for_approval(sr_app: float, sr_new: float) -> float:
    if sr_app <= 0:
        return 1.0
    return min(1.0, sr_new / sr_app)


def indifference_curve(sr_app: float, level: float = None, n: int = 101):
    """Locus of (corr, sr_new) holding the combined max-Sharpe == `level`
    (default level = sr_app, i.e. the approval boundary). Returns
    (corr_array, sr_new_array); sr_new is NaN where no real solution exists."""
    if level is None:
        level = sr_app
    corrs = np.linspace(-0.99, 0.99, n)
    out = np.full(n, np.nan)
    for i, rho in enumerate(corrs):
        # solve sr_new^2 - 2*rho*sr_app*sr_new + (sr_app^2 - level^2*(1-rho^2)) = 0
        b = -2.0 * rho * sr_app
        c = sr_app ** 2 - level ** 2 * (1.0 - rho ** 2)
        disc = b * b - 4.0 * c
        if disc < 0:
            continue
        out[i] = (-b + math.sqrt(disc)) / 2.0  # larger root = candidate SR needed
    return corrs, out
```

- [ ] **Step 4: Re-export** the four functions in `__init__.py`.
- [ ] **Step 5: Run tests** → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(selection): Strategy-Approval / indifference curve"`

---

### Task 5: `drawdown.py` — Triple-Penance MaxDD & Time-under-Water

> **NOTE TO CONTROLLER:** exact closed-form constants are being confirmed against Bailey & López de Prado (2014, SSRN 2201302) by a parallel research task. Finalize the formulas and the reference-value test in this task from that brief BEFORE dispatching the implementer. The signatures, assumptions, and property-based tests below are fixed; only the exact algebraic constants in Steps 1 & 3 get pinned.

**Files:**
- Create: `src/qmeta/selection/drawdown.py`
- Modify: `src/qmeta/selection/__init__.py`
- Test: `tests/test_drawdown.py`

**Interfaces:**
- Produces: `max_drawdown_quantile(mu, sigma, prob=0.95, phi=0.0)`, `max_time_under_water(mu, sigma, prob=0.95, phi=0.0)`, `triple_penance(mu, sigma, prob=0.95, phi=0.0) -> dict(max_dd, time_to_maxdd, max_tuw, penance_ratio)`, `from_returns(r, prob=0.95, ar1=False) -> dict`.

- [ ] **Step 1: (fill from research) Write failing test `tests/test_drawdown.py`** with:
  - the paper's worked numeric example as an exact-value anchor,
  - property tests: MaxDD increases with `sigma`, decreases with `mu`; `penance_ratio` ≈ 3 in the symmetric i.i.d. Gaussian limit; `max_drawdown_quantile` returns a positive fraction; higher `prob` → larger MaxDD.
- [ ] **Step 2: Run test** → FAIL.
- [ ] **Step 3: (fill from research) Write `src/qmeta/selection/drawdown.py`** with the confirmed closed forms (i.i.d. Gaussian + AR(1) via effective drift/variance), `from_returns` estimating `mu, sigma` (and `phi` when `ar1=True`) from a realized series.
- [ ] **Step 4: Re-export** in `__init__.py`.
- [ ] **Step 5: Run tests** → PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(selection): Triple-Penance drawdown & TuW"`

---

### Task 6: `scripts/load_streams.py` — build ORB + dip daily returns

**Files:**
- Create: `scripts/load_streams.py`
- Test: `tests/test_load_streams.py`

**Interfaces:**
- Produces: `load_orb() -> pd.Series`, `load_orb_ticker_matrix() -> pd.DataFrame` (date × ticker daily net R, for PBO), `load_dip(cache=True) -> pd.Series`. All indexed by `DatetimeIndex`, values are daily fractional returns (ORB scaled by 0.0025 per net-R, matching the 15%-vol WFT sizing).

**Data sources (absolute paths):**
- ORB: `C:\Users\madas\orb-strategy\data\raw_har_trades_atr3.parquet` (cols `date,ticker,R,entry,risk,w`). Daily net R = `sum(w*(R-0.02))` grouped by `date`; daily return = net R × `0.0025`.
- Dip: `C:\Users\madas\dip-recovery\data\panel.parquet` + `daily_bars.parquet`, run through the capital-constrained dip portfolio sim (logic below), then `.pct_change()`. Cache the equity series to `C:\Users\madas\qmeta\scratch\dip_equity.parquet` (git-ignored) and reuse if present.

- [ ] **Step 1: Write `scripts/load_streams.py`**

```python
"""Build the two live return streams the scorecard grades: the ORB fund and
the dip diversifier. Reads the strategy repos by absolute path; writes only a
git-ignored cache. No lookahead — realized history only."""
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
```

- [ ] **Step 2: Write `tests/test_load_streams.py`** — an integration test that runs only when the data files exist:

```python
from pathlib import Path
import pytest
import pandas as pd

pytestmark = pytest.mark.skipif(
    not Path(r"C:\Users\madas\orb-strategy\data\raw_har_trades_atr3.parquet").exists(),
    reason="strategy data files not present",
)


def test_orb_stream_shape():
    import sys; sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")
    from load_streams import load_orb
    s = load_orb()
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) > 1000
    assert abs(s.mean()) < 0.05  # sane daily return scale


def test_orb_matrix_columns():
    import sys; sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")
    from load_streams import load_orb_ticker_matrix
    m = load_orb_ticker_matrix()
    assert m.shape[1] >= 2 and m.shape[0] > 500
```

- [ ] **Step 3: Run** — `python -m pytest tests/test_load_streams.py -q` (PASS, or SKIP if data absent).
- [ ] **Step 4: Commit** — `git commit -m "feat(scripts): load ORB + dip return streams"`

---

### Task 7: `scripts/scorecard.py` — run every tool on both streams

**Files:**
- Create: `scripts/scorecard.py`
- Test: `tests/test_scorecard.py` (runs on a tiny synthetic stream, no external data)

**Interfaces:**
- Produces: `grade_stream(returns, name, n_trials, ppy=252) -> dict`; `build_scorecard() -> dict`; writes `C:\Users\madas\qmeta\scratch\scorecard.json`; prints a console table.

- [ ] **Step 1: Write `scripts/scorecard.py`**

```python
"""Grade the ORB fund and the dip diversifier with the full selection toolkit,
emit scorecard.json + a console table. Run: python scripts/scorecard.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, r"C:\Users\madas\qmeta\src")
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")

from qmeta.selection import (
    sharpe_per_obs, moments, annualize_sr,
    probabilistic_sharpe_ratio, min_track_record_length,
    expected_max_sharpe, deflated_sharpe_ratio, min_backtest_length,
    probability_of_backtest_overfitting,
    combined_max_sharpe, improves_max_sharpe, max_correlation_for_approval,
    indifference_curve,
)
from qmeta.selection.drawdown import triple_penance, from_returns as dd_from_returns

OUT = Path(r"C:\Users\madas\qmeta\scratch\scorecard.json")
N_TRIALS = 19  # documented gauntlet trial count; adjustable


def grade_stream(returns: pd.Series, name: str, n_trials: int = N_TRIALS, ppy: int = 252) -> dict:
    r = returns.dropna().values
    sr = sharpe_per_obs(r)
    sk, ku = moments(r)
    n = len(r)
    var_sr = 1.0 / n
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    dd = dd_from_returns(r, prob=0.95)
    return {
        "name": name,
        "n_obs": n,
        "sharpe_ann": annualize_sr(sr, ppy),
        "skew": sk, "kurt": ku,
        "psr": probabilistic_sharpe_ratio(sr, n, sk, ku, 0.0),
        "mintrl_obs": min_track_record_length(sr, sk, ku, 0.0, 0.95),
        "mintrl_years": min_track_record_length(sr, sk, ku, 0.0, 0.95) / ppy,
        "expected_max_sr_ann": annualize_sr(expected_max_sharpe(n_trials, var_sr), ppy),
        "dsr": deflated_sharpe_ratio(sr, n, sk, ku, n_trials, var_sr),
        "min_backtest_years": min_backtest_length(n_trials, annualize_sr(sr, ppy)),
        "track_years": n / ppy,
        "drawdown": dd,
    }


def build_scorecard() -> dict:
    from load_streams import load_orb, load_orb_ticker_matrix, load_dip
    orb, dip = load_orb(), load_dip()
    cards = {"orb": grade_stream(orb, "ORB fund"), "dip": grade_stream(dip, "Dip diversifier")}
    # PBO on ORB per-ticker columns (does the best in-sample instrument survive OOS?)
    mat = load_orb_ticker_matrix()
    cards["orb"]["pbo"] = probability_of_backtest_overfitting(mat.values, n_splits=10)["pbo"]
    cards["dip"]["pbo"] = None  # single aggregated stream; PBO N/A
    # Strategy approval: ORB approved, dip candidate (annualized Sharpes)
    sa, sn = cards["orb"]["sharpe_ann"], cards["dip"]["sharpe_ann"]
    j = orb.align(dip, join="inner")
    corr = float(pd.concat([j[0], j[1]], axis=1).corr().iloc[0, 1])
    xs, ys = indifference_curve(sa, level=sa, n=81)
    approval = {
        "sr_approved": sa, "sr_candidate": sn, "correlation": corr,
        "improves": improves_max_sharpe(sa, sn, corr),
        "combined_max_sharpe": combined_max_sharpe(sa, sn, corr),
        "max_corr_for_approval": max_correlation_for_approval(sa, sn),
        "indifference_curve": {"corr": xs.tolist(), "sr_new": [None if np.isnan(v) else v for v in ys]},
    }
    scorecard = {"streams": cards, "approval": approval, "n_trials": N_TRIALS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scorecard, indent=2, default=float))
    return scorecard


def _print(sc: dict) -> None:
    for k, c in sc["streams"].items():
        print(f"\n=== {c['name']} ({c['track_years']:.1f}y, {c['n_obs']} days) ===")
        print(f"  Sharpe(ann)      {c['sharpe_ann']:.2f}")
        print(f"  PSR (>0)         {c['psr']*100:.1f}%")
        print(f"  DSR (deflated)   {c['dsr']*100:.1f}%   [E[maxSR]={c['expected_max_sr_ann']:.2f}, K={sc['n_trials']}]")
        print(f"  MinTRL           {c['mintrl_years']:.1f}y  (have {c['track_years']:.1f}y)")
        print(f"  MinBTL           {c['min_backtest_years']:.1f}y")
        pbo_s = "n/a" if c["pbo"] is None else f"{c['pbo']*100:.0f}%"
        print(f"  PBO              {pbo_s}")
    a = sc["approval"]
    print(f"\n=== Strategy Approval: add dip to ORB? ===")
    print(f"  ORB SR {a['sr_approved']:.2f} | dip SR {a['sr_candidate']:.2f} | corr {a['correlation']:+.2f}")
    print(f"  improves aggregate Sharpe: {a['improves']}  ->  combined {a['combined_max_sharpe']:.2f}")
    print(f"  (would help up to correlation {a['max_corr_for_approval']:.2f})")


if __name__ == "__main__":
    _print(build_scorecard())
```

- [ ] **Step 2: Write `tests/test_scorecard.py`** — test `grade_stream` on a deterministic synthetic series (no external data):

```python
import numpy as np, pandas as pd, sys
sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")


def test_grade_stream_keys_and_ranges():
    from scorecard import grade_stream
    idx = pd.bdate_range("2018-01-01", periods=1500)
    # deterministic mild-positive-drift series
    r = pd.Series(0.0004 + 0.01 * np.sin(np.arange(1500) / 7.0), index=idx)
    c = grade_stream(r, "synthetic", n_trials=19)
    assert 0.0 <= c["psr"] <= 1.0
    assert 0.0 <= c["dsr"] <= 1.0
    assert c["mintrl_years"] > 0
    assert set(["sharpe_ann", "drawdown", "min_backtest_years"]).issubset(c)
```

- [ ] **Step 3: Run** — `python -m pytest tests/test_scorecard.py -q` → PASS. Then run the real thing: `python scripts/scorecard.py` and capture the console output into the task report.
- [ ] **Step 4: Commit** — `git commit -m "feat(scripts): scorecard grading ORB + dip"`

---

### Task 8: `scripts/make_dashboard.py` — self-contained visual scorecard

**Files:**
- Create: `scripts/make_dashboard.py`
- Output: `C:\Users\madas\qmeta\scratch\qmeta_scorecard.html`

**Interfaces:**
- Consumes: `scratch/scorecard.json`.
- Produces: a single self-contained HTML file (inline CSS/SVG/JS, no external assets), theme-aware (light/dark), with hover tooltips.

Follow the **dataviz** skill (validated palette, categorical hues in fixed order, thin marks, legend for ≥2 series, table fallback) and the **artifact-design** fundamentals (both themes via CSS custom properties, tabular-nums for figures). Layout:
1. **Header:** title + one-line verdict ("ORB: real edge, track record sufficient" etc., derived from PSR/DSR/MinTRL vs track_years).
2. **Two strategy cards** (ORB, dip): Sharpe(ann), PSR, DSR with E[maxSR] context, MinTRL-in-years vs track-years (a small bar showing have-vs-need), MinBTL, PBO. Green/amber/red status pills from thresholds (PSR≥0.95 good; DSR≥0.95 good; MinTRL≤track good; PBO≤0.2 good).
3. **Strategy-Approval panel:** plot the Sharpe-indifference curve (corr on x, required candidate SR on y) with the ORB+dip point marked; annotate "improves: yes/no, combined SR = …".
4. **Drawdown panel:** projected Triple-Penance MaxDD & expected recovery time vs the empirical −18% (ORB) / −40% (dip) from the WFTs.

- [ ] **Step 1:** Load the `artifact-design` skill, then write `make_dashboard.py` that emits the HTML per the layout above, reading `scorecard.json`.
- [ ] **Step 2:** Run `python scripts/make_dashboard.py`; open the HTML and eyeball for overflow/label collisions.
- [ ] **Step 3:** Publish as a claude.ai Artifact (favicon 📊).
- [ ] **Step 4: Commit** — `git commit -m "feat(scripts): visual scorecard dashboard"`

---

## Final whole-branch review

After Task 8, dispatch the whole-branch code review (most-capable model). Then run
`superpowers:finishing-a-development-branch` to merge/keep.
