"""Triple-Barrier labeling and meta-labeling (Lopez de Prado, AFML ch. 3).

The triple barrier labels an event by whichever of three barriers is touched
first: an upper (profit-take), a lower (stop-loss), or a vertical (time) barrier.
Meta-labeling then reduces the outcome to a binary "did the primary bet make
money", which a secondary model learns to predict for bet sizing.
"""
import numpy as np


def triple_barrier(prices, entry_idx, pt, sl, max_h, side=1):
    """First-touch outcome for a position opened at `entry_idx`.

    prices : 1D array of prices.
    pt, sl : profit-take / stop-loss thresholds as POSITIVE fractional returns.
    max_h  : maximum holding horizon in bars (the vertical barrier).
    side   : +1 long, -1 short.
    Returns (outcome, exit_idx, ret) with outcome in {'pt','sl','time'} and
    `ret` the signed return realized on the position (side-adjusted)."""
    prices = np.asarray(prices, dtype=float)
    p0 = prices[entry_idx]
    end = min(entry_idx + max_h, len(prices) - 1)
    for j in range(entry_idx + 1, end + 1):
        r = side * (prices[j] / p0 - 1.0)
        if r >= pt:
            return ("pt", j, r)
        if r <= -sl:
            return ("sl", j, r)
    return ("time", end, side * (prices[end] / p0 - 1.0))


def meta_label(realized_return, side=1):
    """Binary meta-label: 1 if the primary model's bet made money, else 0."""
    return int(side * realized_return > 0)


def meta_labels_from_R(R):
    """Vectorized meta-label from realized R-multiples: 1 where R > 0."""
    return (np.asarray(R, dtype=float) > 0).astype(int)
