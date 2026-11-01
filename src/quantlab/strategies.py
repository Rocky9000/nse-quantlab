"""Reference strategies. Each sees only ``ctx.history`` (bars up to the current close)."""

from __future__ import annotations

import pandas as pd

from .engine import Context


class BuyAndHold:
    def target(self, ctx: Context) -> float:
        return 1.0


class SMACrossover:
    """Long when the fast moving average is above the slow one (short too, if allowed)."""

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow, self.allow_short = fast, slow, allow_short

    def target(self, ctx: Context) -> float:
        close = ctx.history["close"]
        if len(close) < self.slow:
            return 0.0
        fast, slow = close.iloc[-self.fast :].mean(), close.iloc[-self.slow :].mean()
        if fast > slow:
            return 1.0
        return -1.0 if self.allow_short else 0.0


class ZScoreMeanReversion:
    """Fade large deviations from a rolling mean; exit when price reverts."""

    def __init__(self, window: int = 20, entry: float = 2.0, exit: float = 0.5, allow_short: bool = True):
        self.window, self.entry, self.exit, self.allow_short = window, entry, exit, allow_short

    def target(self, ctx: Context) -> float:
        close = ctx.history["close"]
        if len(close) < self.window:
            return 0.0
        recent = close.iloc[-self.window :]
        sd = recent.std(ddof=1)
        if sd == 0:
            return 0.0
        z = (close.iloc[-1] - recent.mean()) / sd
        held = 1.0 if ctx.position > 0 else (-1.0 if ctx.position < 0 else 0.0)
        if z <= -self.entry:
            return 1.0
        if z >= self.entry and self.allow_short:
            return -1.0
        if abs(z) <= self.exit:
            return 0.0
        return held


class SignalStrategy:
    """Trade a precomputed, time-indexed signal (e.g. out-of-sample model probabilities).

    ``signal`` must be indexed by bar timestamp and computed only from data
    available at that bar's close, which ``validation.walk_forward_predict``
    guarantees.
    """

    def __init__(self, probabilities: pd.Series, band: float = 0.05, allow_short: bool = True):
        self.p = probabilities
        self.band, self.allow_short = band, allow_short

    def target(self, ctx: Context) -> float:
        p = self.p.get(ctx.timestamp)
        if p is None or pd.isna(p):
            return 0.0
        if p > 0.5 + self.band:
            return 1.0
        if p < 0.5 - self.band and self.allow_short:
            return -1.0
        return 0.0
