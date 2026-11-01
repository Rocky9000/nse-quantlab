"""Event-driven, bar-by-bar backtester with no look-ahead.

Timing model (the part most backtests get wrong):

* At the **close** of bar *t* the strategy sees bars ``0..t`` only and
  returns a target exposure.
* That order fills at the **open** of bar *t+1*, plus slippage, plus
  itemised Indian transaction costs.
* Risk events (daily loss limit, drawdown halt) and intraday square-off
  flatten the position at the close of the bar that triggers them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .costs import IndianEquityCosts
from .data import validate
from .metrics import performance


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    product: str = "intraday"  # "intraday" (MIS, may short, squared off daily) or "delivery" (CNC, long only)
    slippage_bps: float = 2.0
    max_leverage: float = 1.0  # max gross notional / equity
    daily_loss_limit: float | None = 0.02  # stop trading for the day after losing this fraction
    max_drawdown_halt: float | None = 0.25  # stop trading for good after this drawdown from peak
    square_off_eod: bool | None = None  # defaults to True for intraday


class Context:
    """Read-only view handed to the strategy at the close of bar ``t``."""

    def __init__(self, bars: pd.DataFrame, t: int, position: int, equity: float, cash: float):
        self._bars, self.t = bars, t
        self.position, self.equity, self.cash = position, equity, cash

    @property
    def history(self) -> pd.DataFrame:
        return self._bars.iloc[: self.t + 1]  # bars 0..t inclusive; never t+1

    @property
    def bar(self) -> pd.Series:
        return self._bars.iloc[self.t]

    @property
    def timestamp(self) -> pd.Timestamp:
        return self._bars.index[self.t]


class Strategy(Protocol):
    def target(self, ctx: Context) -> float:
        """Desired exposure as a fraction of equity in [-1, 1] (negative = short)."""
        ...


@dataclass
class BacktestResult:
    equity: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    total_charges: float
    config: BacktestConfig
    halted_at: pd.Timestamp | None = None
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        n_trades = len(self.trades)
        traded = float(self.trades["value"].sum()) if n_trades else 0.0
        return {
            **self.metrics,
            "trades": n_trades,
            "charges": self.total_charges,
            "charges_pct_of_capital": self.total_charges / self.config.initial_capital,
            "turnover_x": traded / self.config.initial_capital,
        }


class Backtester:
    def __init__(self, config: BacktestConfig | None = None, costs: IndianEquityCosts | None = None):
        self.config = config or BacktestConfig()
        self.costs = costs or IndianEquityCosts(self.config.product)
        if self.costs.product != self.config.product:
            raise ValueError("costs.product must match config.product")

    def run(self, bars: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        cfg = self.config
        bars = validate(bars)
        n = len(bars)
        opens, closes = bars["open"].to_numpy(), bars["close"].to_numpy()
        days = bars.index.normalize()
        last_of_day = np.r_[days[1:] != days[:-1], True]
        square_off = cfg.square_off_eod if cfg.square_off_eod is not None else cfg.product == "intraday"
        slip = cfg.slippage_bps / 1e4

        cash, pos = cfg.initial_capital, 0
        pending: int | None = None
        peak = day_start_equity = cfg.initial_capital
        halted_today = halted_forever = False
        halted_at = None
        charges_total = 0.0
        equity_curve, positions, trades = np.empty(n), np.empty(n, dtype=np.int64), []

        def fill(t: int, qty: int, ref_price: float, reason: str) -> None:
            nonlocal cash, pos, charges_total
            if qty == 0:
                return
            side = "buy" if qty > 0 else "sell"
            price = ref_price * (1 + slip if qty > 0 else 1 - slip)
            if qty > 0 and pos >= 0 and cfg.max_leverage <= 1.0:
                # Sizing used the previous close; if the open gapped up, buy only what cash covers.
                qty = min(qty, max(0, math.floor(cash / (price * 1.002))))
                if qty == 0:
                    return
            value = abs(qty) * price
            fee = self.costs.charges(value, side).total
            cash -= qty * price + fee
            pos += qty
            charges_total += fee
            trades.append({"time": bars.index[t], "side": side, "qty": abs(qty), "price": price,
                           "value": value, "charges": fee, "reason": reason})

        for t in range(n):
            if t > 0 and days[t] != days[t - 1]:
                day_start_equity = cash + pos * closes[t - 1]
                halted_today = False

            if pending is not None and not (halted_today or halted_forever):
                fill(t, pending - pos, opens[t], "signal")
            pending = None

            equity = cash + pos * closes[t]
            peak = max(peak, equity)

            if cfg.max_drawdown_halt is not None and not halted_forever and equity < peak * (1 - cfg.max_drawdown_halt):
                fill(t, -pos, closes[t], "drawdown_halt")
                halted_forever, halted_at = True, bars.index[t]
            elif cfg.daily_loss_limit is not None and not halted_today and \
                    equity < day_start_equity * (1 - cfg.daily_loss_limit):
                fill(t, -pos, closes[t], "daily_loss_limit")
                halted_today = True

            if not (halted_today or halted_forever):
                ctx = Context(bars, t, pos, cash + pos * closes[t], cash)
                target = float(np.clip(strategy.target(ctx), -1.0, 1.0))
                if cfg.product == "delivery":
                    target = max(target, 0.0)
                budget = target * cfg.max_leverage * ctx.equity
                desired = int(math.copysign(math.floor(abs(budget) / closes[t]), budget))
                # Intraday positions cannot be carried, so an order placed at the
                # last close executes at the next session's open as a fresh position.
                if desired != (0 if (square_off and last_of_day[t]) else pos):
                    pending = desired

            if square_off and last_of_day[t] and pos != 0:
                fill(t, -pos, closes[t], "eod_square_off")

            equity_curve[t] = cash + pos * closes[t]
            positions[t] = pos

        equity = pd.Series(equity_curve, index=bars.index, name="equity")
        result = BacktestResult(
            equity=equity,
            position=pd.Series(positions, index=bars.index, name="position"),
            trades=pd.DataFrame(trades, columns=["time", "side", "qty", "price", "value", "charges", "reason"]),
            total_charges=charges_total,
            config=cfg,
            halted_at=halted_at,
        )
        result.metrics = performance(equity)
        return result
