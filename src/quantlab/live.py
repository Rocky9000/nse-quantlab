"""Paper trading on live NSE quotes.

The same ``Strategy`` objects used in backtests run here against live
prices, with orders filled by ``PaperBroker`` (simulated fills, real Indian
charges, slippage). No real orders are ever sent.

Going live is deliberately left out. Before sending real orders through a
broker API, read SEBI's framework for retail algorithmic trading and your
broker's API terms, and run in paper mode long enough to trust the system.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

from .costs import IndianEquityCosts
from .engine import Context, Strategy

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN, MARKET_CLOSE = dtime(9, 15), dtime(15, 30)
SQUARE_OFF = dtime(15, 15)  # flatten before the broker's MIS auto square-off


@dataclass
class Fill:
    time: datetime
    symbol: str
    side: str
    qty: int
    price: float
    charges: float
    reason: str


class OrderRejected(Exception):
    pass


@dataclass
class PaperBroker:
    cash: float
    costs: IndianEquityCosts = field(default_factory=IndianEquityCosts)
    slippage_bps: float = 2.0
    max_leverage: float = 1.0
    positions: dict[str, int] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    last_prices: dict[str, float] = field(default_factory=dict)

    def equity(self, prices: dict[str, float] | None = None) -> float:
        if prices:
            self.last_prices.update(prices)
        return self.cash + sum(q * self.last_prices[s] for s, q in self.positions.items() if q)

    def place_market_order(self, symbol: str, qty: int, price: float, when: datetime, reason: str = "signal") -> Fill:
        if qty == 0:
            raise OrderRejected("quantity must be non-zero")
        if price <= 0:
            raise OrderRejected("invalid price")
        held = self.positions.get(symbol, 0)
        if self.costs.product == "delivery" and held + qty < 0:
            raise OrderRejected("cannot short in delivery (CNC)")
        self.last_prices[symbol] = price
        new_pos = held + qty
        gross = abs(new_pos) * price + sum(
            abs(q) * self.last_prices[s] for s, q in self.positions.items() if s != symbol
        )
        if abs(new_pos) > abs(held) and gross > self.max_leverage * self.equity() * 1.001:
            raise OrderRejected("order exceeds leverage limit")
        side = "buy" if qty > 0 else "sell"
        fill_price = price * (1 + self.slippage_bps / 1e4 if qty > 0 else 1 - self.slippage_bps / 1e4)
        value = abs(qty) * fill_price
        charges = self.costs.charges(value, side).total
        self.cash -= qty * fill_price + charges
        self.positions[symbol] = held + qty
        f = Fill(when, symbol, side, abs(qty), fill_price, charges, reason)
        self.fills.append(f)
        return f


class CandleBuilder:
    """Aggregates polled last-traded prices into fixed-interval OHLC candles."""

    def __init__(self, minutes: int = 1):
        self.step = timedelta(minutes=minutes)
        self.rows: list[dict] = []
        self._current: dict | None = None

    def _bucket(self, ts: datetime) -> datetime:
        epoch = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        return epoch + ((ts - epoch) // self.step) * self.step

    def add(self, ts: datetime, price: float, volume: float = 0.0) -> bool:
        """Add a tick. Returns True when it closed the previous candle."""
        b = self._bucket(ts)
        if self._current and self._current["time"] == b:
            c = self._current
            c["high"], c["low"], c["close"] = max(c["high"], price), min(c["low"], price), price
            c["volume"] += volume
            return False
        closed = self._current is not None
        if closed:
            self.rows.append(self._current)
        self._current = {"time": b, "open": price, "high": price, "low": price, "close": price, "volume": volume}
        return closed

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(self.rows).set_index("time")


class PaperTrader:
    """Runs a strategy on completed candles and trades through ``PaperBroker``.

    ``step(ts, price)`` is the whole state machine, so it can be driven by a
    live quote loop (``run``) or replayed in tests.
    """

    def __init__(self, symbol: str, strategy: Strategy, broker: PaperBroker, candle_minutes: int = 5,
                 daily_loss_limit: float = 0.02):
        self.symbol, self.strategy, self.broker = symbol, strategy, broker
        self.candles = CandleBuilder(candle_minutes)
        self.daily_loss_limit = daily_loss_limit
        self._day = None
        self._day_start_equity = broker.cash
        self.halted = False
        self.log: list[str] = []

    def _flatten(self, price: float, ts: datetime, reason: str) -> None:
        held = self.broker.positions.get(self.symbol, 0)
        if held:
            self.broker.place_market_order(self.symbol, -held, price, ts, reason)

    def step(self, ts: datetime, price: float) -> None:
        if ts.date() != self._day:
            self._day, self.halted = ts.date(), False
            self._day_start_equity = self.broker.equity({self.symbol: price})
        if self.halted:
            return
        equity = self.broker.equity({self.symbol: price})
        if equity < self._day_start_equity * (1 - self.daily_loss_limit):
            self._flatten(price, ts, "daily_loss_limit")
            self.halted = True
            self.log.append(f"{ts}: daily loss limit hit, flat until tomorrow")
            return
        if ts.time() >= SQUARE_OFF:
            self._flatten(price, ts, "eod_square_off")
            return
        if not self.candles.add(ts, price):
            return
        bars = self.candles.frame()
        held = self.broker.positions.get(self.symbol, 0)
        ctx = Context(bars, len(bars) - 1, held, equity, self.broker.cash)
        target = max(-1.0, min(1.0, float(self.strategy.target(ctx))))
        desired = int(math.copysign(math.floor(abs(target) * self.broker.max_leverage * equity / price), target))
        if desired != held:
            try:
                self.broker.place_market_order(self.symbol, desired - held, price, ts)
            except OrderRejected as e:
                self.log.append(f"{ts}: rejected ({e})")

    def run(self, quote: Callable[[], float], poll_seconds: float = 2.0) -> None:  # pragma: no cover - live loop
        while True:
            now = datetime.now(IST)
            if MARKET_OPEN <= now.time() <= MARKET_CLOSE and now.weekday() < 5:
                self.step(now, quote())
            time.sleep(poll_seconds)


def kite_ltp_quote(api_key: str, access_token: str, instrument: str) -> Callable[[], float]:  # pragma: no cover
    """Read-only last-traded-price feed from Zerodha Kite Connect, e.g. ``NSE:RELIANCE``."""
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return lambda: float(kite.ltp([instrument])[instrument]["last_price"])
