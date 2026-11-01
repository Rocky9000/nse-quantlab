"""Indian equity transaction costs, itemised the way a broker contract note shows them.

Defaults match Zerodha's published NSE equity charges (checked Nov 2026,
https://zerodha.com/charges). Exchanges and the government revise these, so
every rate is a field you can override.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CRORE = 1e7


@dataclass(frozen=True)
class ChargeBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    dp: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst + self.dp


@dataclass(frozen=True)
class ProductRates:
    brokerage_pct: float
    brokerage_cap: float | None  # rupees per executed order; None = no cap
    stt_buy_pct: float
    stt_sell_pct: float
    stamp_buy_pct: float
    dp_per_sell: float


INTRADAY = ProductRates(brokerage_pct=0.03, brokerage_cap=20.0, stt_buy_pct=0.0, stt_sell_pct=0.025,
                        stamp_buy_pct=0.003, dp_per_sell=0.0)
DELIVERY = ProductRates(brokerage_pct=0.0, brokerage_cap=None, stt_buy_pct=0.1, stt_sell_pct=0.1,
                        stamp_buy_pct=0.015, dp_per_sell=15.34)


@dataclass(frozen=True)
class IndianEquityCosts:
    product: str = "intraday"  # "intraday" (MIS) or "delivery" (CNC)
    exchange_pct: float = 0.00307  # NSE transaction charge
    sebi_per_crore: float = 10.0
    gst_pct: float = 18.0
    rates: dict = field(default_factory=lambda: {"intraday": INTRADAY, "delivery": DELIVERY})

    def __post_init__(self):
        if self.product not in self.rates:
            raise ValueError(f"product must be one of {sorted(self.rates)}")

    def charges(self, value: float, side: str) -> ChargeBreakdown:
        """Charges for one executed order of ``value`` rupees on ``side`` ('buy' or 'sell')."""
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if value <= 0:
            return ChargeBreakdown(0, 0, 0, 0, 0, 0, 0)
        r: ProductRates = self.rates[self.product]
        brokerage = value * r.brokerage_pct / 100
        if r.brokerage_cap is not None:
            brokerage = min(brokerage, r.brokerage_cap)
        stt = value * (r.stt_buy_pct if side == "buy" else r.stt_sell_pct) / 100
        exchange = value * self.exchange_pct / 100
        sebi = value * self.sebi_per_crore / CRORE
        stamp = value * r.stamp_buy_pct / 100 if side == "buy" else 0.0
        gst = (brokerage + exchange + sebi) * self.gst_pct / 100
        dp = r.dp_per_sell if side == "sell" else 0.0
        return ChargeBreakdown(brokerage, stt, exchange, sebi, stamp, gst, dp)

    def round_trip_bps(self, value: float) -> float:
        """Total cost of buying and selling ``value`` rupees, in basis points of ``value``."""
        total = self.charges(value, "buy").total + self.charges(value, "sell").total
        return total / value * 1e4
