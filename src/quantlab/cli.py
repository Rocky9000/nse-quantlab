"""quantlab backtest | charges"""

from __future__ import annotations

import argparse
import sys

from .costs import IndianEquityCosts
from .data import fetch_yahoo, load_csv
from .engine import BacktestConfig, Backtester
from .metrics import bootstrap_sharpe_ci, daily_returns, probabilistic_sharpe
from .strategies import BuyAndHold, SMACrossover, ZScoreMeanReversion


def _strategy(args):
    if args.strategy == "sma":
        return SMACrossover(args.fast, args.slow, allow_short=args.product == "intraday")
    if args.strategy == "meanrev":
        return ZScoreMeanReversion(args.window, args.entry, allow_short=args.product == "intraday")
    return BuyAndHold()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quantlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="backtest a strategy on a CSV (date,open,high,low,close,volume) or a symbol")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--symbol", help="Yahoo symbol, e.g. RELIANCE.NS (needs yfinance)")
    b.add_argument("--start", default="2015-01-01")
    b.add_argument("--strategy", choices=["sma", "meanrev", "buyhold"], default="sma")
    b.add_argument("--product", choices=["intraday", "delivery"], default="delivery")
    b.add_argument("--fast", type=int, default=20)
    b.add_argument("--slow", type=int, default=50)
    b.add_argument("--window", type=int, default=20)
    b.add_argument("--entry", type=float, default=2.0)
    b.add_argument("--capital", type=float, default=1_000_000)
    b.add_argument("--slippage-bps", type=float, default=2.0)

    c = sub.add_parser("charges", help="itemised charges for a round trip")
    c.add_argument("value", type=float)
    c.add_argument("--product", choices=["intraday", "delivery"], default="intraday")

    a = p.parse_args(argv)

    if a.cmd == "charges":
        costs = IndianEquityCosts(a.product)
        for side in ("buy", "sell"):
            ch = costs.charges(a.value, side)
            items = ", ".join(f"{k} ₹{v:,.2f}" for k, v in vars(ch).items())
            print(f"{side:>4}: {items}  | total ₹{ch.total:,.2f}")
        print(f"round trip: {costs.round_trip_bps(a.value):.2f} bps of traded value")
        return 0

    bars = load_csv(a.csv) if a.csv else fetch_yahoo(a.symbol, a.start)
    cfg = BacktestConfig(initial_capital=a.capital, product=a.product, slippage_bps=a.slippage_bps,
                         daily_loss_limit=0.02 if a.product == "intraday" else None)
    res = Backtester(cfg).run(bars, _strategy(a))
    r = daily_returns(res.equity)
    lo, hi = bootstrap_sharpe_ci(r)
    for k, v in res.summary().items():
        print(f"{k:>24}: {v:,.4f}" if isinstance(v, float) else f"{k:>24}: {v}")
    print(f"{'sharpe_95%_ci':>24}: [{lo:.2f}, {hi:.2f}]")
    print(f"{'P(sharpe > 0)':>24}: {probabilistic_sharpe(r):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
