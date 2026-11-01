import numpy as np
import pandas as pd
import pytest

from quantlab import BacktestConfig, Backtester, BuyAndHold, IndianEquityCosts, SMACrossover, ZScoreMeanReversion
from quantlab.costs import ProductRates
from quantlab.data import synthetic_bars, validate

FREE = {"exchange_pct": 0, "sebi_per_crore": 0, "gst_pct": 0}


def free_costs(product):
    rates = {product: ProductRates(0, None, 0, 0, 0, 0)}
    return IndianEquityCosts(product, rates=rates, **FREE)


def bars_from(opens, closes, start="2024-01-01 15:30", freq="B"):
    idx = pd.date_range(start, periods=len(opens), freq=freq)
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                         "volume": 1000.0}, index=idx)


class Constant:
    def __init__(self, value):
        self.value = value

    def target(self, ctx):
        return self.value


class Spy:
    def __init__(self):
        self.seen = []

    def target(self, ctx):
        self.seen.append((ctx.t, len(ctx.history), ctx.history.index[-1] == ctx.timestamp))
        return 0.0


def cfg(**kw):
    base = {"product": "delivery", "slippage_bps": 0.0, "daily_loss_limit": None, "max_drawdown_halt": None,
            "initial_capital": 10_000.0}
    base.update(kw)
    return BacktestConfig(**base)


def test_strategy_never_sees_the_future():
    spy = Spy()
    Backtester(cfg()).run(synthetic_bars(days=30), spy)
    assert all(n == t + 1 and last for t, n, last in spy.seen)


def test_orders_fill_at_next_open():
    bars = bars_from([100, 100.5, 104, 106], [101, 103, 105, 107])
    res = Backtester(cfg(), free_costs("delivery")).run(bars, BuyAndHold())
    first = res.trades.iloc[0]
    assert first["time"] == bars.index[1] and first["price"] == 100.5
    assert first["qty"] == 10_000 // 101  # sized on bar-0 close
    assert res.equity.iloc[0] == 10_000  # nothing held during the signal bar


def test_pnl_reconciles_with_charges():
    bars = synthetic_bars(days=120, seed=5)
    res = Backtester(cfg(slippage_bps=5)).run(bars, SMACrossover(5, 20))
    t = res.trades
    signed = np.where(t["side"] == "buy", -1, 1) * t["value"]
    final_pos = res.position.iloc[-1]
    expected = 10_000 + signed.sum() - t["charges"].sum() + final_pos * bars["close"].iloc[-1]
    assert res.equity.iloc[-1] == pytest.approx(expected)
    assert res.total_charges == pytest.approx(t["charges"].sum())


def test_delivery_cannot_short():
    res = Backtester(cfg()).run(synthetic_bars(days=20), Constant(-1.0))
    assert (res.position == 0).all() and res.trades.empty


def test_gap_up_open_never_overspends_cash():
    bars = bars_from([100, 150, 150], [100, 150, 150])
    res = Backtester(cfg()).run(bars, BuyAndHold())
    cash = res.equity.iloc[-1] - res.position.iloc[-1] * 150
    assert cash >= 0


def test_intraday_is_flat_at_every_close():
    bars = synthetic_bars(days=10, bars_per_day=25, seed=2)
    res = Backtester(cfg(product="intraday", initial_capital=1e6)).run(bars, ZScoreMeanReversion(10, 1.0))
    eod = bars.index.normalize()
    last_bar = np.r_[eod[1:] != eod[:-1], True]
    assert (res.position[last_bar] == 0).all()
    assert (res.trades["reason"] == "eod_square_off").any()


def test_daily_loss_limit_stops_trading_for_the_day():
    idx = pd.date_range("2024-01-01 09:15", periods=6, freq="5min").append(
        pd.date_range("2024-01-02 09:15", periods=3, freq="5min"))
    o = np.array([100, 100, 100, 90, 90, 90, 90, 91, 92], float)
    bars = pd.DataFrame({"open": o, "high": o + 1, "low": o - 1, "close": o, "volume": 1.0}, index=idx)
    res = Backtester(cfg(product="intraday", daily_loss_limit=0.02, initial_capital=1e6),
                     free_costs("intraday")).run(bars, Constant(1.0))
    reasons = res.trades["reason"].tolist()
    assert "daily_loss_limit" in reasons
    halted_at = res.trades.loc[res.trades.reason == "daily_loss_limit", "time"].iloc[0]
    same_day_after = res.trades[(res.trades.time > halted_at) & (res.trades.time.dt.date == halted_at.date())]
    assert same_day_after.empty
    day2 = res.trades[res.trades.time.dt.date == pd.Timestamp("2024-01-02").date()]
    assert (day2["reason"] == "signal").any()  # trading resumes the next session


def test_drawdown_halt_is_permanent():
    closes = [100, 100, 70, 70, 80, 90]
    res = Backtester(cfg(max_drawdown_halt=0.2), free_costs("delivery")).run(bars_from(closes, closes), BuyAndHold())
    assert res.halted_at is not None
    assert (res.position.loc[res.halted_at:] == 0).all()


def test_validate_rejects_bad_data():
    good = synthetic_bars(days=5)
    with pytest.raises(ValueError):
        validate(good.iloc[::-1])
    bad = good.copy()
    bad.iloc[2, bad.columns.get_loc("high")] = bad["low"].iloc[2] * 0.5
    with pytest.raises(ValueError):
        validate(bad)
    with pytest.raises(ValueError):
        validate(good.drop(columns="volume"))


def test_costs_product_must_match():
    with pytest.raises(ValueError):
        Backtester(cfg(product="delivery"), IndianEquityCosts("intraday"))
