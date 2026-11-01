import numpy as np
import pandas as pd
import pytest

from quantlab.costs import IndianEquityCosts
from quantlab.metrics import (
    bootstrap_sharpe_ci,
    daily_returns,
    deflated_sharpe,
    max_drawdown,
    performance,
    probabilistic_sharpe,
    sharpe,
)


def test_intraday_charges_match_contract_note():
    c = IndianEquityCosts("intraday")
    buy, sell = c.charges(100_000, "buy"), c.charges(100_000, "sell")
    assert buy.brokerage == pytest.approx(20.0)  # 0.03% = 30, capped at 20
    assert buy.stt == 0 and sell.stt == pytest.approx(25.0)
    assert buy.stamp == pytest.approx(3.0) and sell.stamp == 0
    assert buy.exchange == pytest.approx(3.07)
    assert buy.gst == pytest.approx(0.18 * (20 + 3.07 + 0.1))
    assert buy.total + sell.total == pytest.approx(82.6812, abs=1e-3)


def test_small_intraday_order_uses_percentage_brokerage():
    assert IndianEquityCosts("intraday").charges(10_000, "buy").brokerage == pytest.approx(3.0)


def test_delivery_charges():
    c = IndianEquityCosts("delivery")
    buy, sell = c.charges(100_000, "buy"), c.charges(100_000, "sell")
    assert buy.brokerage == 0 and buy.stt == pytest.approx(100.0) and buy.stamp == pytest.approx(15.0)
    assert sell.dp == pytest.approx(15.34)
    assert buy.total == pytest.approx(118.7406, abs=1e-3) and sell.total == pytest.approx(119.0806, abs=1e-3)


def test_round_trip_bps_and_validation():
    assert IndianEquityCosts("intraday").round_trip_bps(100_000) == pytest.approx(8.268, abs=1e-3)
    with pytest.raises(ValueError):
        IndianEquityCosts("futures")
    with pytest.raises(ValueError):
        IndianEquityCosts().charges(100, "hold")
    assert IndianEquityCosts().charges(0, "buy").total == 0


def test_max_drawdown():
    assert max_drawdown(pd.Series([100, 120, 90, 130, 117])) == pytest.approx(-0.25)


def test_sharpe_edge_cases():
    assert sharpe(np.full(10, 0.01)) == 0.0  # zero variance
    assert sharpe(np.array([0.01, -0.01] * 50)) == pytest.approx(0.0)


def test_daily_returns_collapse_intraday_bars():
    idx = pd.to_datetime(["2024-01-01 09:15", "2024-01-01 15:25", "2024-01-02 09:15", "2024-01-02 15:25"])
    r = daily_returns(pd.Series([100, 101, 105, 110], index=idx))
    assert r.tolist() == pytest.approx([110 / 101 - 1])


def test_performance_on_steady_growth():
    idx = pd.bdate_range("2024-01-01", periods=253)
    eq = pd.Series(100 * 1.0004 ** np.arange(253), index=idx)
    p = performance(eq)
    assert p["max_drawdown"] == 0 and p["days"] == 252
    assert p["cagr"] == pytest.approx(1.0004 ** 252 - 1, rel=1e-6)


def test_psr_and_deflation():
    rng = np.random.default_rng(0)
    good = rng.normal(0.001, 0.01, 1000)
    assert probabilistic_sharpe(good) > 0.9
    assert probabilistic_sharpe(-good) < 0.1
    trials = rng.normal(0, 0.05, 50)  # 50 strategies tried
    assert deflated_sharpe(good, trials) < probabilistic_sharpe(good)


def test_bootstrap_ci_brackets_estimate():
    r = np.random.default_rng(1).normal(0.0005, 0.01, 750)
    lo, hi = bootstrap_sharpe_ci(r)
    assert lo < sharpe(r) < hi
