from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from quantlab import IndianEquityCosts, PaperBroker, PaperTrader, PurgedWalkForward, walk_forward_predict
from quantlab.data import synthetic_bars
from quantlab.features import make_features, make_labels
from quantlab.live import CandleBuilder, OrderRejected


def test_features_are_point_in_time():
    bars = synthetic_bars(days=200, seed=4)
    base = make_features(bars)
    shocked = bars.copy()
    shocked.iloc[150:, :4] *= 3.0  # rewrite the future
    after = make_features(shocked)
    pd.testing.assert_frame_equal(base.iloc[:150], after.iloc[:150])


def test_labels_look_forward_exactly_horizon():
    bars = synthetic_bars(days=10)
    y = make_labels(bars, horizon=2)
    assert y.iloc[-2:].isna().all()
    assert y.iloc[0] == float(bars.close.iloc[2] > bars.close.iloc[0])


def test_purged_walk_forward_never_trains_on_test_or_future():
    sp = PurgedWalkForward(train_size=100, test_size=20, horizon=3, embargo=2, expanding=False)
    folds = list(sp.split(300))
    assert folds
    prev_test_end = None
    for train, test in folds:
        assert train.max() <= test.min() - 1 - (3 + 2) + 1  # purge + embargo gap
        assert len(train) <= 100
        if prev_test_end is not None:
            assert test.min() == prev_test_end + 1
        prev_test_end = test.max()


def test_walk_forward_finds_no_edge_in_a_random_walk():
    bars = synthetic_bars(days=1500, seed=9)
    p = walk_forward_predict(lambda: LogisticRegression(max_iter=500), make_features(bars), make_labels(bars),
                             PurgedWalkForward(400, 50))
    y = make_labels(bars).reindex(p.index)
    acc = float(((p > 0.5).astype(float) == y).mean())
    assert 0.44 < acc < 0.56


def test_candle_builder():
    cb = CandleBuilder(minutes=5)
    t = datetime(2026, 11, 2, 9, 15)
    ticks = [(0, 100), (60, 103), (120, 99), (240, 101), (300, 102)]
    closed = [cb.add(t.replace(minute=15 + s // 60, second=s % 60), p) for s, p in ticks]
    assert closed == [False, False, False, False, True]
    row = cb.frame().iloc[0]
    assert (row.open, row.high, row.low, row.close) == (100, 103, 99, 101)


def test_paper_broker_fills_and_rejections():
    b = PaperBroker(cash=100_000, costs=IndianEquityCosts("delivery"), slippage_bps=0)
    t = datetime(2026, 11, 2, 10, 0)
    f = b.place_market_order("INFY", 50, 1500.0, t)
    assert f.price == 1500.0 and b.positions["INFY"] == 50
    assert b.cash == pytest.approx(100_000 - 75_000 - f.charges)
    with pytest.raises(OrderRejected):
        b.place_market_order("INFY", -60, 1500.0, t)  # would go short in CNC
    with pytest.raises(OrderRejected):
        b.place_market_order("TCS", 100, 4000.0, t)  # exceeds 1x leverage


def test_paper_trader_squares_off_and_respects_loss_limit():
    class AlwaysLong:
        def target(self, ctx):
            return 1.0

    broker = PaperBroker(cash=1_000_000, costs=IndianEquityCosts("intraday"))
    trader = PaperTrader("RELIANCE", AlwaysLong(), broker, candle_minutes=1)
    day = datetime(2026, 11, 2, 9, 15)
    for i, price in enumerate(np.linspace(2500, 2510, 30)):
        trader.step(day.replace(minute=15 + i), float(price))
    assert broker.positions["RELIANCE"] > 0
    trader.step(day.replace(hour=15, minute=16), 2512.0)
    assert broker.positions["RELIANCE"] == 0 and broker.fills[-1].reason == "eod_square_off"

    next_day = datetime(2026, 11, 3, 9, 15)
    for i in range(3):
        trader.step(next_day.replace(minute=15 + i), 2500.0)
    trader.step(next_day.replace(minute=19), 2400.0)  # -4% gap down
    assert trader.halted and broker.positions["RELIANCE"] == 0
    assert broker.fills[-1].reason == "daily_loss_limit"
