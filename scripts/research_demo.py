"""Three experiments showing how backtests lie, and how this framework catches it.

All run on synthetic random-walk prices, which contain **no** predictable
signal by construction. Any edge a method "finds" here is an artefact.

  1. ML look-ahead: in-sample vs purged walk-forward evaluation.
  2. Selection bias: best of a parameter sweep vs its deflated Sharpe.
  3. Cost drag: an intraday strategy before and after real Indian charges.

    python scripts/research_demo.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from quantlab.costs import IndianEquityCosts, ProductRates
from quantlab.data import synthetic_bars
from quantlab.engine import BacktestConfig, Backtester
from quantlab.features import make_features, make_labels
from quantlab.metrics import daily_returns, deflated_sharpe, probabilistic_sharpe
from quantlab.strategies import SignalStrategy, SMACrossover, ZScoreMeanReversion
from quantlab.validation import PurgedWalkForward, in_sample_predict, walk_forward_predict


def model():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)


def accuracy(p, y):
    y = y.reindex(p.index)
    return float(((p > 0.5).astype(float) == y).mean())


def experiment_lookahead() -> None:
    bars = synthetic_bars(days=2000, seed=7)
    X, y = make_features(bars), make_labels(bars)
    cfg = BacktestConfig(product="delivery", daily_loss_limit=None, max_drawdown_halt=None)
    rows = []
    for name, p in [("in-sample (wrong)", in_sample_predict(model, X, y)),
                    ("purged walk-forward", walk_forward_predict(model, X, y, PurgedWalkForward(500, 60, 1, 5)))]:
        res = Backtester(cfg).run(bars.loc[p.index[0]:], SignalStrategy(p, band=0.02, allow_short=False))
        rows.append((name, accuracy(p, y), res.metrics["sharpe"], probabilistic_sharpe(daily_returns(res.equity))))
    print("### 1. ML on a random walk: in-sample vs walk-forward\n")
    print("| Evaluation | Direction accuracy | Backtest Sharpe | P(Sharpe > 0) |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.2f} | {r[3]:.3f} |")


def experiment_selection_bias() -> None:
    bars = synthetic_bars(days=1500, seed=3)
    cfg = BacktestConfig(product="delivery", daily_loss_limit=None, max_drawdown_halt=None)
    results = []
    for fast in (5, 10, 15, 20, 30, 40):
        for slow in (50, 75, 100, 150, 200):
            res = Backtester(cfg).run(bars, SMACrossover(fast, slow))
            results.append(((fast, slow), daily_returns(res.equity)))
    trial_sr = [r.mean() / r.std(ddof=1) for _, r in results]
    best = int(np.argmax(trial_sr))
    (fast, slow), r_best = results[best]
    print(f"\n### 2. Selection bias: best of {len(results)} SMA crossovers on a random walk\n")
    print("| Best params | Annualised Sharpe | PSR (vs 0) | Deflated Sharpe (vs best-of-N luck) |\n|---|---|---|---|")
    print(f"| {fast}/{slow} | {trial_sr[best] * np.sqrt(252):.2f} | {probabilistic_sharpe(r_best):.3f} "
          f"| {deflated_sharpe(r_best, trial_sr):.3f} |")


def experiment_costs() -> None:
    bars = synthetic_bars(days=250, bars_per_day=75, seed=11, annual_drift=0.0)  # 5-minute bars
    free = IndianEquityCosts("intraday", exchange_pct=0, sebi_per_crore=0, gst_pct=0,
                             rates={"intraday": ProductRates(0, None, 0, 0, 0, 0)})
    strat = ZScoreMeanReversion(window=12, entry=1.5, exit=0.3)
    print("\n### 3. Cost drag: 5-minute intraday mean reversion, one year\n")
    print("| Costs | Trades | Turnover (x capital) | Charges (% of capital) | Net Sharpe | Total return |")
    print("|---|---|---|---|---|---|")
    scenarios = [("none", free, 0.0), ("Zerodha MIS + 2 bps slippage", IndianEquityCosts("intraday"), 2.0)]
    for name, costs, slip in scenarios:
        cfg = BacktestConfig(product="intraday", slippage_bps=slip, daily_loss_limit=None, max_drawdown_halt=None)
        s = Backtester(cfg, costs).run(bars, strat).summary()
        print(f"| {name} | {s['trades']} | {s['turnover_x']:.0f} | {100 * s['charges_pct_of_capital']:.1f}% "
              f"| {s['sharpe']:.2f} | {100 * s['total_return']:.1f}% |")


if __name__ == "__main__":
    experiment_lookahead()
    experiment_selection_bias()
    experiment_costs()
