"""Performance and statistical-significance metrics.

Sharpe ratios from backtests are biased upward by non-normal returns, short
samples, and above all by trying many strategies and reporting the best.
``probabilistic_sharpe`` and ``deflated_sharpe`` (Bailey & López de Prado,
2012/2014) correct for those effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649


def daily_returns(equity: pd.Series) -> pd.Series:
    """Collapse any bar frequency to daily close-to-close returns."""
    daily = equity.groupby(equity.index.normalize()).last()
    return daily.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def sharpe(returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 1e-12 else 0.0


def performance(equity: pd.Series) -> dict:
    r = daily_returns(equity)
    years = len(r) / TRADING_DAYS
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
    downside = r[r < 0].std(ddof=1) if (r < 0).sum() > 1 else 0.0
    mdd = max_drawdown(equity)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    return {
        "total_return": total,
        "cagr": float(cagr),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0,
        "sharpe": sharpe(r),
        "sortino": float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "positive_days": float((r > 0).mean()) if len(r) else 0.0,
        "days": int(len(r)),
    }


def probabilistic_sharpe(returns: pd.Series | np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``) given sample length, skew and kurtosis."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3 or r.std(ddof=1) <= 1e-12:
        return 0.5
    sr = r.mean() / r.std(ddof=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(returns: pd.Series | np.ndarray, trial_sharpes: list[float] | np.ndarray) -> float:
    """PSR against the Sharpe you would expect from the best of N *unskilled* trials.

    ``trial_sharpes`` are the per-period Sharpe ratios of every configuration
    you tried (including the chosen one). Their dispersion sets how high a
    Sharpe pure luck can produce.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    n = trials.size
    if n < 2:
        return probabilistic_sharpe(returns)
    sd = trials.std(ddof=1)
    expected_max = sd * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e)))
    return probabilistic_sharpe(returns, expected_max)


def bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray, n_boot: int = 2000, block: int = 10, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Moving-block bootstrap CI for annualised Sharpe (blocks keep autocorrelation)."""
    r = np.asarray(returns, dtype=float)
    if r.size < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(r.size / block))
    starts = rng.integers(0, r.size - block + 1, size=(n_boot, n_blocks))
    samples = r[(starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, : r.size]]
    sds = samples.std(axis=1, ddof=1)
    srs = np.where(sds > 0, samples.mean(axis=1) / np.where(sds > 0, sds, 1) * np.sqrt(TRADING_DAYS), 0.0)
    lo, hi = np.quantile(srs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
