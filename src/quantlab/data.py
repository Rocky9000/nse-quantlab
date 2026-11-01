"""Market data: loading, validation, and a synthetic generator for tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def validate(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and fail loudly on data that would corrupt a backtest."""
    df = bars.rename(columns=str.lower)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[COLUMNS].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("timestamps must be strictly increasing")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("price columns contain NaN")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive")
    bad = (df.high < df[["open", "close", "low"]].max(axis=1)) | (df.low > df[["open", "close", "high"]].min(axis=1))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} bars violate low <= open/close <= high")
    return df


def load_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
    return validate(df.sort_index())


def fetch_yahoo(symbol: str, start: str = "2015-01-01", end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    """Daily (or intraday, last 60 days) NSE bars from Yahoo, e.g. ``RELIANCE.NS``. Needs ``yfinance``."""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return validate(df.dropna())


def synthetic_bars(
    days: int = 500,
    bars_per_day: int = 1,
    start_price: float = 1000.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.25,
    seed: int = 0,
    start: str = "2022-01-03",
) -> pd.DataFrame:
    """Geometric random walk with realistic OHLC structure. Has **no** predictable signal by construction.

    With ``bars_per_day > 1`` it produces intraday bars at a fixed interval
    starting 09:15 IST, matching NSE session timing.
    """
    rng = np.random.default_rng(seed)
    n = days * bars_per_day
    dt = 1 / (252 * bars_per_day)
    rets = rng.normal((annual_drift - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[start_price, close[:-1]] * np.exp(rng.normal(0, annual_vol * np.sqrt(dt) * 0.2, n))
    spread = np.abs(rng.normal(0, annual_vol * np.sqrt(dt) * 0.6, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(12, 0.4, n).round()

    dates = pd.bdate_range(start, periods=days)
    if bars_per_day == 1:
        index = dates + pd.Timedelta(hours=15, minutes=30)
    else:
        step = pd.Timedelta(minutes=375 // bars_per_day)
        index = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=15) + i * step
                                  for d in dates for i in range(bars_per_day)])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)
