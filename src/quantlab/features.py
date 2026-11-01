"""Point-in-time features and labels. Row ``t`` uses only bars ``0..t``."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    close, volume = bars["close"], bars["volume"]
    logp = np.log(close)
    r = logp.diff()
    f = pd.DataFrame(index=bars.index)
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_{k}"] = logp.diff(k)
    f["vol_10"] = r.rolling(10).std()
    f["vol_30"] = r.rolling(30).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_30"]
    f["rsi_14"] = rsi(close, 14) / 100
    f["sma_gap_20"] = close / close.rolling(20).mean() - 1
    f["sma_gap_50"] = close / close.rolling(50).mean() - 1
    f["range"] = (bars["high"] - bars["low"]) / close
    f["gap"] = np.log(bars["open"] / close.shift(1))
    f["volume_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    return f


def make_labels(bars: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """1 if the close ``horizon`` bars ahead is higher than today's close, else 0 (NaN at the end)."""
    fwd = bars["close"].shift(-horizon) / bars["close"] - 1
    return (fwd > 0).astype(float).where(fwd.notna())
