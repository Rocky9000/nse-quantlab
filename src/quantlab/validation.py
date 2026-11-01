"""Walk-forward validation with purging and embargo.

Shuffled k-fold cross-validation leaks the future into the past on time
series. Here every model is trained only on data that ends strictly before
its test window, with the last ``horizon`` training rows purged (their labels
peek into the test window) and an extra ``embargo`` gap for serial
correlation (López de Prado, *Advances in Financial Machine Learning*, ch. 7).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForward:
    train_size: int = 500
    test_size: int = 60
    horizon: int = 1
    embargo: int = 5
    expanding: bool = True

    def split(self, n: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        gap = self.horizon + self.embargo
        start = self.train_size + gap
        while start < n:
            test = np.arange(start, min(start + self.test_size, n))
            train_end = start - gap
            train_start = 0 if self.expanding else max(0, train_end - self.train_size)
            yield np.arange(train_start, train_end), test
            start += self.test_size


def walk_forward_predict(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: PurgedWalkForward,
) -> pd.Series:
    """Out-of-sample P(up) for every row the splitter tests. Rows with NaN features/labels are dropped first."""
    data = X.join(y.rename("__y__")).dropna()
    Xv, yv = data.drop(columns="__y__").to_numpy(), data["__y__"].to_numpy()
    out = pd.Series(np.nan, index=data.index, name="p_up")
    for train, test in splitter.split(len(data)):
        if len(np.unique(yv[train])) < 2:
            continue
        model = model_factory()
        model.fit(Xv[train], yv[train])
        out.iloc[test] = model.predict_proba(Xv[test])[:, 1]
    return out.dropna()


def in_sample_predict(model_factory: Callable[[], object], X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """The *wrong* way, kept for comparison: fit and predict on the same rows."""
    data = X.join(y.rename("__y__")).dropna()
    model = model_factory()
    Xv = data.drop(columns="__y__").to_numpy()
    model.fit(Xv, data["__y__"].to_numpy())
    return pd.Series(model.predict_proba(Xv)[:, 1], index=data.index, name="p_up")
