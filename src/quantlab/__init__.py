"""Event-driven backtesting, ML research, and paper trading for Indian equities."""

from .costs import IndianEquityCosts
from .engine import BacktestConfig, Backtester, BacktestResult, Context
from .live import PaperBroker, PaperTrader
from .strategies import BuyAndHold, SignalStrategy, SMACrossover, ZScoreMeanReversion
from .validation import PurgedWalkForward, walk_forward_predict

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "BuyAndHold",
    "Context",
    "IndianEquityCosts",
    "PaperBroker",
    "PaperTrader",
    "PurgedWalkForward",
    "SMACrossover",
    "SignalStrategy",
    "ZScoreMeanReversion",
    "walk_forward_predict",
]
__version__ = "0.1.0"
