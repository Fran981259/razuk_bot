"""Pacote conector assíncrono para a corretora Olymp Trade (FTT / Fixed Time Trades)."""

from .olymp_client import OlympTradeClient
from .olymp_constants import (
    DEFAULT_WS_URL,
    TIMEFRAME_SECONDS,
    PAIR_ALIASES,
    DIRECTION_MAP,
)

__all__ = [
    "OlympTradeClient",
    "DEFAULT_WS_URL",
    "TIMEFRAME_SECONDS",
    "PAIR_ALIASES",
    "DIRECTION_MAP",
]
