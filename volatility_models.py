"""Modelos de Volatilidade de Amplitude de Preço (Parkinson e Garman-Klass).

Inspirado em agent/src/quantlib/volatility.py do Vibe-Trading:
Diferente da volatilidade convencional de fechamento (close-to-close), as métricas
de Parkinson e Garman-Klass utilizam a excursão intradiária de Máximas e Mínimas (High-Low)
e Aberturas (Open), alcançando até 5x mais eficiência estatística e detecção antecipada
de exaustão ou expansão de volatilidade em velas M5.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Final, Tuple

import numpy as np
import pandas as pd

_EPS: Final[float] = 1e-9


class VolatilityRegime(str, Enum):
    """Regime de volatilidade do par."""

    LOW_COMPRESSION = "LOW_COMPRESSION"      # Compressão (ideal para rompimento)
    NORMAL = "NORMAL"                        # Volatilidade equilibrada
    HIGH_EXPANSION = "HIGH_EXPANSION"        # Forte expansão direcional
    EXTREME_EXHAUSTION = "EXTREME_EXHAUSTION"# Exaustão (risco de reversão violenta)


@dataclass(frozen=True)
class VolatilityReport:
    """Relatório estruturado de volatilidade da vela recente."""

    parkinson_vol: float
    garman_klass_vol: float
    vol_ratio_to_mean: float
    regime: VolatilityRegime
    is_safe_for_trend: bool


def parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calcula a volatilidade de Parkinson baseada na amplitude High-Low.

    Fórmula: sqrt( (1 / (4 * ln(2) * window)) * sum( (ln(High / Low))^2 ) )
    """
    if len(df) < 5:
        return pd.Series(0.0, index=df.index)

    h = np.maximum(df["high"].values, _EPS)
    l = np.maximum(df["low"].values, _EPS)
    log_hl = np.log(h / l)
    log_hl_sq = log_hl ** 2.0

    factor = 1.0 / (4.0 * np.log(2.0))
    s_sq = pd.Series(log_hl_sq, index=df.index)
    roll_mean = s_sq.rolling(window=window, min_periods=3).mean()
    parkinson = np.sqrt(factor * roll_mean)
    return parkinson.fillna(0.0)


def garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calcula a volatilidade de Garman-Klass (incorporando Open, High, Low e Close).

    Fórmula: sqrt( mean( 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2 ) )
    """
    if len(df) < 5:
        return pd.Series(0.0, index=df.index)

    h = np.maximum(df["high"].values, _EPS)
    l = np.maximum(df["low"].values, _EPS)
    o = np.maximum(df["open"].values, _EPS)
    c = np.maximum(df["close"].values, _EPS)

    term1 = 0.5 * (np.log(h / l) ** 2.0)
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(c / o) ** 2.0)
    gk_sq = term1 - term2

    s_gk = pd.Series(np.maximum(gk_sq, 0.0), index=df.index)
    roll_mean = s_gk.rolling(window=window, min_periods=3).mean()
    gk_vol = np.sqrt(roll_mean)
    return gk_vol.fillna(0.0)


def evaluate_volatility_regime(df: pd.DataFrame, window: int = 20) -> VolatilityReport:
    """Avalia o regime de volatilidade da última barra em relação ao histórico."""
    if len(df) < window + 5:
        return VolatilityReport(
            parkinson_vol=0.001,
            garman_klass_vol=0.001,
            vol_ratio_to_mean=1.0,
            regime=VolatilityRegime.NORMAL,
            is_safe_for_trend=True,
        )

    gk_series = garman_klass_volatility(df, window=window)
    pk_series = parkinson_volatility(df, window=window)

    curr_gk = float(gk_series.iloc[-1])
    curr_pk = float(pk_series.iloc[-1])

    # Média dos últimos 50 períodos para determinar desvio relativo
    mean_gk = float(gk_series.iloc[-50:].mean()) if len(gk_series) >= 50 else curr_gk
    ratio = curr_gk / max(mean_gk, _EPS)

    if ratio < 0.65:
        regime = VolatilityRegime.LOW_COMPRESSION
        safe = True
    elif ratio > 2.50:
        regime = VolatilityRegime.EXTREME_EXHAUSTION
        safe = False  # Mercado sobre-esticado / risco alto de repique
    elif ratio > 1.40:
        regime = VolatilityRegime.HIGH_EXPANSION
        safe = True
    else:
        regime = VolatilityRegime.NORMAL
        safe = True

    return VolatilityReport(
        parkinson_vol=round(curr_pk, 5),
        garman_klass_vol=round(curr_gk, 5),
        vol_ratio_to_mean=round(ratio, 2),
        regime=regime,
        is_safe_for_trend=safe,
    )
