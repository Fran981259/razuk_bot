"""Indicador e Estratégia EMA Wave + Clock Filter (Raghee Horner).

Implementa a Onda de Médias Exponenciais de 34 períodos (High, Close, Low)
e o Filtro do Relógio Analógico baseado na inclinação angular da média central.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class ClockState(str, Enum):
    """Estados do Relógio Analógico de 12 Horas de Raghee Horner."""

    BULLISH_STRONG = "BULLISH_STRONG"  # 12h00 às 02h30 (+10° a +90°)
    BEARISH_STRONG = "BEARISH_STRONG"  # 03h30 às 06h00 (-10° a -90°)
    NEUTRAL_SIDEWAYS = "NEUTRAL_SIDEWAYS"  # 03h00 / 09h00 (-10° a +10°)


class TradePermission(str, Enum):
    """Permissão de operação de acordo com a maré do relógio."""

    BUY_ONLY = "APENAS COMPRA"
    SELL_ONLY = "APENAS VENDA"
    OUT_OF_MARKET = "FORA DO MERCADO (NEUTRO)"


@dataclass(frozen=True)
class WaveSignal:
    """Resultado da análise da EMA Wave em tempo real."""

    state: ClockState
    angle_degrees: float
    permission: TradePermission
    signal: Optional[str]  # "CALL", "PUT" ou None
    high_wave: float
    close_wave: float
    low_wave: float


class EMAWaveIndicator:
    """Calculador técnico da EMA Wave de 34 períodos com filtro do relógio."""

    def __init__(self, period: int = 34, slope_bars: int = 3) -> None:
        """Inicializa os parâmetros do indicador.

        Args:
            period: Período base Fibonacci das EMAs (padrão: 34).
            slope_bars: Quantidade de barras para medição da inclinação (padrão: 3).
        """
        self.period = period
        self.slope_bars = slope_bars

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula as 3 bandas da onda, inclinação angular e estado do relógio."""
        if len(df) < self.period + self.slope_bars:
            return df.copy()

        df_calc = df.copy()
        high_col = "high" if "high" in df_calc.columns else "max"
        low_col = "low" if "low" in df_calc.columns else "min"
        close_col = "close"

        # 1. Formação da Banda (Onda de 3 Linhas)
        df_calc["wave_high"] = df_calc[high_col].ewm(span=self.period, adjust=False).mean()
        df_calc["wave_close"] = df_calc[close_col].ewm(span=self.period, adjust=False).mean()
        df_calc["wave_low"] = df_calc[low_col].ewm(span=self.period, adjust=False).mean()

        # 2. Cálculo da inclinação normalizada e ângulo
        # Usamos ATR(14) como normalizador de volatilidade de escala de preço
        tr1 = df_calc[high_col] - df_calc[low_col]
        tr2 = (df_calc[high_col] - df_calc[close_col].shift(1)).abs()
        tr3 = (df_calc[low_col] - df_calc[close_col].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=1).mean().replace(0, 1e-6)

        delta_ema = df_calc["wave_close"] - df_calc["wave_close"].shift(self.slope_bars)
        slope_norm = (delta_ema / (atr * self.slope_bars)).fillna(0.0)

        # Conversão em graus (-90° a +90°)
        df_calc["wave_angle"] = np.degrees(np.arctan(slope_norm * 2.5))
        return df_calc

    def classify_clock_state(self, angle: float) -> Tuple[ClockState, TradePermission]:
        """Classifica o ângulo nos ponteiros do relógio analógico de 12 horas."""
        if angle >= 10.0:
            return ClockState.BULLISH_STRONG, TradePermission.BUY_ONLY
        elif angle <= -10.0:
            return ClockState.BEARISH_STRONG, TradePermission.SELL_ONLY
        else:
            return ClockState.NEUTRAL_SIDEWAYS, TradePermission.OUT_OF_MARKET

    def evaluate_latest(self, df: pd.DataFrame) -> WaveSignal:
        """Avalia a última vela para geração de gatilhos operacionais."""
        df_calc = self.calculate(df)
        if len(df_calc) < 2 or "wave_close" not in df_calc.columns:
            return WaveSignal(
                state=ClockState.NEUTRAL_SIDEWAYS,
                angle_degrees=0.0,
                permission=TradePermission.OUT_OF_MARKET,
                signal=None,
                high_wave=0.0,
                close_wave=0.0,
                low_wave=0.0,
            )

        row = df_calc.iloc[-1]
        angle = float(row.get("wave_angle", 0.0))
        state, permission = self.classify_clock_state(angle)

        high_col = "high" if "high" in row else "max"
        low_col = "low" if "low" in row else "min"
        c_open = float(row.get("open", 0.0))
        c_high = float(row.get(high_col, 0.0))
        c_low = float(row.get(low_col, 0.0))
        c_close = float(row.get("close", 0.0))

        w_high = float(row["wave_high"])
        w_close = float(row["wave_close"])
        w_low = float(row["wave_low"])

        signal: Optional[str] = None

        total_range = max(c_high - c_low, 1e-6)
        body = abs(c_close - c_open)
        is_doji = (body / total_range) < 0.08

        # Condição de Compra (CALL - IQ Option M1):
        # 1. Relógio em BULLISH_STRONG (12h-2h)
        # 2. Retração na onda (Low <= w_high e Close >= w_low)
        # 3. Rejeição de fundo ou fechamento comprador sem Doji
        if state == ClockState.BULLISH_STRONG and not is_doji:
            lower_shadow = min(c_open, c_close) - c_low
            rejeitou_fundo = (lower_shadow / total_range) >= 0.20
            if c_low <= w_high and c_close >= w_low and (c_close > c_open or rejeitou_fundo):
                signal = "CALL"

        # Condição de Venda (PUT - IQ Option M1):
        # 1. Relógio em BEARISH_STRONG (3h30-5h)
        # 2. Retração na onda (High >= w_low e Close <= w_high)
        # 3. Rejeição de topo ou fechamento vendedor sem Doji
        elif state == ClockState.BEARISH_STRONG and not is_doji:
            upper_shadow = c_high - max(c_open, c_close)
            rejeitou_topo = (upper_shadow / total_range) >= 0.20
            if c_high >= w_low and c_close <= w_high and (c_close < c_open or rejeitou_topo):
                signal = "PUT"

        return WaveSignal(
            state=state,
            angle_degrees=round(angle, 2),
            permission=permission,
            signal=signal,
            high_wave=round(w_high, 5),
            close_wave=round(w_close, 5),
            low_wave=round(w_low, 5),
        )
