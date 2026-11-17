"""Detector de Tendência por Médias Móveis, Suporte/Resistência e Choppiness Index.

Fornece leitura estrutural de mercado em tempo real para opções binárias e Forex:
1. MATrendDetector: Analisa alinhamento em cascata e inclinação angular das EMAs 9, 21, 50 e 200,
   e calcula o Choppiness Index (CHOP) de Dreiss para quantificar regimes de tendência vs consolidação.
2. SupportResistanceDetector: Mapeia topos e fundos fractais (Pivots), zonas de suporte/resistência,
   distâncias percentuais do preço atual e testes de rompimento/retração.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TrendDirection(str, Enum):
    """Direção da tendência definida pelas médias móveis."""

    ALTA = "ALTA"
    BAIXA = "BAIXA"
    LATERAL = "LATERAL"


@dataclass(frozen=True)
class TrendAnalysis:
    """Resultado detalhado da análise de médias móveis e volatilidade estrutural."""

    direction: TrendDirection
    strength_pct: float
    ema9: float
    ema21: float
    ema50: float
    ema200: float
    is_aligned: bool
    summary: str
    chop_index: float = 50.0  # > 61.8 = Consolidado / < 38.2 = Forte Tendência


@dataclass(frozen=True)
class SRAnalysis:
    """Resultado detalhado da análise de suporte e resistência."""

    current_price: float
    nearest_support: float
    nearest_resistance: float
    dist_support_pct: float
    dist_resistance_pct: float
    is_near_support: bool
    is_near_resistance: bool
    breakout_up: bool
    breakout_down: bool
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


class MATrendDetector:
    """Detector de tendência estrutural baseado em EMAs (9, 21, 50, 200) e Choppiness Index."""

    def __init__(
        self,
        span_fast: int = 9,
        span_med: int = 21,
        span_slow: int = 50,
        span_macro: int = 200,
        chop_period: int = 14,
    ) -> None:
        """Inicializa os períodos das médias móveis exponenciais e índice Choppiness."""
        self.fast = span_fast
        self.med = span_med
        self.slow = span_slow
        self.macro = span_macro
        self.chop_period = chop_period

    def calculate_choppiness(self, df: pd.DataFrame) -> float:
        """Calcula o Choppiness Index (CHOP) de Dreiss (escala 0-100)."""
        if len(df) < self.chop_period + 1:
            return 50.0

        h = df["high"].iloc[-self.chop_period - 1 :]
        l = df["low"].iloc[-self.chop_period - 1 :]
        c = df["close"].iloc[-self.chop_period - 1 :]

        tr1 = h.iloc[1:] - l.iloc[1:]
        tr2 = (h.iloc[1:] - c.iloc[:-1].values).abs()
        tr3 = (l.iloc[1:] - c.iloc[:-1].values).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        sum_tr = float(tr.sum())

        max_h = float(h.iloc[1:].max())
        min_l = float(l.iloc[1:].min())
        diff_hl = max_h - min_l
        if diff_hl <= 0:
            return 50.0

        chop = 100.0 * (np.log10(sum_tr / diff_hl) / np.log10(self.chop_period))
        return float(np.clip(chop, 0.0, 100.0))

    def analyze(self, df: pd.DataFrame) -> TrendAnalysis:
        """Avalia o alinhamento das médias e determina a direção e força da tendência."""
        if len(df) < max(self.fast, self.med, self.slow) + 5:
            return TrendAnalysis(
                direction=TrendDirection.LATERAL, strength_pct=0.0,
                ema9=0.0, ema21=0.0, ema50=0.0, ema200=0.0,
                is_aligned=False, summary="Histórico insuficiente para cálculo de EMAs",
                chop_index=50.0,
            )

        close = df["close"]
        ema9 = close.ewm(span=self.fast, adjust=False).mean()
        ema21 = close.ewm(span=self.med, adjust=False).mean()
        ema50 = close.ewm(span=self.slow, adjust=False).mean()

        span_m = self.macro if len(df) >= self.macro + 5 else min(100, len(df) - 1)
        ema200 = close.ewm(span=span_m, adjust=False).mean()

        v_close = float(close.iloc[-1])
        v_ema9 = float(ema9.iloc[-1])
        v_ema21 = float(ema21.iloc[-1])
        v_ema50 = float(ema50.iloc[-1])
        v_ema200 = float(ema200.iloc[-1])

        slope_9 = v_ema9 - float(ema9.iloc[-4]) if len(ema9) >= 4 else 0.0
        slope_21 = v_ema21 - float(ema21.iloc[-4]) if len(ema21) >= 4 else 0.0

        bullish_alignment = (v_ema9 > v_ema21 > v_ema50) and (v_close > v_ema9)
        bearish_alignment = (v_ema9 < v_ema21 < v_ema50) and (v_close < v_ema9)
        chop_val = self.calculate_choppiness(df)

        if bullish_alignment and slope_9 > 0 and slope_21 > 0:
            direction = TrendDirection.ALTA
            strength = 100.0 if v_ema50 > v_ema200 else 75.0
            summary = f"Forte Tendência de ALTA (EMA 9={v_ema9:.5f} > 21={v_ema21:.5f} > 50={v_ema50:.5f} | CHOP={chop_val:.1f})"
            aligned = True
        elif bearish_alignment and slope_9 < 0 and slope_21 < 0:
            direction = TrendDirection.BAIXA
            strength = 100.0 if v_ema50 < v_ema200 else 75.0
            summary = f"Forte Tendência de BAIXA (EMA 9={v_ema9:.5f} < 21={v_ema21:.5f} < 50={v_ema50:.5f} | CHOP={chop_val:.1f})"
            aligned = True
        else:
            direction = TrendDirection.LATERAL
            strength = 30.0
            summary = f"Mercado Lateral / Médias Cruzadas (CHOP={chop_val:.1f})"
            aligned = False

        return TrendAnalysis(
            direction=direction, strength_pct=strength,
            ema9=round(v_ema9, 5), ema21=round(v_ema21, 5),
            ema50=round(v_ema50, 5), ema200=round(v_ema200, 5),
            is_aligned=aligned, summary=summary,
            chop_index=round(chop_val, 2),
        )


class SupportResistanceDetector:
    """Detector dinâmico de zonas de suporte e resistência baseado em Pivots/Fractais."""

    def __init__(self, pivot_window: int = 3, tolerance_pct: float = 0.0006) -> None:
        """Inicializa parâmetros de detecção fractal."""
        self.window = pivot_window
        self.tolerance = tolerance_pct

    def analyze(self, df: pd.DataFrame) -> SRAnalysis:
        """Calcula os níveis de suporte e resistência mais recentes e analisa o preço atual."""
        if len(df) < (self.window * 2) + 5:
            p_atual = float(df["close"].iloc[-1]) if not df.empty else 0.0
            return SRAnalysis(
                current_price=p_atual, nearest_support=p_atual, nearest_resistance=p_atual,
                dist_support_pct=0.0, dist_resistance_pct=0.0,
                is_near_support=False, is_near_resistance=False,
                breakout_up=False, breakout_down=False,
            )

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)
        w = self.window

        pivots_high: List[float] = []
        pivots_low: List[float] = []

        for i in range(w, n - 1):
            if all(highs[i] > highs[i - j] for j in range(1, w + 1)) and all(highs[i] >= highs[i + j] for j in range(1, min(w + 1, n - i))):
                pivots_high.append(float(highs[i]))
            if all(lows[i] < lows[i - j] for j in range(1, w + 1)) and all(lows[i] <= lows[i + j] for j in range(1, min(w + 1, n - i))):
                pivots_low.append(float(lows[i]))

        current_price = float(closes[-1])
        prev_close = float(closes[-2]) if n >= 2 else current_price

        supports = sorted([p for p in pivots_low if p < current_price], reverse=True)
        resistances = sorted([p for p in pivots_high if p > current_price])

        nearest_supp = supports[0] if supports else float(np.min(lows[-30:]))
        nearest_res = resistances[0] if resistances else float(np.max(highs[-30:]))

        dist_supp_pct = abs(current_price - nearest_supp) / current_price if current_price > 0 else 0.0
        dist_res_pct = abs(nearest_res - current_price) / current_price if current_price > 0 else 0.0

        is_near_support = dist_supp_pct <= self.tolerance
        is_near_resistance = dist_res_pct <= self.tolerance

        breakout_up = (prev_close <= nearest_res) and (current_price > nearest_res)
        breakout_down = (prev_close >= nearest_supp) and (current_price < nearest_supp)

        return SRAnalysis(
            current_price=round(current_price, 5),
            nearest_support=round(nearest_supp, 5),
            nearest_resistance=round(nearest_res, 5),
            dist_support_pct=round(dist_supp_pct * 100, 3),
            dist_resistance_pct=round(dist_res_pct * 100, 3),
            is_near_support=is_near_support,
            is_near_resistance=is_near_resistance,
            breakout_up=breakout_up,
            breakout_down=breakout_down,
            support_levels=[round(s, 5) for s in supports[:3]],
            resistance_levels=[round(r, 5) for r in resistances[:3]],
        )


class MarketStructure:
    """Fachada unificada para análise de Tendência (EMAs, CHOP) e Suporte/Resistência."""

    def __init__(self) -> None:
        """Inicializa os detectores de tendência e estrutura de preço."""
        self.trend_detector = MATrendDetector()
        self.sr_detector = SupportResistanceDetector()

    def evaluate(self, df: pd.DataFrame) -> Tuple[TrendAnalysis, SRAnalysis]:
        """Executa a análise conjunta de tendência e suporte/resistência em um DataFrame OHLCV."""
        trend = self.trend_detector.analyze(df)
        sr = self.sr_detector.analyze(df)
        return trend, sr
