"""Motor de Análise Institucional Smart Money Concepts (SMC).

Inspirado na arquitetura do Vibe-Trading (ICT / SMC):
Identifica blocos de ordens (Order Blocks) e desequilíbrios de preço (Fair Value Gaps - FVG)
em gráficos M5 para validação de entradas em pontos de mitigação institucional.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class ZoneType(str, Enum):
    """Tipo da zona institucional."""

    BULLISH_OB = "BULLISH_OB"      # Bloco de Ordens de Compra
    BEARISH_OB = "BEARISH_OB"      # Bloco de Ordens de Venda
    BULLISH_FVG = "BULLISH_FVG"    # Desequilíbrio de Compra (Gap)
    BEARISH_FVG = "BEARISH_FVG"    # Desequilíbrio de Venda (Gap)


@dataclass(frozen=True)
class SMCZone:
    """Zona de liquidez / bloco de ordens institucional."""

    zone_type: ZoneType
    top: float
    bottom: float
    mid: float
    bar_index: int
    is_mitigated: bool = False


@dataclass(frozen=True)
class SMCAnalysis:
    """Diagnóstico consolidado de Smart Money Concepts para a vela atual."""

    has_bullish_fvg: bool
    has_bearish_fvg: bool
    nearest_bullish_ob: Optional[float]
    nearest_bearish_ob: Optional[float]
    in_bullish_mitigation: bool  # Preço atual testando bloco de compra
    in_bearish_mitigation: bool  # Preço atual testando bloco de venda
    institutional_signal: Optional[str]  # "CALL", "PUT" ou None
    summary: str


class SMCDetector:
    """Detector vetorizado de Order Blocks e Fair Value Gaps."""

    def __init__(self, fvg_min_pct: float = 0.0003, lookback_bars: int = 50) -> None:
        """Inicializa o detector SMC.

        Args:
            fvg_min_pct: Tamanho percentual mínimo de gap entre velas 1 e 3 para validar FVG.
            lookback_bars: Janela máxima de barras para rastrear blocos ativos.
        """
        self.fvg_min_pct = fvg_min_pct
        self.lookback = lookback_bars

    def detect_fvg(self, df: pd.DataFrame) -> Tuple[List[SMCZone], List[SMCZone]]:
        """Identifica Fair Value Gaps (FVG) de 3 velas (candle 1 high vs candle 3 low)."""
        bullish_fvgs: List[SMCZone] = []
        bearish_fvgs: List[SMCZone] = []

        if len(df) < 5:
            return bullish_fvgs, bearish_fvgs

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)
        start_idx = max(2, n - self.lookback)

        for i in range(start_idx, n):
            # Bullish FVG: Low da vela atual > High de 2 velas atrás
            gap_up = lows[i] - highs[i - 2]
            if gap_up > (closes[i] * self.fvg_min_pct):
                top, bottom = lows[i], highs[i - 2]
                bullish_fvgs.append(
                    SMCZone(ZoneType.BULLISH_FVG, top, bottom, (top + bottom) / 2.0, i)
                )

            # Bearish FVG: High da vela atual < Low de 2 velas atrás
            gap_dn = lows[i - 2] - highs[i]
            if gap_dn > (closes[i] * self.fvg_min_pct):
                top, bottom = lows[i - 2], highs[i]
                bearish_fvgs.append(
                    SMCZone(ZoneType.BEARISH_FVG, top, bottom, (top + bottom) / 2.0, i)
                )

        return bullish_fvgs, bearish_fvgs

    def detect_order_blocks(self, df: pd.DataFrame) -> Tuple[List[SMCZone], List[SMCZone]]:
        """Identifica Order Blocks: última vela contrária antes de forte impulso."""
        bull_obs: List[SMCZone] = []
        bear_obs: List[SMCZone] = []

        if len(df) < 6:
            return bull_obs, bear_obs

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)
        start_idx = max(2, n - self.lookback)

        for i in range(start_idx, n - 2):
            # Impulso de alta: duas velas fortes de alta após uma vela de baixa
            is_prev_bear = closes[i] < opens[i]
            impulse_up = (closes[i + 1] > opens[i + 1]) and (closes[i + 2] > highs[i])
            if is_prev_bear and impulse_up:
                top, bot = highs[i], lows[i]
                bull_obs.append(SMCZone(ZoneType.BULLISH_OB, top, bot, (top + bot) / 2.0, i))

            # Impulso de baixa: duas velas fortes de baixa após uma vela de alta
            is_prev_bull = closes[i] > opens[i]
            impulse_dn = (closes[i + 1] < opens[i + 1]) and (closes[i + 2] < lows[i])
            if is_prev_bull and impulse_dn:
                top, bot = highs[i], lows[i]
                bear_obs.append(SMCZone(ZoneType.BEARISH_OB, top, bot, (top + bot) / 2.0, i))

        return bull_obs, bear_obs

    def analyze(self, df: pd.DataFrame) -> SMCAnalysis:
        """Avalia a confluência institucional SMC na vela atual."""
        if len(df) < 10:
            return SMCAnalysis(False, False, None, None, False, False, None, "Dados insuficientes para SMC")

        curr_price = float(df["close"].iloc[-1])
        b_fvgs, s_fvgs = self.detect_fvg(df)
        b_obs, s_obs = self.detect_order_blocks(df)

        # Checa mitigação no Order Block ou FVG mais recente
        in_bull_mitig = any(ob.bottom <= curr_price <= ob.top for ob in b_obs[-3:])
        in_bear_mitig = any(ob.bottom <= curr_price <= ob.top for ob in s_obs[-3:])

        nearest_b_ob = b_obs[-1].mid if b_obs else None
        nearest_s_ob = s_obs[-1].mid if s_obs else None

        sig = None
        if in_bull_mitig and not in_bear_mitig:
            sig = "CALL"
            summary = "Preço mitigando Bullish Order Block (Apoio Institucional)"
        elif in_bear_mitig and not in_bull_mitig:
            sig = "PUT"
            summary = "Preço mitigando Bearish Order Block (Pressão Vendedora)"
        else:
            summary = "SMC Neutro (Sem teste direto de bloco institucional)"

        return SMCAnalysis(
            has_bullish_fvg=len(b_fvgs) > 0,
            has_bearish_fvg=len(s_fvgs) > 0,
            nearest_bullish_ob=nearest_b_ob,
            nearest_bearish_ob=nearest_s_ob,
            in_bullish_mitigation=in_bull_mitig,
            in_bearish_mitigation=in_bear_mitig,
            institutional_signal=sig,
            summary=summary,
        )
