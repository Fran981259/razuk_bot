"""Alpha Factors Quantitativos adaptados do Alpha Zoo (Vibe-Trading / WorldQuant / Qlib).

Implementa fatores de alta capacidade preditiva para séries OHLCV em M5:
1. Alpha101_001 (Kakushadze): Momentum condicional com ponderação de volatilidade.
2. Alpha101_012 (Kakushadze): Sinal de variação de momentum de volume e preço.
3. Qlib_NormReturn (Microsoft Qlib): Variação de preço normalizada pelo desvio padrão móvel.
4. Qlib_EMA_Curvature: Taxa de aceleração da inclinação da média móvel rápida.
"""

from typing import Dict, Final

import numpy as np
import pandas as pd

_EPS: Final[float] = 1e-9


def calc_alpha101_001(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Alpha #1 de Kakushadze (WorldQuant): Momentum condicional de volatilidade.

    Fórmula: rank(SignedPower((returns < 0 ? stddev(returns, 20) : close), 2.0)) - 0.5
    Normalizado para o intervalo [-0.5, +0.5].
    """
    close = df["close"]
    returns = close.pct_change(fill_method=None).fillna(0.0)
    std_ret = returns.rolling(window=window, min_periods=5).std().fillna(0.0)

    cond = returns < 0.0
    x = np.where(cond, std_ret, close)
    signed_pow = np.sign(x) * (np.abs(x) ** 2.0)

    s = pd.Series(signed_pow, index=df.index)
    roll_rank = s.rolling(window=window, min_periods=5).apply(
        lambda w: (pd.Series(w).rank(pct=True).iloc[-1] - 0.5) if len(w) > 0 else 0.0,
        raw=False,
    ).fillna(0.0)
    return roll_rank


def calc_alpha101_012(df: pd.DataFrame) -> pd.Series:
    """Alpha #12 de Kakushadze: Sinal de aceleração de volume e retorno de preço.

    Fórmula: sign(delta(volume, 1)) * (-1 * delta(close, 1)) normalizado.
    Mede a absorção passiva institucional contra a direção do candle.
    """
    delta_vol = df["volume"].diff().fillna(0.0)
    delta_close = df["close"].diff().fillna(0.0)
    alpha = np.sign(delta_vol) * (-1.0 * delta_close)
    amp = np.maximum(df["high"] - df["low"], _EPS)
    return (alpha / amp).clip(-2.0, 2.0)


def calc_qlib_norm_return(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """Fator Qlib de Retorno Normalizado por Volatilidade (Microsoft Qlib Alpha158).

    Fórmula: (close / close[t-window] - 1.0) / (stddev(close, window) / close + EPS)
    Identifica velocidade de tendência ajustada à volatilidade recente.
    """
    close = df["close"]
    lag_close = close.shift(window).fillna(close.iloc[0])
    ret = (close - lag_close) / np.maximum(lag_close, _EPS)
    roll_std = close.rolling(window=window, min_periods=3).std().fillna(0.0)
    vol_rel = roll_std / np.maximum(close, _EPS)
    return (ret / np.maximum(vol_rel, _EPS)).clip(-3.0, 3.0)


def extract_vibe_quant_features(df: pd.DataFrame) -> Dict[str, float]:
    """Extrai os 3 fatores alfa do Vibe-Trading da última vela para filtros de confirmação."""
    if len(df) < 25:
        return {"vibe_alpha101_01": 0.0, "vibe_alpha101_12": 0.0, "vibe_qlib_norm_ret": 0.0}

    a1 = float(calc_alpha101_001(df).iloc[-1])
    a12 = float(calc_alpha101_012(df).iloc[-1])
    qlib_ret = float(calc_qlib_norm_return(df).iloc[-1])

    return {
        "vibe_alpha101_01": round(a1, 4),
        "vibe_alpha101_12": round(a12, 4),
        "vibe_qlib_norm_ret": round(qlib_ret, 4),
    }
