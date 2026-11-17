"""Módulo de dados Multi-Timeframe (M5, M15, M30, H1).

Carrega e reamostra as séries temporais dos 7 pares de Forex e OTC
de forma perfeitamente sincronizada no tempo para testes concorrentes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from backtest_data import DATA_DIR, PARES_DISPONIVEIS
from ema_wave import EMAWaveIndicator
from features import FeatureEngine


@dataclass
class MTFPairData:
    par: str
    df_m5: pd.DataFrame
    df_m15: pd.DataFrame
    df_m30: pd.DataFrame
    df_h1: pd.DataFrame


def resample_series(df_m5_indexed: pd.DataFrame, rule: str, min_bars: int) -> pd.DataFrame:
    """Reamostra a partir de M5 indexado por datetime exigindo barras completas."""
    res = df_m5_indexed.resample(rule, closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "from": "count",
    })
    res = res[res["from"] == min_bars].drop(columns=["from"]).dropna().reset_index()
    return res


def load_mtf_pair(par: str) -> Optional[MTFPairData]:
    """Carrega um par e gera as 4 séries temporais sincronizadas."""
    nome_arq = f"candles_{par.replace('-', '_')}_M5_90d.csv"
    caminho = DATA_DIR / nome_arq
    if not caminho.exists():
        return None

    df_raw = pd.read_csv(caminho)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw.sort_values("datetime", inplace=True)
    df_indexed = df_raw.set_index("datetime")

    df_m5 = df_raw.copy().reset_index(drop=True)
    df_m15 = resample_series(df_indexed, "15min", 3)
    df_m30 = resample_series(df_indexed, "30min", 6)
    df_h1 = resample_series(df_indexed, "1h", 12)

    return MTFPairData(
        par=par,
        df_m5=df_m5,
        df_m15=df_m15,
        df_m30=df_m30,
        df_h1=df_h1,
    )


def load_all_mtf() -> Dict[str, MTFPairData]:
    """Carrega dados multi-timeframe de todos os pares disponíveis."""
    mtf_dict: Dict[str, MTFPairData] = {}
    for par in PARES_DISPONIVEIS:
        data = load_mtf_pair(par)
        if data is not None and len(data.df_h1) >= 100:
            mtf_dict[par] = data
    return mtf_dict
