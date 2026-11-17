"""Módulo de processamento e reamostragem de dados históricos para M15.

Carrega as velas M5 de 90 dias existentes em data/ e realiza
resample estrito para M15, garantindo 3 velas M5 por período.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

PARES_DISPONIVEIS = [
    "AUDUSD",
    "AUDUSD-OTC",
    "EURJPY",
    "EURJPY-OTC",
    "EURUSD",
    "EURUSD-OTC",
    "USDCHF-OTC",
]


def resample_m5_to_m15(df_m5: pd.DataFrame) -> pd.DataFrame:
    """Reamostra DataFrame M5 para M15 com alinhamento estrito.
    
    Exige exatamente 3 velas de M5 por período de 15 minutos para
    evitar distorções de início/fim de série.
    """
    df = df_m5.copy()
    if "datetime" not in df.columns:
        if "from" in df.columns:
            df["datetime"] = pd.to_datetime(df["from"], unit="s")
        else:
            raise ValueError("DataFrame sem coluna de tempo ('datetime' ou 'from').")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])

    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)

    # Agregação OHLCV clássica com contagem de integridade
    m15 = df.resample("15min", closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "from": "count",
    })

    # Filtrar apenas períodos com 3 velas completas de 5 min
    m15 = m15[m15["from"] == 3].drop(columns=["from"]).dropna().reset_index()
    return m15


def load_m15_dataset(par: str) -> Optional[pd.DataFrame]:
    """Carrega dados M5 de um par específico e converte para M15."""
    nome_arq = f"candles_{par.replace('-', '_')}_M5_90d.csv"
    caminho = DATA_DIR / nome_arq
    if not caminho.exists():
        return None

    df_m5 = pd.read_csv(caminho)
    return resample_m5_to_m15(df_m5)


def load_all_m15_datasets() -> Dict[str, pd.DataFrame]:
    """Carrega todos os pares M5 disponíveis e converte para M15."""
    datasets: Dict[str, pd.DataFrame] = {}
    for par in PARES_DISPONIVEIS:
        df_m15 = load_m15_dataset(par)
        if df_m15 is not None and len(df_m15) >= 50:
            datasets[par] = df_m15
    return datasets


@dataclass
class Trade:
    par: str
    datetime: str
    direcao: str
    pnl: float
    is_win: bool
    prob_ml: float


@dataclass
class Metricas:
    nome: str
    total_trades: int
    wins: int
    losses: int
    empates: int
    win_rate: float
    pnl_total: float
    profit_factor: float
    max_drawdown_pct: float
    max_win_streak: int
    max_loss_streak: int
    expectativa_matematica: float


def calcular_metricas(nome: str, trades: List[Trade], banca_inicial: float = 1000.0) -> Metricas:
    """Calcula estatísticas de desempenho, risco e expectativa matemática."""
    if not trades:
        return Metricas(nome, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    wins = sum(1 for t in trades if t.is_win and t.pnl > 0)
    losses = sum(1 for t in trades if not t.is_win and t.pnl < 0)
    empates = sum(1 for t in trades if t.pnl == 0)
    total_decididos = wins + losses
    win_rate = (wins / total_decididos * 100.0) if total_decididos > 0 else 0.0

    pnl_total = sum(t.pnl for t in trades)
    ganhos = sum(t.pnl for t in trades if t.pnl > 0)
    perdas = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = (ganhos / perdas) if perdas > 0 else 99.0

    banca = banca_inicial
    pico = banca_inicial
    max_dd_pct = 0.0
    win_streak, max_win_streak = 0, 0
    loss_streak, max_loss_streak = 0, 0

    for t in trades:
        banca += t.pnl
        if banca > pico:
            pico = banca
        dd = (pico - banca) / pico * 100.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        if t.is_win:
            win_streak += 1
            loss_streak = 0
            if win_streak > max_win_streak:
                max_win_streak = win_streak
        elif t.pnl < 0:
            loss_streak += 1
            win_streak = 0
            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak

    exp_mat = (pnl_total / len(trades)) if trades else 0.0

    return Metricas(
        nome=nome,
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        empates=empates,
        win_rate=win_rate,
        pnl_total=pnl_total,
        profit_factor=pf,
        max_drawdown_pct=max_dd_pct,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        expectativa_matematica=exp_mat,
    )

