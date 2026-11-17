"""Módulo de Comparação Concorrente Multi-Timeframe (MTF).

Compara:
- Cenário 1: Operação Independente Simultânea (M5, M15, M30, H1)
- Cenário 2: Confluência em Cascata (Macro H1/M30 -> Gatilho M5/M15)
"""

from typing import Dict, List, Optional
import pandas as pd
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from backtest_data import Metricas, Trade, calcular_metricas
from backtest_mtf_data import MTFPairData, load_all_mtf
from ema_wave import EMAWaveIndicator
from features import FeatureEngine

console = Console()
PAYOUT_PADRAO = 0.85
STAKE_PADRAO = 10.0


def extrair_sinal_wave(row: pd.Series) -> Optional[str]:
    """Extrai sinal direcional com filtros técnicos (ADX e Bollinger)."""
    angle = float(row["wave_angle"])
    c_open, c_high = float(row["open"]), float(row["high"])
    c_low, c_close = float(row["low"]), float(row["close"])
    rng = max(c_high - c_low, 1e-6)
    body = abs(c_close - c_open)
    if (body / rng) < 0.08:
        return None

    direcao = None
    if angle >= 10.0:
        l_shadow = min(c_open, c_close) - c_low
        rejeitou = (l_shadow / rng) >= 0.20
        if c_low <= float(row["wave_high"]) and c_close >= float(row["wave_low"]) and (c_close > c_open or rejeitou):
            direcao = "CALL"
    elif angle <= -10.0:
        u_shadow = c_high - max(c_open, c_close)
        rejeitou = (u_shadow / rng) >= 0.20
        if c_high >= float(row["wave_low"]) and c_close <= float(row["wave_high"]) and (c_close < c_open or rejeitou):
            direcao = "PUT"

    if not direcao or float(row.get("feat_adx", 25.0)) < 20.0:
        return None
    pct_b = float(row.get("feat_bollinger_pct_b", 0.5))
    if (direcao == "CALL" and pct_b > 1.0) or (direcao == "PUT" and pct_b < 0.0):
        return None
    return direcao


def preparar_dados_par(pdata: MTFPairData, engine: FeatureEngine, wave: EMAWaveIndicator) -> dict:
    """Calcula e funde variáveis de cada timeframe para um par."""
    cols = ["datetime", "wave_high", "wave_close", "wave_low", "wave_angle"]
    m5 = pd.merge(engine.calculate_features(pdata.df_m5, include_raw=True), wave.calculate(pdata.df_m5)[cols], on="datetime")
    m15 = pd.merge(engine.calculate_features(pdata.df_m15, include_raw=True), wave.calculate(pdata.df_m15)[cols], on="datetime").set_index("datetime")
    m30 = pd.merge(engine.calculate_features(pdata.df_m30, include_raw=True), wave.calculate(pdata.df_m30)[cols], on="datetime").set_index("datetime")
    h1 = pd.merge(engine.calculate_features(pdata.df_h1, include_raw=True), wave.calculate(pdata.df_h1)[cols], on="datetime").set_index("datetime")

    split = int(len(m5) * 0.60)
    m5_test = m5.iloc[split:].copy().reset_index(drop=True)
    return {"m5": m5_test, "m15": m15, "m30": m30, "h1": h1}


def simular_mtf(mtf_dict: Dict[str, MTFPairData], modo_cascata: bool = False) -> List[Trade]:
    """Executa simulação passo a passo no período Out-of-Sample."""
    trades: List[Trade] = []
    engine = FeatureEngine()
    wave = EMAWaveIndicator()

    dados_preparados = {p: preparar_dados_par(pdata, engine, wave) for p, pdata in mtf_dict.items()}
    ordens_ativas: Dict[str, int] = {}
    n_passos = min(len(d["m5"]) for d in dados_preparados.values()) - 15

    for t_step in range(10, n_passos):
        for p in list(ordens_ativas.keys()):
            ordens_ativas[p] -= 1
            if ordens_ativas[p] <= 0:
                ordens_ativas.pop(p, None)

        for par, d in dados_preparados.items():
            if par in ordens_ativas:
                continue

            row_m5 = d["m5"].iloc[t_step]
            dt = row_m5["datetime"]
            dt_min = dt.minute

            if modo_cascata:
                if dt.minute not in (0, 15, 30, 45) or dt not in d["m15"].index:
                    continue

                h1_pass = d["h1"].loc[:dt]
                if len(h1_pass) < 2:
                    continue

                ang_h1 = float(h1_pass.iloc[-1]["wave_angle"])
                macro_vies = "CALL" if ang_h1 >= 5.0 else ("PUT" if ang_h1 <= -5.0 else None)
                if not macro_vies:
                    continue

                sig15 = extrair_sinal_wave(d["m15"].loc[dt])
                if sig15 != macro_vies:
                    continue

                dur = 3
                nxt = float(d["m5"].iloc[t_step + dur]["close"])
                cur = float(row_m5["close"])
                win = (nxt > cur) if sig15 == "CALL" else (nxt < cur)
                pnl = (STAKE_PADRAO * PAYOUT_PADRAO) if win else -STAKE_PADRAO
                trades.append(Trade(par, str(dt), sig15, pnl, win, 0.55))
                ordens_ativas[par] = dur

            else:  # Modo Independente
                sig, dur = None, 1
                if dt in d["h1"].index and dt_min == 0:
                    s = extrair_sinal_wave(d["h1"].loc[dt])
                    if s: sig, dur = s, 12
                if not sig and dt in d["m30"].index and dt_min in (0, 30):
                    s = extrair_sinal_wave(d["m30"].loc[dt])
                    if s: sig, dur = s, 6
                if not sig and dt in d["m15"].index and dt_min in (0, 15, 30, 45):
                    s = extrair_sinal_wave(d["m15"].loc[dt])
                    if s: sig, dur = s, 3
                if not sig:
                    s = extrair_sinal_wave(row_m5)
                    if s: sig, dur = s, 1

                if sig:
                    nxt = float(d["m5"].iloc[t_step + dur]["close"])
                    cur = float(row_m5["close"])
                    win = (nxt > cur) if sig == "CALL" else (nxt < cur)
                    pnl = (STAKE_PADRAO * PAYOUT_PADRAO) if win else -STAKE_PADRAO
                    trades.append(Trade(par, str(dt), sig, pnl, win, 0.55))
                    ordens_ativas[par] = dur

    return trades


def exibir_tabela_comparativa(m1: Metricas, m2: Metricas) -> None:
    """Exibe tabela comparativa detalhada entre os dois modelos MTF."""
    table = Table(
        title="⚔️ DISPUTA DE DESEMPENHO MULTI-TIMEFRAME (M5, M15, M30, H1 — 90 DIAS)",
        box=ROUNDED,
        title_style="bold magenta",
    )
    table.add_column("Métrica de Avaliação", justify="left", style="bold white")
    table.add_column("Opção 1: Independente (M5+M15+M30+H1)", justify="center", style="cyan")
    table.add_column("Opção 2: Cascata MTF (Macro H1/M30 -> Gatilho)", justify="center", style="green")

    diff_wr = m2.win_rate - m1.win_rate
    diff_pnl = m2.pnl_total - m1.pnl_total
    cor_wr = "green" if diff_wr >= 0 else "yellow"
    cor_pnl = "green" if diff_pnl >= 0 else "yellow"

    table.add_row("Total de Operações", f"{m1.total_trades}", f"{m2.total_trades}")
    table.add_row("Vitórias / Derrotas", f"{m1.wins}W / {m1.losses}L", f"{m2.wins}W / {m2.losses}L")
    table.add_row("Taxa de Acerto (Win Rate %)", f"{m1.win_rate:.2f}%", f"[{cor_wr}]{m2.win_rate:.2f}% ({diff_wr:+.2f}%)[/{cor_wr}]")
    table.add_row("Ponto de Equilíbrio (Breakeven)", "54.05%", "54.05%")
    table.add_row("PnL Líquido (Stake $10)", f"${m1.pnl_total:+.2f}", f"[{cor_pnl}]${m2.pnl_total:+.2f} ({diff_pnl:+.2f})[/{cor_pnl}]")
    table.add_row("Profit Factor", f"{m1.profit_factor:.2f}", f"{m2.profit_factor:.2f}")
    table.add_row("Drawdown Máximo", f"{m1.max_drawdown_pct:.1f}%", f"{m2.max_drawdown_pct:.1f}%")
    table.add_row("Maior Sequência de Wins", f"{m1.max_win_streak}", f"{m2.max_win_streak}")
    table.add_row("Maior Sequência de Losses", f"{m1.max_loss_streak}", f"{m2.max_loss_streak}")
    table.add_row("Expectativa / Trade", f"${m1.expectativa_matematica:+.2f}", f"${m2.expectativa_matematica:+.2f}")

    console.print()
    console.print(table)
    console.print()


def executar():
    console.print("🚀 [bold green]Iniciando Simulação Multi-Timeframe nos 7 Pares (M5, M15, M30, H1)...[/bold green]")
    mtf = load_all_mtf()

    console.print("⏱️ [cyan]Simulando Opção 1: Operação Independente Simultânea (M5/M15/M30/H1)...[/cyan]")
    trades_op1 = simular_mtf(mtf, modo_cascata=False)
    m1 = calcular_metricas("Opção 1: Independente", trades_op1)

    console.print("🎯 [green]Simulando Opção 2: Confluência em Cascata (Macro H1/M30 -> Gatilho)...[/green]")
    trades_op2 = simular_mtf(mtf, modo_cascata=True)
    m2 = calcular_metricas("Opção 2: Cascata MTF", trades_op2)

    exibir_tabela_comparativa(m1, m2)


if __name__ == "__main__":
    executar()
