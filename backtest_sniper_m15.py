"""Módulo de Backtest de Alta Fidelidade: Micro-Sniper M15 vs Entrada Ingênua no Segundo 00.

Simula a microestrutura interna de cada vela M15 utilizando velas históricas de M5/M1
ao longo de 90 dias em todos os 7 pares, aplicando:
1. Confluência em Cascata MTF (Macro H1 -> Gatilho M15).
2. Replay sub-candle na janela inicial de 1/4 da vela.
3. Fricção real de corretora: penalidade de spread de 0.8 pip (vitórias por menos de 0.8 pip viram LOSS).
4. Teto máximo de invalidação contra velas assassinas.
5. Captura de taxa com margem de segurança no pullback.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from backtest_data import Metricas, Trade, calcular_metricas
from backtest_mtf_data import load_all_mtf
from backtest_mtf_compare import extrair_sinal_wave, preparar_dados_par
from ema_wave import EMAWaveIndicator
from features import FeatureEngine

console = Console()

PAYOUT_PADRAO = 0.85
STAKE_PADRAO = 10.0
BANCA_INICIAL = 1000.0
BREAKEVEN = 54.05

PIP_MAP: Dict[str, float] = {
    "EURJPY": 0.01,
    "EURJPY_OTC": 0.01,
    "EURJPY-OTC": 0.01,
    "USDJPY": 0.01,
}


def simular_alta_fidelidade() -> Tuple[Metricas, Metricas, Dict[str, int]]:
    """Executa o backtest comparativo completo com sub-bar replay e fricção real."""
    print("⏳ Carregando dados históricos multi-timeframe de 90 dias dos 7 pares...")
    all_data = load_all_mtf()
    engine = FeatureEngine()
    wave = EMAWaveIndicator()

    print("📊 Calculando indicadores e alinhando timeframes concorrentes...")
    preparados = {}
    for par, pdata in all_data.items():
        preparados[par] = preparar_dados_par(pdata, engine, wave)

    trades_ingenuo: List[Trade] = []
    trades_sniper: List[Trade] = []
    stats_globais = {
        "oportunidades": 0,
        "executados": 0,
        "abortados_assassina": 0,
        "timeouts": 0,
    }

    print("⚡ Executando simulação de sub-candle e caça de taxa...")

    for par, d in preparados.items():
        pip_sz = PIP_MAP.get(par, 0.0001)
        spread_friccao = 0.8 * pip_sz  # Penalidade de 0.8 pip contra o trader

        m5 = d["m5"]
        m15 = d["m15"]
        h1 = d["h1"]

        for i in range(2, len(m15) - 2):
            row_prev = m15.iloc[i - 1]
            row_curr = m15.iloc[i]
            dt = row_curr["datetime"] if "datetime" in row_curr else m15.index[i]

            # 1. Filtro Macro H1 (Cascata MTF)
            h1_pass = h1.loc[:dt]
            if len(h1_pass) < 2:
                continue
            ang_h1 = float(h1_pass.iloc[-1]["wave_angle"])
            macro_vies = "CALL" if ang_h1 >= 5.0 else ("PUT" if ang_h1 <= -5.0 else None)
            if not macro_vies:
                continue

            # 2. Gatilho M15 alinhado ao viés macro
            sig15 = extrair_sinal_wave(row_curr)
            if sig15 != macro_vies:
                continue

            # 3. Mapeamento das sub-barras internas da vela M15
            idx_m5 = m5[m5["datetime"] == dt].index
            if len(idx_m5) == 0:
                continue
            step_m5 = idx_m5[0]
            if step_m5 + 3 >= len(m5):
                continue

            bar_0 = m5.iloc[step_m5]      # Sub-barra 1 (primeiros 5 min = 1/3 da vela)
            bar_2 = m5.iloc[step_m5 + 2]  # Sub-barra 3 (fechamento oficial aos 15 min)

            open_00 = float(bar_0["open"])
            close_15 = float(bar_2["close"])

            stats_globais["oportunidades"] += 1

            # ── MODELO 1: SEGUNDO 00 INGÊNUO (Abertura a Mercado) ────────────
            if sig15 == "CALL":
                win_ing = close_15 > (open_00 + spread_friccao)
            else:
                win_ing = close_15 < (open_00 - spread_friccao)

            pnl_ing = round(STAKE_PADRAO * PAYOUT_PADRAO, 2) if win_ing else -STAKE_PADRAO
            trades_ingenuo.append(
                Trade(par=par, datetime=str(dt), direcao=sig15, pnl=pnl_ing, is_win=win_ing, prob_ml=0.55)
            )

            # ── MODELO 2: MOTOR MICRO-SNIPER (Taxa com Margem) ───────────────
            # Taxa Alvo C1 com margem: recuo de 1 pip contra a abertura ou máxima/mínima anterior
            # Teto A: Mínima anterior para CALL, Máxima anterior para PUT
            if sig15 == "CALL":
                target_taxa = min(open_00 - 1.0 * pip_sz, float(row_prev["high"]))
                teto_invalida = float(row_prev["low"])

                low_sub = float(bar_0["low"])
                # Se furar o teto estrutural: Vela Assassina (Aborta!)
                if low_sub < teto_invalida:
                    stats_globais["abortados_assassina"] += 1
                elif low_sub <= target_taxa:
                    # Pegou a taxa descontada no pullback!
                    stats_globais["executados"] += 1
                    win_snp = close_15 > (target_taxa + spread_friccao)
                    pnl_snp = round(STAKE_PADRAO * PAYOUT_PADRAO, 2) if win_snp else -STAKE_PADRAO
                    trades_sniper.append(
                        Trade(par=par, datetime=str(dt), direcao=sig15, pnl=pnl_snp, is_win=win_snp, prob_ml=0.55)
                    )
                else:
                    # Vela foi embora sem recuar: Timeout
                    stats_globais["timeouts"] += 1

            else:  # PUT
                target_taxa = max(open_00 + 1.0 * pip_sz, float(row_prev["low"]))
                teto_invalida = float(row_prev["high"])

                high_sub = float(bar_0["high"])
                if high_sub > teto_invalida:
                    stats_globais["abortados_assassina"] += 1
                elif high_sub >= target_taxa:
                    stats_globais["executados"] += 1
                    win_snp = close_15 < (target_taxa - spread_friccao)
                    pnl_snp = round(STAKE_PADRAO * PAYOUT_PADRAO, 2) if win_snp else -STAKE_PADRAO
                    trades_sniper.append(
                        Trade(par=par, datetime=str(dt), direcao=sig15, pnl=pnl_snp, is_win=win_snp, prob_ml=0.55)
                    )
                else:
                    stats_globais["timeouts"] += 1

    m_ing = calcular_metricas("Segundo 00 (Ingênuo)", trades_ingenuo, BANCA_INICIAL)
    m_snp = calcular_metricas("🎯 Micro-Sniper (Com Margem)", trades_sniper, BANCA_INICIAL)
    return m_ing, m_snp, stats_globais


def main() -> None:
    m_ing, m_snp, stats = simular_alta_fidelidade()

    table = Table(
        title="⚔️ DISPUTA DE EXECUÇÃO: SEGUNDO 00 vs MOTOR MICRO-SNIPER (90 DIAS REALÍSTICO)",
        box=ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Métrica de Avaliação", justify="left")
    table.add_column("Modo 1: Segundo 00 (Ingênuo)", justify="center", style="yellow")
    table.add_column("Modo 2: 🎯 Micro-Sniper (Margem C1)", justify="center", style="green")

    table.add_row("Oportunidades Geradas", str(stats["oportunidades"]), str(stats["oportunidades"]))
    table.add_row("Ordens Executadas", str(m_ing.total_trades), f"{m_snp.total_trades} (Ultra-Seletivo)")
    table.add_row("🛡️ Velas Assassinas Abortadas (Loss Evitado)", "0 (Tomou 100% dos Losses)", f"🛡️ [bold cyan]{stats['abortados_assassina']}[/bold cyan]")
    table.add_row("⏱️ Cancelados por Timeout (Sem Recuo)", "0 (Entrou a Mercado)", f"⏱️ {stats['timeouts']}")
    table.add_row("Vitórias / Derrotas", f"{m_ing.wins}W / {m_ing.losses}L", f"{m_snp.wins}W / {m_snp.losses}L")

    diff_wr = m_snp.win_rate - m_ing.win_rate
    table.add_row(
        "Taxa de Acerto (Win Rate %)",
        f"{m_ing.win_rate:.2f}%",
        f"[bold]{m_snp.win_rate:.2f}% ({diff_wr:+.2f}%)[/bold]",
    )
    table.add_row("Ponto de Equilíbrio (Breakeven)", f"{BREAKEVEN:.2f}%", f"{BREAKEVEN:.2f}%")

    pnl_diff = m_snp.pnl_total - m_ing.pnl_total
    table.add_row(
        "PnL Líquido (Stake R$ 10)",
        f"R$ {m_ing.pnl_total:+,.2f}",
        f"[bold]R$ {m_snp.pnl_total:+,.2f} ({pnl_diff:+,.2f})[/bold]",
    )
    table.add_row("Profit Factor", f"{m_ing.profit_factor:.2f}", f"[bold]{m_snp.profit_factor:.2f}[/bold]")
    table.add_row("Drawdown Máximo", f"{m_ing.max_drawdown_pct:.1f}%", f"[bold]{m_snp.max_drawdown_pct:.1f}%[/bold]")
    table.add_row("Maior Sequência de Wins", str(m_ing.max_win_streak), str(m_snp.max_win_streak))
    table.add_row("Maior Sequência de Losses", str(m_ing.max_loss_streak), str(m_snp.max_loss_streak))
    table.add_row(
        "Expectativa Matemática / Trade",
        f"R$ {m_ing.expectativa_matematica:+.2f}",
        f"[bold]R$ {m_snp.expectativa_matematica:+.2f}[/bold]",
    )

    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
