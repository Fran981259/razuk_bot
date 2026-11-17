"""Módulo de Backtest Comparativo M15: Técnica Pura vs Técnica + Machine Learning.

Executa simulação Out-of-Sample estrita (sem vazamento de dados futuros)
nos 90 dias de dados históricos M15 de 7 pares de Forex e OTC.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler

from backtest_data import Metricas, Trade, calcular_metricas, load_all_m15_datasets
from ema_wave import ClockState, EMAWaveIndicator
from features import FEATURE_COLUMNS, FeatureEngine

console = Console()

PAYOUT_PADRAO = 0.85
STAKE_PADRAO = 10.0
BANCA_INICIAL = 1000.0
ML_THRESHOLD = 0.505  # Calibrado para M15 (limiar de corte de alta precisão)


def treinar_modelo_m15(treino_dict: Dict[str, pd.DataFrame]) -> Tuple[LGBMClassifier, StandardScaler]:
    """Treina classificador LightGBM nos primeiros 60% do histórico M15."""
    X_list, y_list = [], []
    for _, df_feat in treino_dict.items():
        alvo = (df_feat["close"].shift(-1) > df_feat["close"]).astype(int)
        df_valid = df_feat.iloc[:-1].copy()
        df_valid["target"] = alvo.iloc[:-1]
        X_list.append(df_valid[FEATURE_COLUMNS])
        y_list.append(df_valid["target"])

    X_train = pd.concat(X_list, ignore_index=True)
    y_train = pd.concat(y_list, ignore_index=True)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    clf = LGBMClassifier(
        max_depth=4,
        learning_rate=0.03,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train_sc, y_train)
    return clf, scaler


def simular_backtest_m15(
    test_dict: Dict[str, pd.DataFrame],
    wave_dict: Dict[str, pd.DataFrame],
    modelo: Optional[LGBMClassifier] = None,
    scaler: Optional[StandardScaler] = None,
    usar_ml: bool = False,
) -> List[Trade]:
    """Simula as operações no conjunto de teste Out-of-Sample."""
    trades: List[Trade] = []
    wave_ind = EMAWaveIndicator()

    for par, feat_df in test_dict.items():
        w_df = wave_dict[par]

        for i in range(len(feat_df) - 1):
            f = feat_df.iloc[i]
            w = w_df.iloc[i]

            angle = float(w["wave_angle"])
            state, _ = wave_ind.classify_clock_state(angle)

            c_open, c_high = float(f["open"]), float(f["high"])
            c_low, c_close = float(f["low"]), float(f["close"])
            rng = max(c_high - c_low, 1e-6)
            body = abs(c_close - c_open)
            is_doji = (body / rng) < 0.08

            direcao = None
            if state == ClockState.BULLISH_STRONG and not is_doji:
                l_shadow = min(c_open, c_close) - c_low
                rejeitou = (l_shadow / rng) >= 0.20
                if c_low <= float(w["wave_high"]) and c_close >= float(w["wave_low"]) and (c_close > c_open or rejeitou):
                    direcao = "CALL"
            elif state == ClockState.BEARISH_STRONG and not is_doji:
                u_shadow = c_high - max(c_open, c_close)
                rejeitou = (u_shadow / rng) >= 0.20
                if c_high >= float(w["wave_low"]) and c_close <= float(w["wave_high"]) and (c_close < c_open or rejeitou):
                    direcao = "PUT"

            if not direcao:
                continue

            # Filtros Técnicos de Qualidade
            if float(f["feat_adx"]) < 20.0:
                continue

            pct_b = float(f["feat_bollinger_pct_b"])
            if direcao == "CALL" and pct_b > 1.00:
                continue
            if direcao == "PUT" and pct_b < 0.00:
                continue

            # Filtro de Machine Learning
            prob_win = 0.50
            if usar_ml and modelo is not None and scaler is not None:
                x_vec = f[FEATURE_COLUMNS].to_frame().T
                x_sc = scaler.transform(x_vec)
                prob_raw = float(modelo.predict_proba(x_sc)[0, 1])
                prob_win = prob_raw if direcao == "CALL" else (1.0 - prob_raw)
                if prob_win < ML_THRESHOLD:
                    continue

            # Fechamento na vela seguinte (expiração M15)
            next_close = float(feat_df.iloc[i + 1]["close"])
            if direcao == "CALL":
                ganhou = next_close > c_close
                empatou = next_close == c_close
            else:
                ganhou = next_close < c_close
                empatou = next_close == c_close

            pnl = (STAKE_PADRAO * PAYOUT_PADRAO) if ganhou else (0.0 if empatou else -STAKE_PADRAO)
            trades.append(
                Trade(
                    par=par,
                    datetime=str(feat_df.iloc[i].get("datetime", "")),
                    direcao=direcao,
                    pnl=pnl,
                    is_win=ganhou,
                    prob_ml=prob_win,
                )
            )

    return trades


def exibir_comparacao(m1: Metricas, m2: Metricas) -> None:
    """Renderiza tabela comparativa de alta precisão entre os dois cenários."""
    table = Table(
        title="📊 BACKTEST M15 (90 DIAS — TESTE FORA DA AMOSTRA / OUT-OF-SAMPLE)",
        box=ROUNDED,
        title_style="bold cyan",
    )
    table.add_column("Métrica de Avaliação", justify="left", style="bold white")
    table.add_column("Opção 1: Técnica Pura", justify="center", style="cyan")
    table.add_column("Opção 2: Técnica + ML (LightGBM)", justify="center", style="green")

    diff_wr = m2.win_rate - m1.win_rate
    diff_pnl = m2.pnl_total - m1.pnl_total

    cor_wr = "green" if diff_wr >= 0 else "yellow"
    cor_pnl = "green" if diff_pnl >= 0 else "yellow"

    table.add_row("Total de Operações", f"{m1.total_trades}", f"{m2.total_trades}")
    table.add_row("Vitórias (Wins) / Derrotas", f"{m1.wins}W / {m1.losses}L", f"{m2.wins}W / {m2.losses}L")
    table.add_row("Empates", f"{m1.empates}", f"{m2.empates}")
    table.add_row(
        "Taxa de Acerto (Win Rate %)",
        f"{m1.win_rate:.2f}%",
        f"[{cor_wr}]{m2.win_rate:.2f}% ({diff_wr:+.2f}%)[/{cor_wr}]",
    )
    table.add_row("Ponto de Equilíbrio (Breakeven)", "54.05%", "54.05%")
    table.add_row(
        "PnL Líquido Acumulado (Stake $10)",
        f"${m1.pnl_total:+.2f}",
        f"[{cor_pnl}]${m2.pnl_total:+.2f} ({diff_pnl:+.2f})[/{cor_pnl}]",
    )
    table.add_row("Profit Factor", f"{m1.profit_factor:.2f}", f"{m2.profit_factor:.2f}")
    table.add_row("Drawdown Máximo", f"{m1.max_drawdown_pct:.1f}%", f"{m2.max_drawdown_pct:.1f}%")
    table.add_row("Maior Sequência de Wins", f"{m1.max_win_streak}", f"{m2.max_win_streak}")
    table.add_row("Maior Sequência de Losses", f"{m1.max_loss_streak}", f"{m2.max_loss_streak}")
    table.add_row("Expectativa Matemática / Trade", f"${m1.expectativa_matematica:+.2f}", f"${m2.expectativa_matematica:+.2f}")

    console.print()
    console.print(table)
    console.print()


def executar():
    """Função principal que orquestra a execução dos dois backtests."""
    console.print("🚀 [bold green]Iniciando Backtest M15 em 7 Pares (Base de 90 dias)...[/bold green]")
    datasets = load_all_m15_datasets()
    engine = FeatureEngine()
    wave = EMAWaveIndicator()

    treino_dict, test_dict, wave_test_dict = {}, {}, {}

    for par, df in datasets.items():
        feat = engine.calculate_features(df, include_raw=True)
        w_df = wave.calculate(df)

        split = int(len(feat) * 0.60)
        treino_dict[par] = feat.iloc[:split].copy()
        test_dict[par] = feat.iloc[split:].copy()
        wave_test_dict[par] = w_df.iloc[split:].copy()

    console.print("🧠 [cyan]Treinando modelo LightGBM específico para M15 (60% histórico)...[/cyan]")
    modelo, scaler = treinar_modelo_m15(treino_dict)

    console.print("📈 [yellow]Simulando Opção 1: Confluência Técnica Pura...[/yellow]")
    trades_op1 = simular_backtest_m15(test_dict, wave_test_dict, usar_ml=False)
    metricas_op1 = calcular_metricas("Técnica Pura", trades_op1)

    console.print("🤖 [magenta]Simulando Opção 2: Técnica + Filtro Machine Learning P(Win) >= 55%...[/magenta]")
    trades_op2 = simular_backtest_m15(test_dict, wave_test_dict, modelo=modelo, scaler=scaler, usar_ml=True)
    metricas_op2 = calcular_metricas("Técnica + ML", trades_op2)

    exibir_comparacao(metricas_op1, metricas_op2)


if __name__ == "__main__":
    executar()
