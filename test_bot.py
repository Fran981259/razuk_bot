"""Bateria de testes abrangente de ponta a ponta para o bot de trading.

Valida conexão real com a IQ Option (conta Practice), integridade de dados OHLCV,
cálculo das 16 features quantitativas, inferência preditiva LightGBM e
dimensionamento de risco via Critério de Kelly Fracionário.
"""

import asyncio
from datetime import datetime, timezone
import logging
import os
import time

os.environ.setdefault("HEADLESS", "1")

import pandas as pd

from config import get_config
from conector import IqOptionClient
from features import FeatureEngine
from iq_constants import get_active_id
from main import BotOrchestrator
from ml_core import MLCore
from risk import RiskManager


async def run_test_suite() -> None:
    """Executa todas as etapas da bateria de testes com dados reais de mercado."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("test_suite")
    logger.info("==================================================")
    logger.info("  🚀 INICIANDO BATERIA DE TESTES DO BOT DE TRADING")
    logger.info("==================================================")

    # 1. TESTE DE CONFIGURAÇÃO
    logger.info("[TESTE 1/7] Validando configurações e leitura do .env...")
    config = get_config()
    assert config.conexao.email, "Email não configurado no .env"
    assert config.conexao.password, "Senha não configurada no .env"
    assert config.conexao.account_type == "PRACTICE", "Conta deve ser PRACTICE"
    logger.info("  ✅ Configuração validada: Conta %s (%s)", config.conexao.account_type, config.conexao.email)

    # 2. TESTE DE CONEXÃO WEBSOCKET REAL
    logger.info("[TESTE 2/7] Conectando à IQ Option via WebSocket oficial...")
    client = IqOptionClient(config.conexao)
    conectado = await client.connect()
    assert conectado and client.is_connected, "Falha na conexão WebSocket"
    saldo_real = await client.get_balance()
    assert saldo_real > 0.0, "Saldo da conta de prática deve ser maior que zero"
    logger.info("  ✅ Conectado com sucesso! Saldo real na conta Practice: $%.2f USD", saldo_real)

    # 3. TESTE DE COLETA DE VELAS REAIS (EURUSD e EURUSD-OTC)
    logger.info("[TESTE 3/7] Coletando velas OHLCV reais de mercado...")
    for par in ["EURUSD", "EURUSD-OTC"]:
        active_id = get_active_id(par)
        velas = await client.get_candles(par=par, timeframe="M1", contagem=100)
        assert len(velas) >= 85, f"Quantidade insuficiente de velas para {par} ({len(velas)})"
        c = velas[-1]
        logger.info(
            "  ✅ %s (Active ID: %d) -> %d velas recebidas | Último Close: %.5f | Vol: %.0f",
            par, active_id, len(velas), c.get("close", 0.0), c.get("volume", 0.0)
        )

    # 4. TESTE DA FEATURE ENGINE COM DADOS REAIS
    logger.info("[TESTE 4/7] Processando dados com a FeatureEngine (16 variáveis)...")
    engine = FeatureEngine(config.estrategia)
    df_raw = pd.DataFrame(velas)
    features_df = engine.calculate_features(df_raw)
    assert not features_df.empty, "DataFrame de features vazio"
    assert features_df.shape[1] == 16, f"Esperadas 16 features, obtidas {features_df.shape[1]}"
    assert not features_df.isna().any().any(), "Features contêm valores NaN"
    logger.info("  ✅ 16 features calculadas com sucesso sem valores nulos (NaN).")
    ultima = features_df.iloc[-1]
    logger.info(
        "     Amostra de features -> Slope EMA21: %.4f | Afastamento 80: %.4f | ADX: %.2f | ATR: %.5f",
        ultima["feat_ema21_slope"], ultima["feat_afastamento_pct"],
        ultima["feat_adx"], ultima["feat_atr_20"]
    )

    # 5. TESTE DE INFERÊNCIA MACHINE LEARNING (MLCore)
    logger.info("[TESTE 5/7] Executando inferência estatística P(Win) via LightGBM...")
    ml = MLCore(config.machine_learning)
    assert ml.is_ready, "Modelo ML não está carregado ou pronto"
    p_call = ml.predict_win_probability(features_df, direction="CALL")
    p_put = ml.predict_win_probability(features_df, direction="PUT")
    assert 0.0 <= p_call <= 1.0, "Probabilidade de CALL fora do intervalo [0, 1]"
    assert 0.0 <= p_put <= 1.0, "Probabilidade de PUT fora do intervalo [0, 1]"
    assert abs((p_call + p_put) - 1.0) < 1e-4, "A soma P(CALL) + P(PUT) deve ser 1.0"
    logger.info("  ✅ MLCore operacional: P(CALL) = %.2f%% | P(PUT) = %.2f%% (Limiar: %.2f%%)",
                p_call * 100, p_put * 100, ml.threshold * 100)

    # 6. TESTE DE GESTÃO DE RISCO E CRITÉRIO DE KELLY
    logger.info("[TESTE 6/7] Validando RiskManager e Dimensionamento de Lote...")
    risk = RiskManager(config.gestao_risco)
    risk.sync_balance(saldo_real)
    pode_operar, motivo = risk.can_trade()
    assert pode_operar, f"Operação bloqueada inesperadamente: {motivo}"
    logger.info("  ✅ Autorização de trade: %s", motivo)

    # Teste de Kelly para probabilidade favorável (ex: 68%)
    lote_otimo = risk.calculate_position_size(win_probability=0.68, payout=0.85)
    assert lote_otimo >= 2.0, "Lote calculado deve respeitar o valor mínimo da corretora (R$2,00)"
    logger.info("  ✅ Critério de Kelly Fracionário: Lote calculado = R$%.2f para P(Win)=68%%", lote_otimo)

    # 7. TESTE DO ORQUESTRADOR COMPLETO (BotOrchestrator)
    logger.info("[TESTE 7/7] Testando ciclo completo do BotOrchestrator em conta real de prática...")
    bot = BotOrchestrator(config)
    await bot._ensure_connection()
    assert bot.conector.is_connected, "BotOrchestrator não conectou à corretora"

    logger.info("  -> Executando process_pair('EURUSD-OTC') com cotações em tempo real...")
    await bot.process_pair("EURUSD-OTC")
    logger.info("  ✅ process_pair executado com sucesso.")

    logger.info("  -> Testando encerramento suave (Graceful Shutdown)...")
    await bot.stop()
    await client.close()
    logger.info("  ✅ Conexões liberadas e recursos finalizados com segurança.")

    # 8. TESTE DE ANALYTICS — PARES E HORÁRIOS
    logger.info("[TESTE 8/8] Validando analytics de par/horário a partir de live_trades.csv...")
    from pathlib import Path
    live_csv = Path("data/live_trades.csv")
    if live_csv.exists():
        import numpy as np
        df_live = pd.read_csv(live_csv)
        df_live = df_live[df_live["direction"].isin(["CALL", "PUT"])]
        df_live = df_live[df_live["is_win"].isin([0, 1])]
        df_live["dt"]   = pd.to_datetime(df_live["timestamp"], unit="s")
        df_live["hora"] = df_live["dt"].dt.hour
        wr_geral = df_live["is_win"].mean()
        # Horas de ouro (12, 8, 19) devem ter WR > WR geral
        wr_horas_top = df_live[df_live["hora"].isin([12, 8, 19])]["is_win"].mean()
        par_wr = df_live.groupby("par")["is_win"].mean()
        melhor_par = par_wr.idxmax()
        melhor_wr  = par_wr.max()
        assert wr_horas_top > wr_geral, (
            f"Horas de ouro deveriam superar WR geral ({wr_horas_top:.1%} vs {wr_geral:.1%})"
        )
        logger.info(
            "  ✅ Analytics validados | WR geral: %.1f%% | WR horas de ouro: %.1f%% | Melhor par: %s (%.1f%%)",
            wr_geral * 100, wr_horas_top * 100, melhor_par, melhor_wr * 100,
        )
    else:
        logger.info("  ⚠️  live_trades.csv ausente — teste de analytics ignorado.")

    logger.info("==================================================")
    logger.info("  TODOS OS 8 TESTES FORAM APROVADOS COM SUCESSO!")
    logger.info("==================================================")


if __name__ == "__main__":
    asyncio.run(run_test_suite())
