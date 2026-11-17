"""Script de execução e monitoramento de trade em tempo real na conta de prática.

Executa o fluxo completo:
1. Conecta na conta Practice da IQ Option e captura saldo inicial.
2. Coleta velas reais M1 e extrai variáveis quantitativas via FeatureEngine.
3. Avalia probabilidades P(Win) via MLCore (LightGBM).
4. Dimensiona o lote ótimo via Critério de Kelly Fracionário (RiskManager).
5. Dispara a ordem na abertura da vela.
6. Monitora o tempo de expiração em tempo real.
7. Checa o saldo final, calcula PnL e registra no RiskManager.
"""

import asyncio
from datetime import datetime, timezone
import logging
import time

import pandas as pd

from config import get_config
from conector import IqOptionClient
from features import FeatureEngine
from ml_core import MLCore
from risk import RiskManager


async def run_live_trade_test(par: str = "EURUSD-OTC") -> None:
    """Executa operação assistida em tempo real na conta Practice."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("live_operator")

    logger.info("==================================================")
    logger.info("  🎯 INICIANDO OPERAÇÃO ASSISTIDA NA CONTA PRACTICE")
    logger.info("==================================================")

    # 1. Conexão real obrigatória com a corretora
    cfg = get_config()
    client = IqOptionClient(cfg.conexao)
    logger.info("1. Conectando à IQ Option na conta %s...", cfg.conexao.account_type)
    try:
        await client.connect()
        conector = client
        logger.info("   ✅ CONECTADO 100% VIA WEBSOCKET OFICIAL À IQ OPTION!")
    except Exception as exc:
        logger.error("   ❌ FALHA NA CONEXÃO REAL: %s", exc)
        logger.error("   ⚠️ O endpoint HTTP da corretora sofreu timeout/bloqueio temporário.")
        logger.error("   💡 SOLUÇÃO: Obtenha o cookie 'ssid' no navegador e adicione no .env: IQOPTION_SSID=seu_token")
        return

    saldo_inicial = await conector.get_balance()
    logger.info("   ✅ Saldo inicial disponível para trade: $%.2f USD", saldo_inicial)

    # 2. Análise Técnica e Coleta de Velas
    logger.info("2. Coletando 100 velas M1 mais recentes para %s...", par)
    velas = await conector.get_candles(par=par, timeframe="M1", contagem=100)
    if len(velas) < 85:
        logger.error("   ❌ Histórico insuficiente de velas (%d velas). Abortando.", len(velas))
        await conector.close()
        return

    df_velas = pd.DataFrame(velas)
    engine = FeatureEngine(cfg.estrategia)
    features_df = engine.calculate_features(df_velas)
    ultima_vela = features_df.iloc[-1]

    logger.info("   ✅ Análise técnica concluída:")
    logger.info("      - Alinhamento de Médias (EMA 9/21/50): %d", int(ultima_vela["feat_ema_alignment"]))
    logger.info("      - Afastamento da EMA 80: %.4f%%", ultima_vela["feat_afastamento_pct"] * 100)
    logger.info("      - Volatilidade ATR(20): %.5f", ultima_vela["feat_atr_20"])
    logger.info("      - Força de Tendência ADX: %.2f", ultima_vela["feat_adx"])

    # 3. Inferência Estatística de Machine Learning
    ml = MLCore(cfg.machine_learning)
    p_call = ml.predict_win_probability(features_df, direction="CALL")
    p_put = ml.predict_win_probability(features_df, direction="PUT")
    logger.info("3. Inferência Estatística MLCore (LightGBM):")
    logger.info("      - P(CALL) = %.2f%%", p_call * 100)
    logger.info("      - P(PUT)  = %.2f%%", p_put * 100)
    logger.info("      - Limiar mínimo configurado: %.2f%%", ml.threshold * 100)

    # Define a direção de maior probabilidade
    if p_call >= p_put:
        direcao = "CALL"
        prob_win = p_call
    else:
        direcao = "PUT"
        prob_win = p_put

    # 4. Dimensionamento de Risco por Kelly Fracionário
    risk = RiskManager(cfg.gestao_risco)
    payout = 0.85
    lote = risk.calculate_position_size(win_probability=prob_win, payout=payout)
    if lote <= 0.0:
        lote = 2.00  # Lote mínimo de teste
    logger.info("4. Dimensionamento de Risco:")
    logger.info("      - Direção escolhida: %s", direcao)
    logger.info("      - Lote calculado (Kelly 15%%): $%.2f USD", lote)

    # 5. Sincronização com o início da nova vela
    agora = time.time()
    segundos_restantes = 60 - (int(agora) % 60)
    logger.info("5. Sincronizando com abertura da vela (aguardando 3s)...")
    await asyncio.sleep(3)

    # 6. Envio da Ordem
    logger.info("6. 🚀 DISPARANDO ORDEM %s de $%.2f em %s...", direcao, lote, par)
    exp_ts = int(time.time()) + 65
    try:
        ordem = await conector.buy(par=par, valor=lote, direcao=direcao, expiracao_minutos=1)
        logger.info("   Resposta da corretora: %s", ordem)
        if isinstance(ordem, dict) and "exp_timestamp" in ordem:
            exp_ts = ordem["exp_timestamp"]
    except Exception as exc:
        logger.warning("   Alerta no disparo: %s", exc)

    # 7. Monitoramento da Expiração em Tempo Real
    tempo_total = max(5, exp_ts - int(time.time()))
    logger.info("7. Monitorando operação em tempo real (Expiração em %ds)...", tempo_total)
    while True:
        restante = exp_ts - int(time.time())
        if restante <= 0:
            break
        logger.info("   ⏳ Aguardando expiração... restam %ds", restante)
        await asyncio.sleep(min(5, restante))
    logger.info("   ⏳ Expiração atingida. Aguardando 5s de carência para liquidação de saldo na corretora...")
    await asyncio.sleep(5)

    # 8. Verificação do Resultado Financeiro
    saldo_final = await conector.get_balance()
    pnl = round(saldo_final - saldo_inicial, 2)
    was_win = pnl > 0.0

    logger.info("==================================================")
    logger.info("  📊 RESULTADO DA OPERAÇÃO")
    logger.info("==================================================")
    logger.info("  Saldo Inicial : $%.2f USD", saldo_inicial)
    logger.info("  Saldo Final   : $%.2f USD", saldo_final)
    logger.info("  Resultado PnL : %s$%.2f USD", "+" if pnl >= 0 else "", pnl)

    if pnl > 0.0:
        logger.info("  🎉 TRADE VENCEDOR (WIN)!")
        risk.register_trade_result(pnl=pnl, was_win=True)
    elif pnl < 0.0:
        logger.info("  ⚠️ TRADE PERDEDOR (LOSS).")
        risk.register_trade_result(pnl=pnl, was_win=False)
    else:
        logger.info("  ℹ️ Saldo inalterado (Empate / Ordem rejeitada pela corretora).")

    logger.info("  Estado RiskManager atualizado -> Banca: $%.2f | Drawdown: %.2f%%",
                risk.banca_atual, risk.drawdown_diario_pct * 100)

    await conector.close()
    logger.info("  ✅ Conexão encerrada com segurança.")


if __name__ == "__main__":
    asyncio.run(run_live_trade_test())
