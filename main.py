"""Ponto de entrada e orquestrador principal assíncrono do bot autônomo de trading.

# GOLDCODE EXEMPTION: Orquestrador assíncrono acoplado com injeção de dependências e threading lifecycle.
Gerencia conexão WebSocket, detecção de nova vela, extração de features,
direção técnica primária, inferência estatística P(Win) via MLCore, gestão de risco
e execução assíncrona não-bloqueante de ordens.
"""

import asyncio
import datetime
import logging
import signal
import threading
import time
from typing import Dict, List, Optional

# Mapeamento TF → segundos e minutos de expiração
_TF_SECONDS: Dict[str, int] = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}
_TF_EXP_MINUTES: Dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}

import pandas as pd

from config import BotConfig, ConexaoConfig, EstrategiaConfig, GestaoRiscoConfig, MachineLearningConfig, TradingConfig, get_config
from dummy_conector import DummyConector
from ml_core import MLCore
from ema_wave import ClockState, EMAWaveIndicator
from features import FeatureEngine
from online_learning import OnlineLearner
from radar import PairRadar
from risk import RiskManager
import os as _os
if _os.getenv("HEADLESS"):
    from ui_dashboard import TerminalUI as AppWindow  # type: ignore[assignment]  # noqa: F401
else:
    try:
        from gui_dashboard import AppWindow  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        from ui_dashboard import TerminalUI as AppWindow  # type: ignore[assignment]  # noqa: F401
from trend_sr import MarketStructure, TrendDirection
from smc_detector import SMCDetector
from volatility_models import evaluate_volatility_regime
from quant_factors import extract_vibe_quant_features
from errors import BrokerOrderRefusalError
from telemetry import LatencyTracker
from brain import CerebroQuant
from micro_sniper import MicroSniper, PreDecisaoSniper

try:
    from conector import IqOptionClient
except ImportError:
    from iq_client import IqOptionClient  # type: ignore


class BotOrchestrator:
    """Orquestrador principal assíncrono para o bot de trading autônomo."""

    def __init__(self, config: Optional[BotConfig] = None, ui: Optional["AppWindow"] = None) -> None:
        """Inicializa Config, FeatureEngine, RiskManager, MLCore e Conector."""
        self.logger = logging.getLogger("bot_orchestrator")

        self.config = config or (get_config() if get_config else BotConfig())
        self.engine = FeatureEngine(self.config.estrategia)
        self.risk = RiskManager(self.config.gestao_risco)
        self.ml = MLCore(self.config.machine_learning)
        self.learner = OnlineLearner(
            self.config.machine_learning, timeframe=self.config.trading.timeframe
        )
        self.wave = EMAWaveIndicator(period=34, slope_bars=3)
        self.market_structure = MarketStructure()
        self.smc = SMCDetector()
        self.telemetry = LatencyTracker()
        self.ui = ui if ui is not None else AppWindow(self.config.trading.pares_moedas)
        self.ui.conta_tipo = self.config.conexao.account_type

        try:
            self.conector = IqOptionClient(self.config.conexao)
        except Exception:
            self.conector = DummyConector()

        self.radar = PairRadar(
            conector=self.conector,
            wave=self.wave,
            engine=self.engine,
            candidates=list(self.config.trading.pares_moedas) if self.config.trading.pares_moedas else None,
            min_payout=self.config.trading.min_payout,
            min_adx=20.0,
            timeframe=self.config.trading.timeframe,
        )
        self.active_pairs: List[str] = list(self.radar.active_pairs)

        self.cerebro = CerebroQuant(threshold_ml=self.config.machine_learning.ml_threshold)
        self.sniper = MicroSniper(ml_threshold=self.config.machine_learning.ml_threshold)

        # threading.Event é thread-safe: funciona entre a thread Tkinter e a asyncio
        self._stop_event = threading.Event()
        # Base sempre M5 (menor TF multi-operação) — garante tick rápido
        self._tf_seg = 300
        self._timeframes: Dict[str, int] = {
            tf: _TF_SECONDS[tf] for tf in self.config.trading.timeframes
            if tf in _TF_SECONDS
        } or {"M5": 300, "M15": 900}
        self._pending_trades: int = 0
        self._max_simultaneous: int = (
            1 if getattr(self.config.gestao_risco, "martingale_enabled", False)
            else self.config.trading.max_simultaneous_trades
        )
        self._ultimo_dia: Optional[datetime.date] = None

    async def _ensure_connection(self) -> None:
        """Estabelece ou restaura a conexão com a corretora com retentativas."""
        tentativa = 1
        while not self.conector.is_connected and not self._stop_event.is_set():
            try:
                self.logger.info("Conectando à corretora (tentativa %d)...", tentativa)
                await self.conector.close()
                await self.conector.connect()
                saldo = await self.conector.get_balance()
                self.risk.sync_balance(saldo)
                self.ui.status_conexao = "ONLINE (CONECTADO)"
                self.ui.update_risk(
                    self.risk.banca_inicial, self.risk.banca_atual, self.risk.lucro_diario,
                    self.risk.drawdown_diario_pct, self.config.gestao_risco.max_daily_drawdown_pct,
                )
                self.ui.refresh()
                self.logger.info("Conectado com sucesso! Saldo: $%.2f", saldo)
                return
            except Exception as exc:
                atraso = min(5 * tentativa, 30)
                self.logger.warning("Falha na conexão (%s). Retentando em %ds...", exc, atraso)
                await asyncio.sleep(atraso)
                tentativa += 1

    async def _wait_for_next_candle(self) -> None:
        """Pausa até a abertura da próxima vela, sincronizando com a virada do timeframe."""
        agora = time.time()
        proxima = ((int(agora) // self._tf_seg) + 1) * self._tf_seg
        while not self._stop_event.is_set():
            restante = proxima - time.time()
            if restante <= 0.4:
                # Pequena folga para a corretora fechar a vela e disponibilizar os dados
                await asyncio.sleep(max(0.0, restante) + 0.6)
                break
            espera = min(2.0, max(0.1, restante - 0.4))
            await asyncio.sleep(espera)
            self.ui.refresh()

    def _get_active_timeframes(self, ts: int) -> List[str]:
        """Retorna quais TFs têm vela nova no timestamp ts."""
        return [tf for tf, sec in self._timeframes.items() if ts % sec == 0] or ["M5"]

    @staticmethod
    def _is_weekend_pause(now: Optional[datetime.datetime] = None) -> bool:
        """Verifica se estamos na janela de pausa de fim de semana.

        Pausa: Sexta-feira 18:00 BRT (21:00 UTC) até Domingo 20:00 BRT (23:00 UTC).
        - Sexta-feira >= 21h UTC (18h BRT)
        - Sábado inteiro (00h às 23h59 UTC)
        - Domingo < 23h UTC (20h BRT)
        """
        now_utc = now or datetime.datetime.now(datetime.timezone.utc)
        wd = now_utc.weekday()  # 0=Seg, 4=Sex, 5=Sáb, 6=Dom
        h = now_utc.hour
        if wd == 4 and h >= 21:
            return True
        if wd == 5:
            return True
        if wd == 6 and h < 23:
            return True
        return False

    @staticmethod
    def _resample_to_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Reamostra DataFrame M5 para o TF alvo (M15, M30, H1)."""
        secs = _TF_SECONDS.get(tf, 900)
        if "from" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["from"], unit="s")
        elif "datetime" not in df.columns:
            return df
        else:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        rule = f"{secs}s"
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        if "from" in df.columns:
            agg["from"] = "count"
        resampled = df.resample(rule, closed="left", label="left").agg(agg)
        if "from" in resampled.columns:
            n_per_bar = secs // 300
            resampled = resampled[resampled["from"] == n_per_bar].drop(columns=["from"])
        return resampled.dropna().reset_index()

    async def _execute_and_await_trade(
        self,
        par: str,
        lote: float,
        direcao: str,
        prob_win: float,
        features_row: Optional[pd.Series] = None,
        expiracao_minutos: Optional[int] = None,
    ) -> None:
        """Envia ordem de entrada e aguarda expiração de forma não-bloqueante."""
        exp_min = expiracao_minutos if expiracao_minutos is not None else self.config.trading.expiration_minutes
        self._pending_trades += 1
        current_lote = lote
        current_gale = 0
        max_gales = (
            self.config.gestao_risco.martingale_max_levels
            if getattr(self.config.gestao_risco, "martingale_enabled", False)
            else 0
        )

        try:
            while True:
                gale_tag = f" [GALE {current_gale}]" if current_gale > 0 else ""
                self.logger.info(
                    "Disparando %s em %s%s | Lote: $%.2f | P(Win): %.2f%% | Exp: %dm",
                    direcao.upper(), par, gale_tag, current_lote, prob_win * 100, exp_min,
                )
                saldo_antes = await self.conector.get_balance()
                t_buy_start = time.perf_counter()
                resp = await self.conector.buy(
                    par=par, valor=current_lote, direcao=direcao, expiracao_minutos=exp_min
                )
                t_buy_ms = (time.perf_counter() - t_buy_start) * 1000
                self.telemetry.record("order", t_buy_ms)

                if isinstance(resp, dict) and "message" in resp and "id" not in resp:
                    err = BrokerOrderRefusalError(par=par, reason=resp["message"])
                    self.logger.warning("Corretora recusou ordem%s (latência: %.1fms): %s", gale_tag, t_buy_ms, err)
                    break
                order_id = resp.get("id") if isinstance(resp, dict) else resp
                exp_ts = resp.get("exp_timestamp") if isinstance(resp, dict) else None
                restante = max(0, exp_ts - int(time.time())) if exp_ts else (exp_min * 60)
                self.logger.info(
                    "Ordem confirmada em %s%s (ID: %s | %.1fms). Monitorando expiração (%ds restantes)...",
                    par, gale_tag, order_id, t_buy_ms, restante,
                )

                # Payout decimal da operação (ex: 0.85 = 85%)
                payout = getattr(self.conector, "get_payout", lambda p: 0.85)(par)
                if isinstance(resp, dict):
                    if "profit_percent" in resp:
                        payout = float(resp["profit_percent"]) / 100.0
                    elif "profit_income" in resp:
                        payout = (float(resp["profit_income"]) - 100.0) / 100.0

                # 1. Tenta capturar a resolução oficial da ordem emitida pela corretora (option-closed via WebSocket)
                opt_result: Optional[Dict[str, Any]] = None
                if hasattr(self.conector, "wait_for_option_result") and order_id:
                    opt_result = await self.conector.wait_for_option_result(order_id, timeout=restante + 6.0)

                strike = float(resp.get("value") or 0.0) if isinstance(resp, dict) else 0.0
                close_price = 0.0

                if opt_result is not None:
                    # Resolução oficial direta da corretora
                    win_str = str(opt_result.get("win", "")).lower()
                    profit_amount = float(opt_result.get("profit_amount", 0.0))
                    amount = float(opt_result.get("amount", current_lote))
                    close_price = float(opt_result.get("close_quote") or opt_result.get("close_price") or 0.0)
                    if not strike:
                        strike = float(opt_result.get("value") or 0.0)

                    if win_str == "win" or profit_amount > amount:
                        is_win, is_tie = True, False
                        # BUG FIX #2: usa profit_amount real da corretora como base do lucro líquido
                        pnl = round(profit_amount - amount, 2) if profit_amount > amount else round(current_lote * payout, 2)
                    elif win_str in {"equal", "tie"} or (profit_amount == amount and profit_amount > 0):
                        is_win, is_tie = False, True
                        pnl = 0.0
                    else:
                        is_win, is_tie = False, False
                        pnl = -round(current_lote, 2)
                else:
                    # 2. Fallback: Aguarda término da expiração e consulta a vela exata que expirou (penúltima vela)
                    restante_agora = max(0, exp_ts - int(time.time())) if exp_ts else 0
                    if restante_agora > 0:
                        await asyncio.sleep(restante_agora + 2.0)

                    velas = await self.conector.get_candles(
                        par=par, timeframe=self.config.trading.timeframe, contagem=3
                    )
                    trade_candle = None
                    if velas and exp_ts:
                        # Busca vela cujo início seja exatamente tf_seg segundos antes da expiração
                        alvo_from = exp_ts - self._tf_seg
                        for v in reversed(velas):
                            if abs(v.get("from", 0) - alvo_from) <= 10:
                                trade_candle = v
                                break
                    if not trade_candle and velas:
                        trade_candle = velas[-2] if len(velas) >= 2 else velas[-1]

                    close_price = float(trade_candle.get("close", 0.0)) if trade_candle else 0.0
                    dir_norm = direcao.strip().upper()
                    is_win = (close_price > strike) if dir_norm == "CALL" else (close_price < strike)
                    is_tie = (close_price == strike) or (strike <= 0.0 or close_price <= 0.0)
                    pnl = round(current_lote * payout, 2) if is_win else (-round(current_lote, 2) if not is_tie else 0.0)

                # Cálculo ESTRITAMENTE LÍQUIDO do PnL:
                if is_win:
                    self.logger.info(
                        "🎉 Trade VENCEDOR em %s%s | Strike: %.5f | Close: %.5f | Stake: $%.2f | PnL Líquido: +$%.2f",
                        par, gale_tag, strike, close_price, current_lote, pnl,
                    )
                    self.risk.register_trade_result(pnl=pnl, was_win=True)
                    self.ui.add_trade_log(par, direcao, current_lote, pnl, True)
                elif is_tie:
                    pnl = 0.0
                    self.logger.info(
                        "ℹ️ Trade EMPATADO em %s%s | Strike: %.5f | Close: %.5f | Stake: $%.2f | PnL: $0.00",
                        par, gale_tag, strike, close_price, current_lote,
                    )
                    self.risk.register_trade_result(pnl=0.0, was_win=True)
                    self.ui.add_trade_log(par, direcao, current_lote, 0.0, True)
                else:
                    pnl = -round(current_lote, 2)
                    self.logger.warning(
                        "⚠️ Trade PERDEDOR em %s%s | Strike: %.5f | Close: %.5f | Stake: $%.2f | PnL Líquido: -$%.2f",
                        par, gale_tag, strike, close_price, current_lote, abs(pnl),
                    )
                    self.risk.register_trade_result(pnl=pnl, was_win=False)
                    self.ui.add_trade_log(par, direcao, current_lote, abs(pnl), False)

                if self.conector.__class__.__name__ != "DummyConector":
                    saldo_atual = await self.conector.get_balance()
                    self.risk.sync_balance(saldo_atual)

                self.ui.update_risk(
                    self.risk.banca_inicial, self.risk.banca_atual, self.risk.lucro_diario,
                    self.risk.drawdown_diario_pct, self.config.gestao_risco.max_daily_drawdown_pct,
                )
                self.ui.refresh()

                if not is_tie:
                    self.cerebro.recompensar(is_win)

                if not is_tie and features_row is not None:
                    self.learner.record_trade(features=features_row, direction=direcao, is_win=is_win, par=par)
                    if self.learner.should_retrain_trades():
                        asyncio.create_task(
                            self.learner.run_learning_cycle(
                                self.conector, self.ml, par_coleta=par, motivo="5 trades reais concluídos"
                            )
                        )

                # Controle de fluxo do Martingale:
                if is_win or is_tie:
                    if current_gale > 0:
                        self.logger.info("✅ [MARTINGALE RECUPERADO] Sucesso no Gale %d em %s! Ciclo concluído.", current_gale, par)
                    break
                else:
                    if current_gale < max_gales:
                        current_gale += 1
                        current_lote = self.risk.calculate_gale_stake(current_lote)
                        pode_operar, motivo = self.risk.can_trade()
                        if not pode_operar:
                            self.logger.warning("🚨 Martingale abortado pelas travas de risco: %s", motivo)
                            break
                        if current_lote > self.risk.banca_atual:
                            self.logger.warning(
                                "🚨 Martingale abortado: Lote R$%.2f excede banca disponível de R$%.2f.",
                                current_lote, self.risk.banca_atual,
                            )
                            break
                        self.logger.warning(
                            "🚨 [MARTINGALE GALE %d/%d] Disparando entrada imediata na próxima vela em %s (%s) | Lote: R$%.2f (%.1fx) | Exp: %dm",
                            current_gale, max_gales, par, direcao, current_lote,
                            getattr(self.config.gestao_risco, "martingale_multiplier", 2.2), exp_min,
                        )
                        continue
                    else:
                        if max_gales > 0:
                            self.logger.warning("❌ [MARTINGALE STOP] Ciclo encerrou em perda após %d Gales em %s.", max_gales, par)
                        break
        except Exception as exc:
            self.logger.error("Erro na ordem de %s (%s): %s", par, direcao, exc)
            self.risk.register_trade_result(pnl=-lote, was_win=False)
        finally:
            self._pending_trades = max(0, self._pending_trades - 1)

    async def _sniper_hunt_target(self, alvo: PreDecisaoSniper, df_historico: pd.DataFrame) -> None:
        """Monitora tick-a-tick nos primeiros 1/4 da vela para disparar na taxa com rejeição."""
        par = alvo.par
        timeframe = alvo.timeframe
        self.logger.info(
            "🦅 [SNIPER ATIVO] %s [%s] Caçando taxa %s (alvo: %.5f | teto: %.5f | máx %.0fs)",
            par, timeframe, alvo.direcao, alvo.taxa_alvo, alvo.teto_invalida, alvo.max_segundos_espera,
        )

        intervalo_poll = 0.5  # Amostragem de 500ms
        try:
            if hasattr(self.conector, "subscribe_candle"):
                await self.conector.subscribe_candle(par)
        except Exception:
            pass

        while not self._stop_event.is_set():
            try:
                preco_agora = await self.conector.get_current_price(par)
                if preco_agora <= 0.0:
                    await asyncio.sleep(intervalo_poll)
                    continue

                status, _, motivo = self.sniper.processar_tick(par, preco_agora)

                if status == "TRIGGER":
                    # REGRA 4: Revalidação em Tempo Real (Comportamento A)
                    self.logger.info("⚡ %s: Revalidando telemetria do sinal em tempo real...", par)
                    valido = True
                    try:
                        if alvo.features_row is not None and not alvo.features_row.empty:
                            prob_reval = self.ml.predict_proba(alvo.features_row)
                            if prob_reval < self.config.machine_learning.ml_threshold:
                                self.logger.warning(
                                    "🚫 %s: Sinal se dissipou na revalidação (P(Win) %.1f%% < %.1f%%). Abortando disparo.",
                                    par, prob_reval * 100, self.config.machine_learning.ml_threshold * 100,
                                )
                                valido = False
                    except Exception as _e_rev:
                        self.logger.debug("Revalidação ML silenciosa: %s", _e_rev)

                    if valido:
                        self.logger.info(
                            "🎯 [SNIPER SUCESSO] Disparando %s %s na taxa de ouro %.5f | P(Win): %.1f%% | Exp: %dm",
                            par, alvo.direcao, preco_agora, alvo.prob_win_inicial * 100, alvo.exp_minutos,
                        )
                        await self._execute_and_await_trade(
                            par, alvo.lote, alvo.direcao, alvo.prob_win_inicial,
                            alvo.features_row, expiracao_minutos=alvo.exp_minutos,
                        )
                    break

                elif status in ("TIMEOUT", "INVALIDATED"):
                    self.logger.info("❌ [SNIPER ENCERRADO] %s [%s]: %s", par, timeframe, motivo)
                    break

                await asyncio.sleep(intervalo_poll)

            except Exception as e_hunt:
                self.logger.error("Erro na caça sniper de %s: %s", par, e_hunt)
                self.sniper.limpar_alvo(par)
                break

    async def process_pair(
        self,
        par: str,
        velas_cache: Optional[list] = None,
        timeframe: str = "M15",
        expiracao_minutos: Optional[int] = None,
    ) -> None:
        """Processa ciclo completo de análise quantitativa, ML e ordem para um par."""
        exp_min = expiracao_minutos if expiracao_minutos is not None else _TF_EXP_MINUTES.get(timeframe, 15)

        # 0. Limite de trades simultâneos
        if self._pending_trades >= self._max_simultaneous:
            self.logger.debug("%s [%s]: %d trades abertos ≥ limite %d. Pulando.", par, timeframe, self._pending_trades, self._max_simultaneous)
            return

        # 1. Consulta RiskManager.can_trade()
        permitido, motivo = self.risk.can_trade()
        if not permitido:
            self.logger.warning("Análise interrompida para %s: %s", par, motivo)
            return

        # 1.1 Pausa de Fim de Semana (Sex 18:00 BRT até Dom 20:00 BRT)
        if self._is_weekend_pause():
            self.logger.info("⏸️ %s [%s]: Pausa de Fim de Semana ativa (Sex 18h a Dom 20h BRT). Pulando ciclo.", par, timeframe)
            return

        # 1.2 Filtro de Horário (UTC) — bloqueia horas com WR historicamente < 42%
        # UTC: {2, 4, 13, 18, 21, 22, 23} -> BRT: {23h, 01h, 10h, 15h, 18h, 19h, 20h}
        _HORAS_BLOQUEADAS = {2, 4, 13, 18, 21, 22, 23}
        _hora_utc = datetime.datetime.now(datetime.timezone.utc).hour
        if _hora_utc in _HORAS_BLOQUEADAS:
            self.logger.info("🕐 %s [%s]: Hora %dh UTC (%dh BRT) bloqueada. Pulando ciclo.", par, timeframe, _hora_utc, (_hora_utc - 3) % 24)
            return

        try:
            # 2. Coleta de velas: se M5 usa cache do radar (0ms); outros TFs buscam nativo da corretora
            if timeframe == "M5" and velas_cache and len(velas_cache) >= 85:
                velas = velas_cache
                t_candles_ms = 0.0
            else:
                t_c0 = time.perf_counter()
                velas = await self.conector.get_candles(
                    par=par, timeframe=timeframe, contagem=100
                )
                t_candles_ms = (time.perf_counter() - t_c0) * 1000

            if len(velas) < 85:
                self.logger.info("Histórico insuficiente para %s [%s] (%d velas).", par, timeframe, len(velas))
                self.ui.update_pair(par, "NEUTRO", 0.0, "3h")
                return

            df = pd.DataFrame(velas)
            if "min" in df.columns and "low" not in df.columns:
                df = df.rename(columns={"min": "low", "max": "high"})

            # 3. Processa histórico via FeatureEngine.calculate_features()
            t_f0 = time.perf_counter()
            features_df = self.engine.calculate_features(df)
            t_features_ms = (time.perf_counter() - t_f0) * 1000

            if features_df.empty:
                self.ui.update_pair(par, "NEUTRO", 0.0, "3h")
                return

            # 3.1 Mapeia Estrutura de Mercado: Tendência por Médias (9,21,50,200) e Suporte/Resistência
            trend_res, sr_res = self.market_structure.evaluate(df)
            self.logger.info(
                "%s [Estrutura] %s | Sup: %.5f (-%.2f%%) | Res: %.5f (+%.2f%%)",
                par, trend_res.summary, sr_res.nearest_support, sr_res.dist_support_pct,
                sr_res.nearest_resistance, sr_res.dist_resistance_pct
            )

            # 3.2 Confluência Institucional SMC & Volatilidade Parkinson/Garman-Klass (Vibe-Trading)
            smc_res = self.smc.analyze(df)
            vol_rep = evaluate_volatility_regime(df)
            quant_alphas = extract_vibe_quant_features(df)

            # 4. Mapeia direção técnica (EMA Wave Rag Horner + Alinhamento)
            wave_sig = self.wave.evaluate_latest(df)
            ultima = features_df.iloc[-1]
            relogio_txt = wave_sig.state.value if hasattr(wave_sig, "state") else "3h"

            # FILTRO 1: Bloqueio do Relógio com Gatilho Inteligente de Rompimento (Breakout S/R + CHOP + SMC)
            if wave_sig.state == ClockState.BULLISH_STRONG:
                direcao: Optional[str] = wave_sig.signal or "CALL"
            elif wave_sig.state == ClockState.BEARISH_STRONG:
                direcao = wave_sig.signal or "PUT"
            elif smc_res.institutional_signal and (trend_res.direction == TrendDirection.ALTA if smc_res.institutional_signal == "CALL" else TrendDirection.BAIXA):
                direcao = smc_res.institutional_signal
                self.logger.info("🏛️ %s: Confluência Institucional SMC — Mitigação de Order Block (%s).", par, direcao)
            elif sr_res.breakout_up and (trend_res.direction == TrendDirection.ALTA or trend_res.chop_index < 58.0):
                direcao = "CALL"
                self.logger.info("🚀 %s: Rompimento Altista de Resistência detectado (Breakout CALL | CHOP=%.1f).", par, trend_res.chop_index)
            elif sr_res.breakout_down and (trend_res.direction == TrendDirection.BAIXA or trend_res.chop_index < 58.0):
                direcao = "PUT"
                self.logger.info("🚀 %s: Rompimento Baixista de Suporte detectado (Breakout PUT | CHOP=%.1f).", par, trend_res.chop_index)
            else:
                direcao = None

            # FILTRO 1.0: Confluência em Cascata Multi-Timeframe (Macro H1 -> Gatilho M15)
            # Regra comprovada: eleva assertividade para 59.40% ao vetar ordens contra a tendência macro H1
            if direcao:
                try:
                    velas_h1 = await self.conector.get_candles(par=par, timeframe="H1", contagem=40)
                    if len(velas_h1) >= 20:
                        df_h1 = pd.DataFrame(velas_h1)
                        w_h1 = self.wave.calculate(df_h1)
                        ang_h1 = float(w_h1["wave_angle"].iloc[-1])
                        macro_vies = "CALL" if ang_h1 >= 5.0 else ("PUT" if ang_h1 <= -5.0 else None)
                        if direcao != macro_vies:
                            self.logger.info(
                                "🚫 %s descartado por filtro Macro H1: Sinal M15=%s contra Viés H1=%s (Ângulo H1=%.1f°).",
                                par, direcao, macro_vies or "LATERAL", ang_h1,
                            )
                            self.ui.update_pair(par, "MACRO/H1", 0.0, relogio_txt)
                            return
                except Exception as _e_h1:
                    self.logger.debug("Falha na checagem macro H1 (%s). Prosseguindo com análise nativa.", _e_h1)

            # FILTRO 1.1: Filtro de Exaustão Extrema de Volatilidade (Garman-Klass / Parkinson)
            if direcao and not vol_rep.is_safe_for_trend:
                self.logger.info("🚫 %s descartado: Exaustão extrema de volatilidade (GK ratio=%.2f > 2.50).", par, vol_rep.vol_ratio_to_mean)
                self.ui.update_pair(par, "VOL/EXAUST", 0.0, relogio_txt)
                return

            # FILTRO 1.2: Absorção Institucional Alpha101-12 (Kakushadze / WorldQuant)
            # alpha12 > 0 → volume subindo contra preço → pressão vendedora institucional (bearish)
            # alpha12 < 0 → volume subindo a favor do preço → pressão compradora institucional (bullish)
            # Só ativo quando ALPHA12_FILTER=true — desativado por padrão até validação histórica
            if direcao and self.config.machine_learning.alpha12_filter:
                a12 = quant_alphas.get("vibe_alpha101_12", 0.0)
                thr12 = self.config.machine_learning.alpha12_threshold
                bloqueado_a12 = (direcao == "CALL" and a12 >= thr12) or (direcao == "PUT" and a12 <= -thr12)
                if bloqueado_a12:
                    self.logger.info(
                        "🏛️ %s descartado: Absorção institucional Alpha12=%.3f contra %s (limiar=±%.2f).",
                        par, a12, direcao, thr12,
                    )
                    self.ui.update_pair(par, "ALPHA12/BLK", 0.0, relogio_txt)
                    return

            t_ml0 = time.perf_counter()
            prob_win = self.ml.predict_win_probability(features_df, direction=direcao, par=par) if direcao else 0.0
            t_ml_ms = (time.perf_counter() - t_ml0) * 1000

            self.telemetry.log_hot_path(par, t_candles_ms, t_features_ms, t_ml_ms)

            if not direcao:
                self.ui.update_pair(par, "NEUTRO/3h", 0.0, relogio_txt)
                self.logger.info("%s: Descartado por mercado lateral / consolidação (Relógio: %s).", par, relogio_txt)
                return

            # FILTRO 2: Hard Gate de Força de Tendência (ADX >= 20.0)
            adx_val = float(ultima.get("feat_adx", 0.0))
            if adx_val < 20.0:
                self.logger.info("🚫 %s descartado por ADX fraco: %.1f < 20.0 (sem tendência).", par, adx_val)
                self.ui.update_pair(par, "ADX/FRACO", prob_win, relogio_txt)
                return

            # FILTRO 3: Exaustão de Bollinger (%B Anti-Topo e Anti-Fundo)
            pct_b = float(ultima.get("feat_bollinger_pct_b", 0.5))
            if direcao == "CALL" and pct_b > 1.00:
                self.logger.info("🚫 %s descartado: CALL em topo esticado de Bollinger (%%B=%.2f > 1.00).", par, pct_b)
                self.ui.update_pair(par, "BOLL/TOPO", prob_win, relogio_txt)
                return
            if direcao == "PUT" and pct_b < 0.00:
                self.logger.info("🚫 %s descartado: PUT em fundo esticado de Bollinger (%%B=%.2f < 0.00).", par, pct_b)
                self.ui.update_pair(par, "BOLL/FUNDO", prob_win, relogio_txt)
                return

            # FILTRO 4: Filtro Geométrico de Pavio (Anti-Ruído / Rejeição)
            pavio_ok, motivo_pavio = self.engine.validar_ruido_pavio(features_df, direcao)
            if not pavio_ok:
                self.logger.info("🚫 %s descartado por excesso de pavio: %s", par, motivo_pavio)
                self.ui.update_pair(par, "PAVIO/DESC", prob_win, relogio_txt)
                return

            # FILTRO 5: Payout Mínimo da Corretora (MIN_PAYOUT >= 84%)
            payout = getattr(self.conector, "get_payout", lambda p: 0.85)(par)
            min_payout = getattr(self.config.trading, "min_payout", 0.84)
            if payout < min_payout:
                self.logger.info("🚫 %s descartado por payout insuficiente: %.0f%% < %.0f%%.", par, payout * 100, min_payout * 100)
                self.ui.update_pair(par, "PAYOUT/BX", prob_win, relogio_txt)
                return

            # Atualiza sempre o estado do par aprovado na interface gráfica
            self.ui.update_pair(par, direcao, prob_win, relogio_txt)

            # CÉREBRO CENTRAL: decisão por votação ponderada (A1-Tendência, A2-Volatilidade, A3-Institucional, A4-ML)
            cerebro_res = self.cerebro.decidir(
                features_df, wave_sig, smc_res, trend_res, vol_rep, direcao, prob_win
            )
            self.logger.info("🧠 %s | %s", par, cerebro_res.motivo)

            if cerebro_res.entrar:
                lote_base = self.risk.calculate_position_size(win_probability=prob_win, payout=payout)
                # No Martingale, o lote_base é a mão inicial (5% da banca sem divisão)
                if getattr(self.config.gestao_risco, "martingale_enabled", False):
                    lote = lote_base
                else:
                    lote = max(1.0, round(lote_base / self._max_simultaneous, 2))
                if lote > 0:
                    self.logger.info(
                        "📊 [%s] %s → %s | Lote: $%.2f (base $%.2f / %d simult.) | P(Win): %.1f%%",
                        timeframe, par, direcao, lote, lote_base, self._max_simultaneous, prob_win * 100,
                    )
                    if getattr(self.config.trading, "sniper_mode", True):
                        duracao_seg = _TF_SECONDS.get(timeframe, 900)
                        vela_ant = df.iloc[-1].to_dict() if not df.empty else {}
                        preco_abertura = float(df.iloc[-1]["close"]) if not df.empty else 0.0
                        alvo = self.sniper.registrar_pre_decisao(
                            par=par,
                            timeframe=timeframe,
                            direcao=direcao,
                            vela_anterior=vela_ant,
                            preco_abertura=preco_abertura,
                            prob_win=prob_win,
                            lote=lote,
                            exp_minutos=exp_min,
                            duracao_vela_segundos=duracao_seg,
                            features_row=ultima,
                        )
                        asyncio.create_task(self._sniper_hunt_target(alvo, df))
                    else:
                        asyncio.create_task(
                            self._execute_and_await_trade(par, lote, direcao, prob_win, ultima, expiracao_minutos=exp_min)
                        )
                else:
                    self.logger.info("%s [%s]: Lote zerado pelo gerenciamento de risco.", par, timeframe)
            else:
                self.logger.info(
                    "%s: Cérebro bloqueou entrada (score=%.3f < %.2f). Pesos: %s",
                    par, cerebro_res.score_final, self.cerebro.THRESHOLD_ENTRADA, self.cerebro.status(),
                )
        except Exception as exc:
            self.logger.error("Erro ao processar par %s: %s", par, exc)
            if "close frame" in str(exc) or "Connection" in str(type(exc).__name__):
                if hasattr(self.conector, "_is_connected"):
                    self.conector._is_connected = False

    async def run_loop(self) -> None:
        """Loop contínuo de varredura assíncrona dos pares monitorados."""
        self.logger.info("Iniciando loop do BotOrchestrator...")
        await self._ensure_connection()

        while not self._stop_event.is_set():
            try:
                if not self.conector.is_connected:
                    await self._ensure_connection()

                self.ui.set_current_analyzing_pair("Aguardando virada de vela...")
                await self._wait_for_next_candle()
                if self._stop_event.is_set():
                    break

                # 0. Verificação de Pausa de Fim de Semana (Sex 18h a Dom 20h BRT)
                if self._is_weekend_pause():
                    self.ui.set_current_analyzing_pair("⏸️ Pausa Fim de Semana (Reabre Dom 20h BRT)")
                    self.logger.info("⏸️ Pausa de Fim de Semana ativa (Sex 18h a Dom 20h BRT). Ciclo em repouso.")
                    self.ui.refresh()
                    await asyncio.sleep(10)
                    continue

                # Determina quais TFs têm vela nova neste tick
                ts_agora = int(time.time())
                tfs_ativos = self._get_active_timeframes(ts_agora)
                self.logger.info("⏱ Tick M5 | TFs ativos: %s | Trades abertos: %d/%d",
                                 tfs_ativos, self._pending_trades, self._max_simultaneous)

                # Radar Dinâmico: varre catálogo e retorna cache de velas M5 já coletadas
                candle_cache = await self.radar.scan_pairs()
                if candle_cache:
                    self.active_pairs = list(candle_cache.keys())
                if hasattr(self.ui, "sync_active_pairs"):
                    self.ui.sync_active_pairs(self.active_pairs)

                # Para cada TF ativo, analisa todos os pares
                for tf in tfs_ativos:
                    exp_min = _TF_EXP_MINUTES.get(tf, 15)
                    self.logger.info("📡 Varredura %s (%dm exp) | %d pares", tf, exp_min, len(self.active_pairs))
                    for par in self.active_pairs:
                        self.ui.set_current_analyzing_pair(f"[{tf}] {par}...")
                        await self.process_pair(par, velas_cache=candle_cache.get(par), timeframe=tf, expiracao_minutos=exp_min)
                self.ui.set_current_analyzing_pair("Aguardando próxima vela...")

                # Sincronização periódica de saldo (a cada ciclo de vela)
                if self.conector.__class__.__name__ != "DummyConector":
                    try:
                        saldo_ciclo = await self.conector.get_balance()
                        self.risk.sync_balance(saldo_ciclo)
                    except Exception as _e_bal:
                        self.logger.debug("Falha ao sincronizar saldo no ciclo: %s", _e_bal)

                # Reset automático ao virar o dia (reinicia lucro_diario e banca_inicial)
                hoje = datetime.date.today()
                if self._ultimo_dia is not None and self._ultimo_dia != hoje:
                    self.risk.reset_daily()
                    self.logger.info(
                        "📅 Virada do dia (%s → %s) | Nova base: $%.2f",
                        self._ultimo_dia, hoje, self.risk.banca_inicial,
                    )
                self._ultimo_dia = hoje

                self.ui.update_risk(
                    self.risk.banca_inicial, self.risk.banca_atual, self.risk.lucro_diario,
                    self.risk.drawdown_diario_pct, self.config.gestao_risco.max_daily_drawdown_pct,
                )

                # Atualiza indicador de aprendizado na UI
                _l_estado, _l_pct = self.learner.learning_state
                if hasattr(self.ui, "update_learning"):
                    self.ui.update_learning(_l_estado, _l_pct)
                self.ui.refresh()

                if self.learner.should_retrain_time():
                    par_padrao = self.active_pairs[0] if self.active_pairs else "EURUSD"
                    asyncio.create_task(
                        self.learner.run_learning_cycle(
                            self.conector, self.ml, par_coleta=par_padrao, motivo="janela periódica de 30 minutos"
                        )
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Erro inesperado no loop de execução: %s", exc)
                await asyncio.sleep(2.0)

        await self.stop()

    async def stop(self) -> None:
        """Encerramento seguro com liberação de conexões e recursos."""
        if not self._stop_event.is_set():
            self._stop_event.set()
        self.logger.info("Encerrando conexões e liberando recursos...")
        await self.conector.close()
        self.logger.info("BotOrchestrator finalizado com segurança.")


async def main(ui: AppWindow, stop_event: "threading.Event") -> None:  # type: ignore[name-defined]
    """Ponto de entrada assíncrono do bot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler("bot.log", mode="a", encoding="utf-8")],
    )

    orchestrator = BotOrchestrator(ui=ui)
    # Compartilha o mesmo threading.Event com o orquestrador
    orchestrator._stop_event = stop_event

    try:
        await orchestrator.run_loop()
    except asyncio.CancelledError:
        pass
    finally:
        if not stop_event.is_set():
            stop_event.set()
        await orchestrator.conector.close()
        logging.getLogger("main").info("Bot encerrado com segurança.")


if __name__ == "__main__":
    _config = get_config()
    _ui = AppWindow(_config.trading.pares_moedas)
    _stop = threading.Event()

    def _run_async() -> None:
        """Roda o event loop asyncio em thread de background."""
        try:
            asyncio.run(main(_ui, _stop))
        except Exception as exc:
            logging.getLogger("main").error("Erro fatal no bot: %s", exc)
        finally:
            _stop.set()

    if _os.getenv("HEADLESS"):
        # VPS/servidor sem display: asyncio roda direto na thread principal
        # Ctrl+C → KeyboardInterrupt → encerramento gracioso via signal handler
        def _handle_sigint(sig: int, frame: object) -> None:  # type: ignore[type-arg]
            _stop.set()

        signal.signal(signal.SIGINT, _handle_sigint)
        signal.signal(signal.SIGTERM, _handle_sigint)
        try:
            asyncio.run(main(_ui, _stop))
        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
    else:
        # Desktop: asyncio em thread de background, Tkinter mainloop na thread principal
        _bot_thread = threading.Thread(target=_run_async, name="bot-asyncio", daemon=True)
        _bot_thread.start()
        try:
            _ui.run_mainloop()  # Bloqueia aqui até o usuário fechar a janela
        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            _bot_thread.join(timeout=8.0)
