"""Módulo do Radar Dinâmico de Mercado (PairRadar).

Varre continuamente o catálogo de pares em tempo real de forma assíncrona,
adiciona à fila operacional apenas os pares que cumprem os requisitos
técnicos e de payout, e remove imediatamente os que entram em consolidação.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ema_wave import ClockState, EMAWaveIndicator
from features import FeatureEngine
from iq_constants import ACTIVE_IDS


class PairRadar:
    """Scanner de alta velocidade para seleção dinâmica de pares em tempo real."""

    def __init__(
        self,
        conector: Any,
        wave: EMAWaveIndicator,
        engine: FeatureEngine,
        candidates: Optional[List[str]] = None,
        min_payout: float = 0.84,
        min_adx: float = 22.0,
        timeframe: str = "M5",
    ) -> None:
        """Inicializa o radar de varredura dinâmica."""
        self._logger = logging.getLogger("pair_radar")
        self.conector = conector
        self.wave = wave
        self.engine = engine
        self.min_payout = min_payout
        self.min_adx = min_adx
        self.timeframe = timeframe

        # Catálogo de candidatos: pares normais + pares OTC aprovados
        # Pares OTC com taxa <= 43% (EURGBP, GBPUSD, NZDUSD, USDCAD) são bloqueados
        blacklist_otc = {"EURGBP-OTC", "GBPUSD-OTC", "NZDUSD-OTC", "USDCAD-OTC"}
        if candidates:
            self.candidates = [p for p in candidates if p not in blacklist_otc]
        else:
            candidatos_normais = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURJPY"]
            candidatos_otc_aprovados = ["AUDUSD-OTC", "EURJPY-OTC", "USDCHF-OTC", "EURUSD-OTC"]
            self.candidates = candidatos_normais + candidatos_otc_aprovados

        # Prioriza os pares vencedores por assertividade histórica comprovada (WR acima do breakeven)
        top_iniciais = [p for p in ["AUDUSD-OTC", "EURJPY", "USDJPY", "USDCHF-OTC", "EURJPY-OTC", "EURUSD-OTC"] if p in self.candidates]
        self.active_pairs: List[str] = top_iniciais[:4] if top_iniciais else list(self.candidates[:4])

    async def _evaluate_candidate(self, par: str) -> Optional[Tuple[str, float, list]]:
        """Avalia um único par contra os pré-requisitos do radar."""
        # 0. Filtro de Mercado Aberto na Corretora (ignora pares suspensos/fechados)
        if hasattr(self.conector, "is_asset_open") and not self.conector.is_asset_open(par):
            self._logger.debug("%s ignorado pelo radar: Ativo fechado ou suspenso na corretora.", par)
            return None

        # 1. Filtro de Payout Mínimo
        payout = getattr(self.conector, "get_payout", lambda p: 0.85)(par)
        if payout < self.min_payout:
            self._logger.debug("%s ignorado pelo radar: Payout %.0f%% < %.0f%%", par, payout * 100, self.min_payout * 100)
            return None

        # 2. Coleta de histórico recente de velas (FeatureEngine requer mínimo de 85 velas)
        try:
            velas = await self.conector.get_candles(par=par, timeframe=self.timeframe, contagem=100)
            if len(velas) < 85:
                return None
        except Exception as exc:
            self._logger.debug("Erro ao coletar velas no radar para %s: %s", par, exc)
            return None

        df = pd.DataFrame(velas)
        if "min" in df.columns and "low" not in df.columns:
            df = df.rename(columns={"min": "low", "max": "high"})

        # 3. Filtro de Relógio (EMA Wave 34) — Rejeita 3h (Consolidação)
        wave_sig = self.wave.evaluate_latest(df)
        if wave_sig.state == ClockState.NEUTRAL_SIDEWAYS:
            self._logger.debug("%s ignorado pelo radar: Relógio 3h (Lateral)", par)
            return None

        # 4. Filtro de Força de Tendência (ADX >= min_adx)
        try:
            feats = self.engine.calculate_features(df)
            if feats.empty:
                return None
            adx_val = float(feats.iloc[-1].get("feat_adx", 0.0))
            if adx_val < self.min_adx:
                self._logger.debug("%s ignorado pelo radar: ADX %.1f < %.1f", par, adx_val, self.min_adx)
                return None
        except Exception as exc:
            self._logger.debug("Erro no cálculo de features para %s: %s", par, exc)
            return None

        # Par aprovado em todos os crivos técnicos de tendência
        self._logger.info(
            "🎯 Radar APROVOU: %s | Payout: %.0f%% | Relógio: %s | ADX: %.1f",
            par, payout * 100, wave_sig.state.value, adx_val
        )
        return (par, adx_val, velas)

    async def scan_pairs(self) -> Dict[str, list]:
        """Varre concorrentemente todos os candidatos e retorna cache de velas por par.

        Retorna dict {par: velas} para os pares aprovados (máx. 4, por ADX desc.).
        Retorna {} quando desconectado — o orquestrador mantém sua fila atual.
        """
        if not getattr(self.conector, "is_connected", False):
            return {}

        # Sincroniza catálogo da corretora periodicamente a cada 3 minutos (transição Normal/OTC)
        agora = time.time()
        if hasattr(self.conector, "load_initialization_data"):
            if agora - getattr(self, "_last_catalog_refresh", 0.0) > 180.0:
                await self.conector.load_initialization_data()
                self._last_catalog_refresh = agora

        # Dispara avaliação concorrente assíncrona para máxima velocidade (<300ms)
        tarefas = [self._evaluate_candidate(par) for par in self.candidates]
        resultados = await asyncio.gather(*tarefas, return_exceptions=True)

        pares_validos = [r for r in resultados if isinstance(r, tuple) and len(r) == 3 and r[0]]
        # Ordena candidatos por maior ADX (tendência mais vigorosa primeiro)
        pares_validos.sort(key=lambda x: x[1], reverse=True)
        selecionados = pares_validos[:4]

        if selecionados:
            novos_nomes = [item[0] for item in selecionados]
            removidos = set(self.active_pairs) - set(novos_nomes)
            adicionados = set(novos_nomes) - set(self.active_pairs)
            if adicionados or removidos:
                self._logger.info(
                    "🔄 Radar Atualizou Fila | Ativos: %s | +Entraram: %s | -Saíram: %s",
                    novos_nomes, list(adicionados) or "Nenhum", list(removidos) or "Nenhum"
                )
            self.active_pairs = novos_nomes
            # Cache: par → velas já coletadas pelo radar (evita segundo round-trip na API)
            return {item[0]: item[2] for item in selecionados}
        else:
            abertos = [p for p in self.candidates if getattr(self.conector, "is_asset_open", lambda x: True)(p)]
            if abertos:
                self.active_pairs = abertos[:4]
            else:
                self._logger.warning("⚠️ Todos os pares configurados estão fechados na corretora no momento. Ative os pares OTC no .env para operar após as 14:30.")
                self.active_pairs = []
            return {}
