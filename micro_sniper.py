"""Módulo de Micro-Sniper e Telemetria Intracandle em Tempo Real.

Implementa a estratégia de Micro-Pullback com Margem de Segurança:
1. Recebe a Pré-Decisão do Cérebro Quant no fechamento da vela anterior.
2. Define a taxa alvo estrutural (C1: máxima anterior para CALL, mínima para PUT).
3. Define o teto máximo de invalidação (A: mínima anterior para CALL, máxima para PUT).
4. Monitora tick-a-tick em tempo real durante até 1/4 do tempo da vela (75s M5 / 225s M15).
5. Ao atingir a taxa desejada e registrar o 1º tick de reversão a favor (Tipo B),
   revalida a telemetria do Cérebro/ML em tempo real (Comportamento A).
6. Se o sinal persistir com P(Win) >= threshold, autoriza o disparo imediato com expiração
   estrita alinhada ao término da vela atual.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_LOG = logging.getLogger("micro_sniper")


@dataclass
class PreDecisaoSniper:
    """Representa uma oportunidade pré-aprovada aguardando taxa ideal."""

    par: str
    timeframe: str
    direcao: str                  # "CALL" ou "PUT"
    taxa_alvo: float              # C1: máx anterior (CALL) ou mín anterior (PUT)
    teto_invalida: float          # A: mín anterior (CALL) ou máx anterior (PUT)
    preco_abertura: float         # Preço de abertura da nova vela
    prob_win_inicial: float       # P(Win) da análise inicial no segundo 00
    ts_inicio_vela: float         # Epoch do início da vela
    max_segundos_espera: float    # 1/4 da duração da vela (75s para M5, 225s para M15)
    exp_minutos: int              # Expiração para o final da vela atual
    lote: float                   # Tamanho da posição calculada pelo RiskManager
    features_row: Optional[pd.Series] = None

    # Rastreamento dinâmico de micro-pullback
    taxa_atingida: bool = False
    menor_preco_tocou: float = float("inf")
    maior_preco_tocou: float = float("-inf")
    ticks_monitorados: int = 0
    criado_em: float = field(default_factory=time.time)


class MicroSniper:
    """Gerenciador de telemetria intracandle e disparo em taxa com desconto."""

    def __init__(self, ml_threshold: float = 0.55) -> None:
        self.ml_threshold = ml_threshold
        self._alvos: Dict[str, PreDecisaoSniper] = {}

    def registrar_pre_decisao(
        self,
        par: str,
        timeframe: str,
        direcao: str,
        vela_anterior: Dict[str, Any],
        preco_abertura: float,
        prob_win: float,
        lote: float,
        exp_minutos: int,
        duracao_vela_segundos: int = 900,
        features_row: Optional[pd.Series] = None,
    ) -> PreDecisaoSniper:
        """Arma o sniper para aguardar a retração na taxa favorável."""
        high_ant = float(vela_anterior.get("high", vela_anterior.get("max", preco_abertura)))
        low_ant = float(vela_anterior.get("low", vela_anterior.get("min", preco_abertura)))

        if direcao.upper() == "CALL":
            taxa_alvo = high_ant      # Pullback C1: antiga resistência virando suporte
            teto_invalida = low_ant   # Rompeu o fundo anterior = estrutura destruída
        else:
            taxa_alvo = low_ant       # Pullback C1: antigo suporte virando resistência
            teto_invalida = high_ant  # Rompeu o topo anterior = estrutura destruída

        max_espera = max(10.0, duracao_vela_segundos / 4.0)  # Limite estrito de 1/4 da vela

        alvo = PreDecisaoSniper(
            par=par,
            timeframe=timeframe,
            direcao=direcao.upper(),
            taxa_alvo=taxa_alvo,
            teto_invalida=teto_invalida,
            preco_abertura=preco_abertura,
            prob_win_inicial=prob_win,
            ts_inicio_vela=time.time(),
            max_segundos_espera=max_espera,
            exp_minutos=exp_minutos,
            lote=lote,
            features_row=features_row,
        )
        self._alvos[par] = alvo
        _LOG.info(
            "🎯 [SNIPER ARMADO] %s [%s] %s | Alvo C1: %.5f | Teto Invalidação: %.5f | Janela máx: %.0fs",
            par, timeframe, direcao, taxa_alvo, teto_invalida, max_espera
        )
        return alvo

    def processar_tick(
        self,
        par: str,
        preco_atual: float,
        ts_agora: Optional[float] = None,
    ) -> Tuple[str, Optional[PreDecisaoSniper], str]:
        """Avalia um tick de preço em tempo real contra as regras do Sniper.

        Retornos:
        - "WAITING": Aguardando taxa ou janela de reversão
        - "TRIGGER": Taxa atingida + 1º tick de rejeição confirmado!
        - "TIMEOUT": 1/4 do tempo da vela expirou sem retração suficiente
        - "INVALIDATED": Preço violou o teto estrutural (vela assassina)
        - "NONE": Par não está na mira do sniper
        """
        alvo = self._alvos.get(par)
        if not alvo:
            return "NONE", None, "Sem alvo ativo para o par."

        agora = ts_agora or time.time()
        tempo_decorrido = agora - alvo.ts_inicio_vela
        alvo.ticks_monitorados += 1

        # 1. Regra 5: Limite de 1/4 do tempo da vela (Timeout)
        if tempo_decorrido > alvo.max_segundos_espera:
            self._alvos.pop(par, None)
            motivo = f"Timeout de 1/4 atingido ({tempo_decorrido:.1f}s > {alvo.max_segundos_espera:.0f}s). Sem recuo à taxa."
            _LOG.info("⏱️ [SNIPER CANCELADO] %s: %s", par, motivo)
            return "TIMEOUT", alvo, motivo

        # 2. Regra 2: Teto Máximo Anti-Rompimento (Critério A)
        if alvo.direcao == "CALL" and preco_atual < alvo.teto_invalida:
            self._alvos.pop(par, None)
            motivo = f"Vela assassina violou fundo anterior (preço {preco_atual:.5f} < {alvo.teto_invalida:.5f})."
            _LOG.warning("🚫 [SNIPER INVALIDADO] %s: %s", par, motivo)
            return "INVALIDATED", alvo, motivo

        if alvo.direcao == "PUT" and preco_atual > alvo.teto_invalida:
            self._alvos.pop(par, None)
            motivo = f"Vela assassina violou topo anterior (preço {preco_atual:.5f} > {alvo.teto_invalida:.5f})."
            _LOG.warning("🚫 [SNIPER INVALIDADO] %s: %s", par, motivo)
            return "INVALIDATED", alvo, motivo

        # 3. Regra 1: Taxa Alvo C1 (Pullback) & Regra 3: 1º Tick Rejeição (Tipo B)
        if alvo.direcao == "CALL":
            # Para CALL, preço recuou até a taxa alvo ou abaixo dela
            if preco_atual <= alvo.taxa_alvo:
                if not alvo.taxa_atingida:
                    alvo.taxa_atingida = True
                    alvo.menor_preco_tocou = preco_atual
                    _LOG.debug("🎯 %s: Entrou na zona de desconto (preço %.5f <= alvo %.5f).", par, preco_atual, alvo.taxa_alvo)
                    return "WAITING", alvo, "Entrou na zona de taxa; aguardando 1º tick de rejeição."

                # Atualiza mínima atingida na retração
                if preco_atual < alvo.menor_preco_tocou:
                    alvo.menor_preco_tocou = preco_atual
                    return "WAITING", alvo, "Aprofundando mínima na taxa."

                # Primeiro tick de reversão a favor (preço subiu acima da mínima)
                if preco_atual > alvo.menor_preco_tocou:
                    self._alvos.pop(par, None)
                    motivo = f"Taxa de ouro atingida (mín: {alvo.menor_preco_tocou:.5f}) + 1º tick de alta ({preco_atual:.5f})!"
                    _LOG.info("⚡ [SNIPER GATILHO] %s: %s", par, motivo)
                    return "TRIGGER", alvo, motivo

        else:  # PUT
            # Para PUT, preço subiu até a taxa alvo ou acima dela
            if preco_atual >= alvo.taxa_alvo:
                if not alvo.taxa_atingida:
                    alvo.taxa_atingida = True
                    alvo.maior_preco_tocou = preco_atual
                    _LOG.debug("🎯 %s: Entrou na zona de prêmio (preço %.5f >= alvo %.5f).", par, preco_atual, alvo.taxa_alvo)
                    return "WAITING", alvo, "Entrou na zona de taxa; aguardando 1º tick de rejeição."

                # Atualiza máxima atingida na retração
                if preco_atual > alvo.maior_preco_tocou:
                    alvo.maior_preco_tocou = preco_atual
                    return "WAITING", alvo, "Aprofundando máxima na taxa."

                # Primeiro tick de reversão a favor (preço caiu abaixo da máxima)
                if preco_atual < alvo.maior_preco_tocou:
                    self._alvos.pop(par, None)
                    motivo = f"Taxa de ouro atingida (máx: {alvo.maior_preco_tocou:.5f}) + 1º tick de baixa ({preco_atual:.5f})!"
                    _LOG.info("⚡ [SNIPER GATILHO] %s: %s", par, motivo)
                    return "TRIGGER", alvo, motivo

        return "WAITING", alvo, "Monitorando recuo à taxa."

    def limpar_alvo(self, par: str) -> None:
        """Remove o alvo do par monitorado."""
        self._alvos.pop(par, None)

    def obter_alvo(self, par: str) -> Optional[PreDecisaoSniper]:
        """Retorna o alvo ativo do par se houver."""
        return self._alvos.get(par)

    @property
    def alvos_ativos(self) -> List[str]:
        """Lista de pares atualmente na mira do sniper."""
        return list(self._alvos.keys())
