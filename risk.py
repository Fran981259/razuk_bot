"""Módulo de Gestão de Risco e Dimensionamento de Lote (RiskManager).

Implementa proteção estrita de capital com Critério de Kelly Fracionário,
tetos percentuais por trade, controle de drawdown diário e cooldown automático
após sequência de derrotas consecutivas.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from config import GestaoRiscoConfig
from cvar_risk import compute_cvar_kelly_damping

# Valor mínimo de entrada aceito pela corretora
_ENTRADA_MINIMA: float = 2.00


class RiskManager:
    """Gerenciador quantitativo de risco operacional e proteção de banca.

    Controla todas as decisões financeiras do bot: autorização de abertura de
    ordens, dimensionamento por Kelly Fracionário e registro dos resultados.
    """

    def __init__(self, config: Optional[GestaoRiscoConfig] = None) -> None:
        """Inicializa o gerenciador com a configuração de risco fornecida.

        Args:
            config: Instância de GestaoRiscoConfig. Se None, usa os padrões.
        """
        self._config = config or GestaoRiscoConfig()
        self._logger = logging.getLogger("risk_manager")

        self.banca_inicial: float = float(self._config.banca_inicial)
        self.banca_atual: float = self.banca_inicial
        self.lucro_diario: float = 0.0
        self.consecutive_losses: int = 0
        self.bloqueado_ate: Optional[datetime] = None
        self._trade_history: List[float] = []

        if self.banca_inicial > 0.0:
            self._logger.info(
                "RiskManager inicializado | Banca: R$%.2f | Risco/trade: %.1f%% | "
                "Kelly: %.0f%% | Drawdown máx: %.1f%%",
                self.banca_atual,
                self._config.risk_per_trade_pct * 100,
                self._config.kelly_fraction * 100,
                self._config.max_daily_drawdown_pct * 100,
            )
        else:
            self._logger.info(
                "RiskManager inicializado em modo AUTO | Aguardando leitura do saldo real da corretora..."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def drawdown_diario_pct(self) -> float:
        """Percentual de perda acumulada no dia em relação à banca inicial.

        Returns:
            Valor entre 0.0 e 1.0. Retorna 0.0 se o dia estiver no lucro.
        """
        if self.lucro_diario < 0.0 and self.banca_inicial > 0.0:
            return abs(self.lucro_diario) / self.banca_inicial
        return 0.0

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def can_trade(self) -> Tuple[bool, str]:
        """Avalia se novas ordens estão autorizadas pelas diretrizes de risco."""
        if self.bloqueado_ate and datetime.now() < self.bloqueado_ate:
            restante = int((self.bloqueado_ate - datetime.now()).total_seconds() / 60) + 1
            return False, f"BLOQUEADO — Cooldown ativo ({restante}m restantes após {self._config.max_consecutive_losses} losses seguidos)."

        # 1. Stop Loss por Porcentagem de Drawdown da Banca
        if self.drawdown_diario_pct >= self._config.max_daily_drawdown_pct:
            return False, f"BLOQUEADO — Stop Loss por Drawdown ({self.drawdown_diario_pct:.2%} >= {self._config.max_daily_drawdown_pct:.2%})."

        # 2. Saldo insuficiente para cobrir entrada mínima
        if self.banca_atual < _ENTRADA_MINIMA:
            return False, f"BLOQUEADO — Saldo insuficiente: R${self.banca_atual:.2f} (mínimo: R${_ENTRADA_MINIMA:.2f})."

        return True, f"LIBERADO | Banca: R${self.banca_atual:.2f} | Drawdown: {self.drawdown_diario_pct:.2%}."

    def calculate_position_size(self, win_probability: float, payout: float) -> float:
        """Calcula o valor de entrada via Critério de Kelly Fracionário.

        Fórmula:  f* = (p × (b + 1) − 1) / b
        Onde `p` é a probabilidade de vitória e `b` é o payout decimal.
        O valor final é multiplicado pela KELLY_FRACTION e limitado ao
        RISK_PER_TRADE_PCT. Garante o mínimo de R$2,00.

        Args:
            win_probability: Probabilidade estimada pelo modelo ML (0.0–1.0).
            payout: Retorno líquido decimal da corretora (ex: 0.85 = 85%).

        Returns:
            Valor monetário da ordem, arredondado em 2 casas decimais.
            Retorna 0.0 se a expectativa matemática for nula ou negativa.
        """
        if payout <= 0.0 or win_probability <= 0.0:
            self._logger.debug("Parâmetros inválidos: prob=%.4f payout=%.4f", win_probability, payout)
            return 0.0

        p, b = win_probability, payout

        # Se houver valor fixo configurado (ex: $2.00 para treino), respeita o valor fixo
        if getattr(self._config, "stake_fixa", 0.0) > 0.0:
            lote_fixo = float(self._config.stake_fixa)
            self._logger.info(
                "Stake Fixa (Modo Treino) | Prob=%.2f%% | Payout=%.2f%% | Lote=$%.2f",
                p * 100, b * 100, lote_fixo
            )
            return lote_fixo

        f_star = (p * (b + 1.0) - 1.0) / b
        if f_star <= 0.0:
            self._logger.debug(
                "Expectativa matemática negativa (f*=%.4f). Lote zero.", f_star
            )
            return 0.0

        # Fração conservadora de Kelly ajustada por CVaR (Vibe-Trading Tail Risk Guard)
        damping, cvar = compute_cvar_kelly_damping(self._trade_history)
        f_kelly = f_star * self._config.kelly_fraction * damping
        f_final = min(f_kelly, self._config.risk_per_trade_pct)

        tamanho_bruto = self.banca_atual * f_final
        # Garante o valor mínimo de entrada aceito pela corretora
        tamanho_ordem = round(max(tamanho_bruto, _ENTRADA_MINIMA), 2)

        self._logger.info(
            "Dimensionamento | Prob=%.2f%% | Payout=%.2f%% | f*=%.4f | "
            "CVaR=%.1f%% (Damp=%.2f) | Kelly=%.2f%% | Lote=R$%.2f",
            p * 100,
            b * 100,
            f_star,
            cvar * 100,
            damping,
            f_final * 100,
            tamanho_ordem,
        )
        return tamanho_ordem

    def register_trade_result(self, pnl: float, was_win: bool) -> None:
        """Registra o resultado financeiro de uma operação concluída.

        Atualiza banca, lucro diário e ativa o cooldown se necessário.

        Args:
            pnl: Lucro (positivo) ou prejuízo (negativo) da operação em R$.
            was_win: True se a ordem foi vencedora, False caso contrário.
        """
        ret = pnl / self.banca_atual if self.banca_atual > 0 else 0.0
        self._trade_history.append(ret)

        self.banca_atual += pnl
        self.lucro_diario += pnl

        if was_win:
            self.consecutive_losses = 0
            self._logger.info(
                "✅ Trade VENCEDOR | PnL: +R$%.2f | Banca: R$%.2f | Lucro diário: R$%.2f",
                pnl,
                self.banca_atual,
                self.lucro_diario,
            )
        else:
            self.consecutive_losses += 1
            self._logger.warning(
                "❌ Trade PERDEDOR | PnL: -R$%.2f | Banca: R$%.2f | Drawdown atual: %.2f%%",
                abs(pnl), self.banca_atual, self.drawdown_diario_pct * 100
            )
            # Trava de cooldown por perdas consecutivas (desativada quando cooldown_minutes_after_loss_streak == 0)
            if (
                self._config.cooldown_minutes_after_loss_streak > 0
                and self.consecutive_losses >= self._config.max_consecutive_losses
            ):
                cd_min = self._config.cooldown_minutes_after_loss_streak
                self.bloqueado_ate = datetime.now() + timedelta(minutes=cd_min)
                self._logger.warning(
                    "🚨 %d perdas consecutivas! Cooldown de %d min ativado até %s.",
                    self.consecutive_losses, cd_min, self.bloqueado_ate.strftime("%H:%M:%S")
                )

    def sync_balance(self, saldo_real: float) -> None:
        """Sincroniza a banca do RiskManager com o saldo real da corretora.

        Se a banca inicial for 0.0 (modo automático) ou se for a primeira
        sincronização da sessão, estabelece a banca_inicial com o saldo real.

        Args:
            saldo_real: Saldo retornado pela API da corretora em USD/BRL.
        """
        if saldo_real > 0.0:
            if self.banca_inicial <= 0.0 or (abs(self.banca_atual - self.banca_inicial) < 1e-4 and self.lucro_diario == 0.0):
                self.banca_inicial = float(saldo_real)
            self.banca_atual = float(saldo_real)
            self._logger.info(
                "RiskManager sincronizado com saldo real da corretora: R$%.2f (Base inicial: R$%.2f)",
                self.banca_atual, self.banca_inicial
            )

    def reset_daily(self) -> None:
        """Reinicia contadores e métricas para o novo dia operacional.

        A banca inicial diária é atualizada para o saldo atual,
        garantindo que o drawdown seja calculado sobre o novo baseline.
        """
        self.banca_inicial = self.banca_atual
        self.lucro_diario = 0.0
        self.consecutive_losses = 0
        self.bloqueado_ate = None
        self._logger.info(
            "📅 Reinício diário | Nova base: R$%.2f", self.banca_inicial
        )



# ----------------------------------------------------------------------
# Teste funcional — demonstra todos os estados do gerenciador
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 65)
    print("RiskManager — Teste de Integração Funcional")
    print("=" * 65)

    # Configuração de teste: banca R$500, drawdown max 5%, cooldown 1 min
    cfg = GestaoRiscoConfig(
        banca_inicial=500.0,
        risk_per_trade_pct=0.02,
        kelly_fraction=0.15,
        max_daily_drawdown_pct=0.05,
        max_consecutive_losses=3,
        cooldown_minutes_after_loss_streak=1,
        stake_fixa=0.0,
    )
    rm = RiskManager(config=cfg)

    print("\n--- [1] Verificação inicial de permissão ---")
    ok, motivo = rm.can_trade()
    print(f"  can_trade() → {ok} | {motivo}")
    assert ok, "FALHA: deveria estar liberado na inicialização"

    print("\n--- [2] Cálculo de dimensionamento de lote ---")
    lote = rm.calculate_position_size(win_probability=0.70, payout=0.85)
    print(f"  Prob=70% | Payout=85% → Lote=R${lote:.2f}")
    assert lote >= _ENTRADA_MINIMA, f"FALHA: lote R${lote} abaixo do mínimo"

    lote_neg = rm.calculate_position_size(win_probability=0.40, payout=0.85)
    print(f"  Prob=40% | Payout=85% → Lote=R${lote_neg:.2f} (expectativa negativa)")
    assert lote_neg == 0.0, "FALHA: expectativa negativa deveria retornar 0.0"

    print("\n--- [3] Simulação de 3 perdas consecutivas (trava de cooldown) ---")
    for i in range(1, 4):
        entrada = 10.0
        rm.register_trade_result(pnl=-entrada, was_win=False)
        ok, motivo = rm.can_trade()
        print(f"  Perda #{i} | can_trade()={ok} | {motivo[:70]}...")

    print("\n--- [4] Verificação do drawdown acumulado ---")
    print(f"  lucro_diario:       R${rm.lucro_diario:.2f}")
    print(f"  drawdown_diario_pct: {rm.drawdown_diario_pct:.2%}")
    print(f"  banca_atual:        R${rm.banca_atual:.2f}")
    assert rm.bloqueado_ate is not None, "FALHA: cooldown deveria estar ativo"

    print("\n--- [5] Registro de vitória após reset do cooldown manual ---")
    rm.bloqueado_ate = None
    rm.consecutive_losses = 0
    rm.register_trade_result(pnl=25.0, was_win=True)
    ok, motivo = rm.can_trade()
    print(f"  Após vitória → can_trade()={ok} | {motivo}")
    assert ok, "FALHA: deveria estar liberado após vitória"

    print("\n" + "=" * 65)
    print("  ✅ TODOS OS TESTES PASSARAM — RiskManager operacional.")
    print("=" * 65 + "\n")
