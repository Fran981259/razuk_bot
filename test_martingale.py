"""Bateria de testes unitários da estratégia de Martingale do Razuk Bot."""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from config import BotConfig, GestaoRiscoConfig
from risk import RiskManager


class TestMartingaleStrategy(unittest.TestCase):
    """Testes de dimensionamento e transição de estado da estratégia Martingale."""

    def setUp(self):
        self.config_risco = GestaoRiscoConfig(
            banca_inicial=4000.0,
            stake_fixa=0.0,
            martingale_enabled=True,
            martingale_base_pct=0.05,
            martingale_max_levels=2,
            martingale_multiplier=2.2,
            max_daily_drawdown_pct=0.40,
        )
        self.risk = RiskManager(self.config_risco)
        self.risk.sync_balance(4000.0)

    def test_base_stake_is_5_percent(self):
        """Verifica se a mão inicial é rigorosamente 5% da banca."""
        lote = self.risk.calculate_position_size(win_probability=0.60, payout=0.85)
        self.assertEqual(lote, 200.0)  # 5% de 4000.00 = 200.00

    def test_gale_multiplier_progression(self):
        """Verifica progressão correta dos lotes em Gale 1 e Gale 2 com multiplicador 2.2x."""
        lote_base = self.risk.calculate_position_size(win_probability=0.60, payout=0.85)
        self.assertEqual(lote_base, 200.0)

        lote_gale1 = self.risk.calculate_gale_stake(lote_base)
        self.assertEqual(lote_gale1, 440.0)  # 200 * 2.2 = 440.0

        lote_gale2 = self.risk.calculate_gale_stake(lote_gale1)
        self.assertEqual(lote_gale2, 968.0)  # 440 * 2.2 = 968.0

    def test_win_on_base_trade_keeps_drawdown_zero(self):
        """Verifica registro de vitória na mão base."""
        self.risk.register_trade_result(pnl=170.0, was_win=True)
        self.assertEqual(self.risk.banca_atual, 4170.0)
        self.assertEqual(self.risk.consecutive_losses, 0)
        pode, _ = self.risk.can_trade()
        self.assertTrue(pode)

    def test_stop_loss_trigger_after_gale2_loss(self):
        """Verifica proteção quando Mão Base + Gale 1 + Gale 2 resultam em Loss."""
        # 1. Loss Mão Base (-$200)
        self.risk.register_trade_result(pnl=-200.0, was_win=False)
        self.assertEqual(self.risk.banca_atual, 3800.0)
        pode, _ = self.risk.can_trade()
        self.assertTrue(pode)  # Drawdown 5% < 40%

        # 2. Loss Gale 1 (-$440)
        self.risk.register_trade_result(pnl=-440.0, was_win=False)
        self.assertEqual(self.risk.banca_atual, 3360.0)
        pode, _ = self.risk.can_trade()
        self.assertTrue(pode)  # Drawdown 16% < 40%

        # 3. Loss Gale 2 (-$968)
        self.risk.register_trade_result(pnl=-968.0, was_win=False)
        self.assertEqual(self.risk.banca_atual, 2392.0)
        # Total perdido: 1608 / 4000 = 40.2% -> Deve estar bloqueado por Drawdown ou Cooldown
        pode, motivo = self.risk.can_trade()
        self.assertFalse(pode)
        self.assertTrue("Stop Loss por Drawdown" in motivo or "Cooldown" in motivo)


class TestMartingaleExecutionLoop(unittest.IsolatedAsyncioTestCase):
    """Testa o loop de execução assíncrona do Martingale no BotOrchestrator."""

    async def test_martingale_loop_recovery_on_gale1(self):
        from main import BotOrchestrator
        from dummy_conector import DummyConector

        config = BotConfig()
        config.gestao_risco.martingale_enabled = True
        config.gestao_risco.martingale_base_pct = 0.05
        config.gestao_risco.martingale_max_levels = 2
        config.gestao_risco.martingale_multiplier = 2.2
        config.gestao_risco.max_daily_drawdown_pct = 0.40

        conector = DummyConector()
        orchestrator = BotOrchestrator(config=config)
        orchestrator.conector = conector
        orchestrator.risk.sync_balance(1000.0)

        # Simula: Mão base perde, Gale 1 ganha
        # Mock do buy para retornar ordem
        conector.buy = AsyncMock(side_effect=[
            {"id": "ordem_base", "exp_timestamp": 100},
            {"id": "ordem_gale1", "exp_timestamp": 200},
        ])
        # Mock do wait_for_option_result: 1º loss, 2º win
        conector.wait_for_option_result = AsyncMock(side_effect=[
            {"win": "loose", "profit_amount": 0.0, "amount": 50.0},
            {"win": "win", "profit_amount": 203.5, "amount": 110.0},
        ])
        conector.get_balance = AsyncMock(return_value=1043.5)

        # Executa ordem de teste
        await orchestrator._execute_and_await_trade(
            par="EURUSD",
            direcao="CALL",
            lote=50.0,
            prob_win=0.60,
            expiracao_minutos=5
        )

        # Deve ter executado exatamente 2 compras (mão base + gale 1)
        self.assertEqual(conector.buy.call_count, 2)
        # 1ª compra: lote 50.0 (5% de 1000)
        self.assertEqual(conector.buy.call_args_list[0].kwargs["valor"], 50.0)
        # 2ª compra: lote 110.0 (50 * 2.2)
        self.assertEqual(conector.buy.call_args_list[1].kwargs["valor"], 110.0)
        # Mesma direção preservada
        self.assertEqual(conector.buy.call_args_list[1].kwargs["direcao"], "CALL")
        # _pending_trades zerado no final
        self.assertEqual(orchestrator._pending_trades, 0)


if __name__ == "__main__":
    unittest.main()
