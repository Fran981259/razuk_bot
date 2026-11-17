"""Conector mock de demonstração e fallback para o bot de trading.

Permite testes offline, validação de fluxo assíncrono e simulação sem
necessidade de credenciais ativas da corretora.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np


class DummyConector:
    """Conector fallback simulado para testes e execução offline."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.is_connected: bool = False
        self._balance: float = 1000.0
        self._logger = logging.getLogger("dummy_conector")

    async def connect(self) -> bool:
        """Simula conexão WebSocket com a corretora."""
        self.is_connected = True
        self._logger.info("DummyConector ativo em simulação (Saldo: $%.2f).", self._balance)
        return True

    async def get_balance(self) -> float:
        """Retorna o saldo simulado da conta."""
        return self._balance

    async def get_candles(
        self,
        par: str,
        timeframe: str = "M1",
        contagem: int = 100,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Gera velas sintéticas com as colunas OHLCV necessárias."""
        fim = end_time if end_time is not None else int(time.time())
        agora = fim - (contagem * 60)
        p, velas = 1.1000, []
        for i in range(contagem):
            c = p + float(np.random.normal(0, 0.0003))
            velas.append({
                "from": agora + (i * 60),
                "open": round(p, 5),
                "close": round(c, 5),
                "high": round(max(p, c) + 0.0002, 5),
                "low": round(min(p, c) - 0.0002, 5),
                "volume": float(np.random.randint(50, 400)),
            })
            p = c
        return velas

    async def buy(
        self, par: str, valor: float, direcao: str, expiracao_minutos: int = 1
    ) -> Dict[str, Any]:
        """Simula abertura de ordem."""
        self._logger.info("DummyConector: Ordem %s de R$%.2f em %s.", direcao.upper(), valor, par)
        return {"status": "ok", "id": int(time.time()), "amount": valor, "direction": direcao}

    def get_payout(self, par: str) -> float:
        """Retorna o payout simulado (padrão: 86%)."""
        return 0.86

    async def get_current_price(self, par: str) -> float:
        """Retorna cotação simulada em tempo real."""
        return 1.1000

    async def subscribe_candle(self, par: str) -> None:
        """Stub fallback de subscrição de streaming."""
        pass

    async def unsubscribe_candle(self, par: str) -> None:
        """Stub fallback de cancelamento de subscrição."""
        pass

    async def wait_for_option_result(self, order_id: Any, timeout: float = 70.0) -> Optional[Dict[str, Any]]:
        """Stub fallback para simulações offline sem evento de WebSocket."""
        return None

    async def close(self) -> None:
        """Encerra a conexão simulada."""
        self.is_connected = False
        self._logger.info("DummyConector desconectado.")
