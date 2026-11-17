"""Módulo centralizado de hierarquia de erros e exceções tipadas do bot.

Implementado conforme a skill 'error-handling' do projeto.
Permite captura precisa de falhas de conexão, recusas da corretora,
erros de cálculo de features e integridade de inferência de ML.
"""

from typing import Any, Dict, Optional


class BotError(Exception):
    """Exceção base para todas as falhas de domínio do bot."""

    def __init__(self, message: str, code: str = "BOT_ERROR", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class BrokerConnectionError(BotError):
    """Falha de conexão física ou queda de sessão WebSocket/REST com a corretora."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="BROKER_CONNECTION_ERROR", details=details)


class BrokerOrderRefusalError(BotError):
    """Ordem recusada pela corretora (ativo suspenso, fechado, payout indisponível)."""

    def __init__(self, par: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        msg = f"Ordem recusada em {par}: {reason}"
        det = details or {}
        det.update({"par": par, "reason": reason})
        super().__init__(msg, code="BROKER_ORDER_REFUSAL", details=det)
        self.par = par
        self.reason = reason


class BrokerTimeoutError(BotError):
    """Tempo limite excedido aguardando resposta assíncrona da corretora."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        msg = f"Timeout de {timeout_seconds:.1f}s excedido na operação '{operation}'"
        super().__init__(msg, code="BROKER_TIMEOUT", details={"operation": operation, "timeout": timeout_seconds})


class FeatureCalculationError(BotError):
    """Falha matemática ou insuficiência de dados ao gerar variáveis quantitativas."""

    def __init__(self, par: str, reason: str) -> None:
        msg = f"Falha no cálculo de features para {par}: {reason}"
        super().__init__(msg, code="FEATURE_CALCULATION_ERROR", details={"par": par, "reason": reason})


class ModelInferenceError(BotError):
    """Falha na predição P(Win) estatística do modelo LightGBM."""

    def __init__(self, par: str, reason: str) -> None:
        msg = f"Falha na inferência de ML para {par}: {reason}"
        super().__init__(msg, code="MODEL_INFERENCE_ERROR", details={"par": par, "reason": reason})
