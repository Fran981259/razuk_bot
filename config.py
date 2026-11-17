"""Módulo de configuração centralizada para o bot de trading quantitativo.

Este módulo gerencia e valida todas as variáveis de ambiente utilizando
pydantic-settings, garantindo tipagem estrita e integridade operacional.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AccountType = Literal["PRACTICE", "REAL"]
TimeFrame = Literal["M1", "M5", "M15", "M30", "H1"]
ExpirationMinutes = Literal[1, 5, 15, 30, 60]


class BaseConfigSection(BaseSettings):
    """Configuração base compartilhada para seções do bot."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )


class ConexaoConfig(BaseConfigSection):
    """Parâmetros de autenticação e comunicação com a corretora (IQ Option)."""

    api_key: str = Field(
        ...,
        validation_alias=AliasChoices("API_KEY", "IQ_EMAIL", "EMAIL", "IQOPTION_EMAIL"),
        description="Email de login da IQ Option ou chave de API.",
    )
    api_secret: str = Field(
        ...,
        validation_alias=AliasChoices("API_SECRET", "IQ_PASSWORD", "PASSWORD", "IQOPTION_PASSWORD"),
        description="Senha da conta IQ Option ou segredo de API.",
    )
    ssid: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("IQ_SSID", "SSID", "IQOPTION_SSID"),
        description="Token de sessão SSID pré-autenticado para ignorar login REST/2FA.",
    )
    account_type: AccountType = Field(
        default="PRACTICE",
        validation_alias=AliasChoices("ACCOUNT_TYPE", "IQ_ACCOUNT_TYPE", "IQOPTION_ACCOUNT"),
        description="Tipo de ambiente de execução: PRACTICE ou REAL.",
    )
    corretora_url: str = Field(
        default="https://auth.iqoption.com/api/v2/login",
        validation_alias=AliasChoices("CORRETORA_URL", "IQ_AUTH_URL"),
        description="URL base de autenticação REST da corretora.",
    )
    ws_url: str = Field(
        default="wss://iqoption.com/echo/websocket",
        validation_alias=AliasChoices("WS_URL", "IQ_WS_URL"),
        description="URL do gateway WebSocket da IQ Option.",
    )

    @property
    def email(self) -> str:
        """Alias legível para o email do usuário."""
        return self.api_key

    @property
    def password(self) -> str:
        """Alias legível para a senha do usuário."""
        return self.api_secret

    @field_validator("account_type", mode="before")
    @classmethod
    def normalizar_account_type(cls, valor: str) -> str:
        """Converte o tipo de conta para maiúsculas e remove espaços."""
        return valor.strip().upper() if isinstance(valor, str) else valor

    @field_validator("api_key", "api_secret", mode="after")
    @classmethod
    def validar_credenciais(cls, valor: str) -> str:
        """Garante que as chaves de API não estejam vazias."""
        if not valor.strip():
            raise ValueError("Chave de API/Email não pode ser vazia.")
        return valor.strip()

    @field_validator("corretora_url", mode="after")
    @classmethod
    def validar_url(cls, valor: str) -> str:
        """Valida se o protocolo da URL é HTTP ou HTTPS."""
        url_limpa = valor.strip()
        if not (url_limpa.startswith("http://") or url_limpa.startswith("https://")):
            raise ValueError("CORRETORA_URL deve iniciar com 'http://' ou 'https://'.")
        return url_limpa


class TradingConfig(BaseConfigSection):
    """Parâmetros operacionais e ativos a serem negociados."""

    timeframe: TimeFrame = Field(
        default="M5",
        alias="TIMEFRAME",
        description="Timeframe base para coleta de velas (M1, M5 — menor TF dos timeframes ativos).",
    )
    timeframes: List[str] = Field(
        default=["M5", "M15", "M30", "H1"],
        alias="TIMEFRAMES",
        description="Lista de timeframes para operação multi-TF (CSV ou JSON no .env).",
    )
    expiration_minutes: ExpirationMinutes = Field(
        default=5,
        alias="EXPIRATION_MINUTES",
        description="Expiração padrão em minutos quando não em modo multi-TF.",
    )
    max_simultaneous_trades: int = Field(
        default=5,
        alias="MAX_SIMULTANEOUS_TRADES",
        ge=1,
        le=10,
        description="Máximo de trades abertos simultaneamente em todos os timeframes (5 para acelerar coleta de dados em demo).",
    )
    pares_moedas: Union[List[str], str] = Field(
        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        alias="PARES_MOEDAS",
        description="Lista de pares de moedas para monitoramento e trade.",
    )
    min_payout: float = Field(
        default=0.84,
        alias="MIN_PAYOUT",
        ge=0.50,
        le=1.0,
        description="Payout mínimo exigido da corretora para autorizar entrada.",
    )
    sniper_mode: bool = Field(
        default=True,
        alias="SNIPER_MODE",
        description="Ativa entrada por Micro-Sniper na retração da taxa com rejeição.",
    )
    @field_validator("timeframe", mode="before")
    @classmethod
    def normalizar_timeframe(cls, valor: str) -> str:
        return valor.strip().upper() if isinstance(valor, str) else valor

    @field_validator("timeframes", mode="before")
    @classmethod
    def normalizar_timeframes(cls, valor) -> List[str]:
        _validos = {"M1", "M5", "M15", "M30", "H1"}
        if isinstance(valor, str):
            texto = valor.strip()
            if texto.startswith("["):
                import json as _json
                tfs = _json.loads(texto)
            else:
                tfs = [t.strip() for t in texto.split(",") if t.strip()]
        else:
            tfs = list(valor)
        result = [str(t).strip().upper() for t in tfs if str(t).strip().upper() in _validos]
        return result if result else ["M5", "M15", "M30", "H1"]

    @field_validator("expiration_minutes", mode="before")
    @classmethod
    def converter_expiration_minutes(cls, valor: Union[int, str]) -> int:
        """Converte o tempo de expiração para inteiro caso seja recebido como string."""
        return int(valor) if isinstance(valor, str) else valor

    @field_validator("pares_moedas", mode="after")
    @classmethod
    def validar_e_formatar_pares(cls, valor: Union[List[str], str]) -> List[str]:
        """Converte e normaliza lista de pares a partir de JSON ou CSV."""
        if isinstance(valor, str):
            texto = valor.strip()
            if texto.startswith("[") and texto.endswith("]"):
                pares = json.loads(texto)
            else:
                pares = [item.strip() for item in texto.split(",") if item.strip()]
        else:
            pares = valor

        pares_formatados = [str(p).strip().upper() for p in pares if str(p).strip()]
        if not pares_formatados:
            raise ValueError("PARES_MOEDAS não pode ser uma lista vazia.")
        return pares_formatados

class EstrategiaConfig(BaseConfigSection):
    """Parâmetros técnicos e períodos dos indicadores quantitativos da estratégia."""

    ema_fast: int = Field(default=9, alias="EMA_FAST", gt=0, description="Período da EMA rápida.")
    ema_med: int = Field(default=21, alias="EMA_MED", gt=0, description="Período da EMA intermediária (Onda).")
    ema_slow: int = Field(default=50, alias="EMA_SLOW", gt=0, description="Período da EMA lenta (Tendência).")
    ema_dist: int = Field(default=80, alias="EMA_DIST", gt=0, description="EMA de Afastamento Médio.")
    ema_sig: int = Field(default=5, alias="EMA_SIG", gt=0, description="Linha de sinal exponencial do Afastamento.")
    adx_period: int = Field(default=14, alias="ADX_PERIOD", gt=0, description="Período do oscilador ADX.")
    adx_trend_threshold: float = Field(
        default=25.0,
        alias="ADX_TREND_THRESHOLD",
        gt=0.0,
        description="ADX mínimo para confirmar mercado em tendência.",
    )
    adx_range_threshold: float = Field(
        default=20.0,
        alias="ADX_RANGE_THRESHOLD",
        gt=0.0,
        description="ADX máximo para confirmar mercado consolidado (range).",
    )
    atr_period: int = Field(default=20, alias="ATR_PERIOD", gt=0, description="Período do Average True Range.")
    bollinger_period: int = Field(default=20, alias="BOLLINGER_PERIOD", gt=0, description="Período de Bollinger.")
    bollinger_std: float = Field(default=2.0, alias="BOLLINGER_STD", gt=0.0, description="Desvio padrão de Bollinger.")
    max_pavio_total: float = Field(default=0.65, alias="MAX_PAVIO_TOTAL", ge=0.10, le=0.95, description="Pavio total máximo na vela (0.0 a 1.0).")
    max_pavio_rejeicao: float = Field(default=0.40, alias="MAX_PAVIO_REJEICAO", ge=0.10, le=0.90, description="Pavio de rejeição contrário máximo.")

    @model_validator(mode="after")
    def validar_hierarquia_emas(self) -> "EstrategiaConfig":
        """Valida a hierarquia lógica completa entre todas as médias móveis exponenciais."""
        if not (self.ema_fast < self.ema_med < self.ema_slow < self.ema_dist):
            raise ValueError(
                f"Hierarquia inválida: EMA_FAST ({self.ema_fast}) < EMA_MED ({self.ema_med}) "
                f"< EMA_SLOW ({self.ema_slow}) < EMA_DIST ({self.ema_dist}) é obrigatório."
            )
        if self.adx_range_threshold >= self.adx_trend_threshold:
            raise ValueError(
                f"ADX_RANGE_THRESHOLD ({self.adx_range_threshold}) deve ser menor que "
                f"ADX_TREND_THRESHOLD ({self.adx_trend_threshold})."
            )
        return self


class MachineLearningConfig(BaseConfigSection):
    """Configurações do pipeline de Machine Learning e inferência."""

    ml_threshold: float = Field(
        default=0.55,
        alias="ML_THRESHOLD",
        ge=0.50,
        le=0.95,
        description="Probabilidade mínima exigida pelo modelo para validar entradas (mínimo matemático: 1/(1+payout) ≈ 0.541 para payout=85%).",
    )
    alpha12_filter: bool = Field(
        default=False,
        alias="ALPHA12_FILTER",
        description="Ativa o filtro de absorção institucional Alpha101-12. Bloqueia entradas quando o fluxo de volume aponta contra a direção do trade.",
    )
    alpha12_threshold: float = Field(
        default=0.70,
        alias="ALPHA12_THRESHOLD",
        ge=0.10,
        le=2.0,
        description="Intensidade mínima do sinal Alpha101-12 para acionar o bloqueio (escala: 0.0–2.0). Padrão 0.70 = absorção moderada.",
    )
    model_path: Path = Field(
        default=Path("models/lgbm_model.joblib"),
        alias="MODEL_PATH",
        description="Caminho do arquivo serializado do modelo treinado.",
    )

    @field_validator("model_path", mode="after")
    @classmethod
    def validar_extensao_modelo(cls, caminho: Path) -> Path:
        """Garante que o caminho do modelo possua uma extensão válida de artefato."""
        extensoes_permitidas = {".joblib", ".pkl", ".pickle", ".model", ".bin"}
        if caminho.suffix.lower() not in extensoes_permitidas:
            raise ValueError(f"Extensão de MODEL_PATH inválida. Permitidas: {extensoes_permitidas}")
        return caminho

class GestaoRiscoConfig(BaseConfigSection):
    """Controles quantitativos de risco, dimensionamento e proteção de capital."""

    banca_inicial: float = Field(
        default=0.0,
        alias="BANCA_INICIAL",
        ge=0.0,
        description="Valor do capital inicial disponível na conta (0.0 = automático via corretora).",
    )
    stake_fixa: float = Field(
        default=2.0,
        alias="STAKE_FIXA",
        ge=0.0,
        description="Valor fixo por entrada em USD/BRL (0.0 = calcula via Kelly Fracionário).",
    )
    risk_per_trade_pct: float = Field(
        default=0.01,
        alias="RISK_PER_TRADE_PCT",
        gt=0.0,
        le=0.20,
        description="Risco percentual máximo por ordem (ex: 0.01 = 1%).",
    )
    kelly_fraction: float = Field(
        default=0.15,
        alias="KELLY_FRACTION",
        ge=0.05,
        le=0.50,
        description="Fração conservadora do Critério de Kelly (entre 0.05 e 0.50).",
    )
    max_daily_drawdown_pct: float = Field(
        default=0.05,
        alias="MAX_DAILY_DRAWDOWN_PCT",
        ge=0.01,
        le=0.20,
        description="Limite diário de rebaixamento da banca antes do bloqueio (1% a 20%).",
    )
    max_consecutive_losses: int = Field(
        default=3,
        alias="MAX_CONSECUTIVE_LOSSES",
        ge=1,
        le=20,
        description="Limite de derrotas consecutivas permitidas.",
    )
    cooldown_minutes_after_loss_streak: int = Field(
        default=60,
        alias="COOLDOWN_MINUTES_AFTER_LOSS_STREAK",
        ge=0,
        description="Tempo de espera em minutos após limite de derrotas consecutivas (60 min padrão para proteção em produção).",
    )

class BotConfig(BaseConfigSection):
    """Agregador raiz de todas as seções de configuração do bot de trading."""
    conexao: ConexaoConfig = Field(default_factory=ConexaoConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    estrategia: EstrategiaConfig = Field(default_factory=EstrategiaConfig)
    machine_learning: MachineLearningConfig = Field(default_factory=MachineLearningConfig)
    gestao_risco: GestaoRiscoConfig = Field(default_factory=GestaoRiscoConfig)

def load_config(env_file: Optional[str] = ".env") -> BotConfig:
    """Carrega as configurações instanciando cada seção a partir do arquivo .env."""
    kwargs = {"_env_file": env_file} if env_file else {}
    return BotConfig(
        conexao=ConexaoConfig(**kwargs), trading=TradingConfig(**kwargs),
        estrategia=EstrategiaConfig(**kwargs), machine_learning=MachineLearningConfig(**kwargs),
        gestao_risco=GestaoRiscoConfig(**kwargs),
    )

@lru_cache(maxsize=1)
def get_config() -> BotConfig:
    """Retorna uma instância única em cache das configurações do bot."""
    return load_config()
