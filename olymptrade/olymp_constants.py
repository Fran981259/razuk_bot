"""Constantes e mapeamentos de mercado para a corretora Olymp Trade."""

from typing import Dict

# URL padrão de conexão WebSocket (Olymp Trade FTT)
DEFAULT_WS_URL = (
    "wss://ws.olymptrade.com/otp?cid_ver=1"
    "&cid_app=web%40OlympTrade%402025.3.26878%4026878"
    "&cid_device=%40%40desktop&cid_os=linux%406.8.0"
)

# Mapeamento de timeframes em segundos
TIMEFRAME_SECONDS: Dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

# Normalização de direção: Razuk Bot ("CALL"/"PUT") -> Olymp Trade ("up"/"down")
DIRECTION_MAP: Dict[str, str] = {
    "CALL": "up",
    "PUT": "down",
    "call": "up",
    "put": "down",
    "up": "up",
    "down": "down",
}

# Inverso de direção para leitura de resultados
REVERSE_DIRECTION_MAP: Dict[str, str] = {
    "up": "CALL",
    "down": "PUT",
}

# Mapeamento de tipo de conta
ACCOUNT_TYPE_MAP: Dict[str, str] = {
    "PRACTICE": "demo",
    "DEMO": "demo",
    "REAL": "live",
    "LIVE": "live",
}

# Mapeamento de nomes de pares (Razuk Bot -> Olymp Trade)
PAIR_ALIASES: Dict[str, str] = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD",
    "EURJPY": "EURJPY",
    "USDCAD": "USDCAD",
    "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD",
    "EURGBP": "EURGBP",
    "AUDCAD": "AUDCAD",
    # Mapeamento OTC / Composites se aplicável
    "EURUSD-OTC": "EURUSD",
    "AUDUSD-OTC": "AUDUSD",
    "EURJPY-OTC": "EURJPY",
    "USDCHF-OTC": "USDCHF",
}


def normalize_pair(par: str) -> str:
    """Normaliza o nome do par para o formato aceito pela Olymp Trade."""
    clean = par.replace("/", "").replace(" ", "").upper()
    return PAIR_ALIASES.get(clean, clean)


def normalize_direction(direcao: str) -> str:
    """Converte direção padrão (CALL/PUT) para formato Olymp Trade (up/down)."""
    return DIRECTION_MAP.get(direcao.strip(), "up")
