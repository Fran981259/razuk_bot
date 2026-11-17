"""Constantes, identificadores e mapeamentos do protocolo da IQ Option.

Centraliza mapeamentos de pares para Active IDs, tipos de conta e períodos
conforme o protocolo WebSocket da corretora.
"""

from typing import Dict, Final

# Mapeamento dos principais pares de moedas para os Active IDs da IQ Option
# Inclui os IDs de Opções Binárias/Turbo (família 18xx) e OTC oficiais
ACTIVE_IDS: Final[Dict[str, int]] = {
    # Pares de mercado regular (Active IDs de Opções ativas na IQ Option)
    "EURUSD": 1861,
    "EURJPY": 1864,
    "USDJPY": 1865,
    "GBPUSD": 1867,
    "GBPJPY": 1866,
    "EURGBP": 1862,
    "USDCAD": 1878,
    "EURCAD": 1875,
    "NZDUSD": 1896,
    "AUDUSD": 1870,
    "AUDCAD": 6,
    "USDCHF": 72,
    "AUDJPY": 102,
    # Pares OTC (Over-The-Counter)
    "EURUSD-OTC": 76,
    "EURGBP-OTC": 77,
    "USDCHF-OTC": 78,
    "EURJPY-OTC": 79,
    "NZDUSD-OTC": 80,
    "GBPUSD-OTC": 81,
    "AUDUSD-OTC": 2111,
    "AUDCAD-OTC": 85,
    "USDCAD-OTC": 86,
}

# Tipos de saldo na plataforma IQ Option
BALANCE_TYPE_REAL: Final[int] = 1
BALANCE_TYPE_PRACTICE: Final[int] = 4

# Duração das velas (timeframes) em segundos
TIMEFRAME_SECONDS: Final[Dict[str, int]] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
}

# Tipos de opções
OPTION_TURBO: Final[int] = 3  # Expirações curtas (< 5 min)
OPTION_BINARY: Final[int] = 1  # Expirações padrão


def get_active_id(par: str) -> int:
    """Retorna o ID numérico do ativo na IQ Option a partir de seu ticker.

    Args:
        par: Nome do par de moedas (ex: 'EURUSD', 'EURUSD-OTC').

    Returns:
        Identificador inteiro do ativo.

    Raises:
        KeyError: Se o par não for suportado no mapeamento.
    """
    par_normalizado = par.replace("/", "").replace("_", "-").strip().upper()
    if par_normalizado in ACTIVE_IDS:
        return ACTIVE_IDS[par_normalizado]
    alt = par_normalizado.replace("-", "")
    for k, v in ACTIVE_IDS.items():
        if k.replace("-", "") == alt:
            return v
    raise KeyError(f"Par de moedas '{par}' não encontrado no catálogo da IQ Option.")
