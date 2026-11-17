"""Coletor de dados históricos de velas OHLCV para treinamento e backtest.

Realiza paginação retroativa para coletar 90 dias de histórico em lotes,
com suporte ao conector oficial da corretora e fallback para simulação offline.
"""

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from config import get_config
from dummy_conector import DummyConector

try:
    from conector import IqOptionClient
except ImportError:
    try:
        from iq_client import IqOptionClient  # type: ignore
    except ImportError:
        IqOptionClient = None


class CandleCollector:
    """Coletor assíncrono paginado de histórico de velas de longo prazo."""

    def __init__(self, dias: int = 90, timeframe: str = "M1") -> None:
        """Inicializa o coletor com horizonte temporal e timeframe especificados."""
        self.dias = dias
        self.timeframe = timeframe.upper()
        self.tf_seg = 60 if self.timeframe == "M1" else 300
        self.logger = logging.getLogger("candle_collector")

        # Inicializa cliente da corretora ou fallback
        self.conector = None
        if IqOptionClient is not None:
            try:
                cfg = get_config()
                self.conector = IqOptionClient(cfg.conexao)
            except Exception:
                self.conector = None

        if self.conector is None:
            self.logger.info("Usando DummyConector para geração sintética de 90 dias.")
            self.conector = DummyConector()

    async def collect(self, par: str = "EURUSD-OTC") -> Path:
        """Coleta 90 dias de velas retroativas e salva em arquivo CSV."""
        self.logger.info("Iniciando coleta de %d dias de velas para %s (%s)...", self.dias, par, self.timeframe)
        try:
            await self.conector.connect()
        except Exception as exc:
            self.logger.warning("Falha na conexão oficial (%s). Alternando para DummyConector.", exc)
            self.conector = DummyConector()
            await self.conector.connect()

        agora = int(time.time())
        limite_passado = agora - (self.dias * 86400)
        cursor_tempo = agora
        todas_velas: List[Dict[str, Any]] = []

        batch_count = 1000
        lote_num = 1

        while cursor_tempo > limite_passado:
            self.logger.info(
                "Lote %02d: buscando até %s (restam %.1f dias)...",
                lote_num,
                datetime.fromtimestamp(cursor_tempo, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                max(0.0, (cursor_tempo - limite_passado) / 86400),
            )
            velas_lote = None
            for tentativa in range(1, 4):
                try:
                    velas_lote = await self.conector.get_candles(
                        par=par, timeframe=self.timeframe, contagem=batch_count, end_time=cursor_tempo
                    )
                    if velas_lote:
                        break
                except Exception as exc:
                    self.logger.debug("Tentativa %d falhou no lote %d para %s: %s", tentativa, lote_num, par, exc)
                await asyncio.sleep(0.3 * tentativa)

            if not velas_lote:
                self.logger.warning("Nenhuma vela retornada no lote %d após retentativas. Encerrando coleta para %s.", lote_num, par)
                break

            todas_velas.extend(velas_lote)

            # Identifica o timestamp mais antigo do lote para paginar para trás
            timestamps = [int(v.get("from", v.get("at", cursor_tempo))) for v in velas_lote]
            menor_ts = min(timestamps)

            if menor_ts >= cursor_tempo:
                # Evita loop infinito se o conector retornar o mesmo timestamp
                cursor_tempo -= (batch_count * self.tf_seg)
            else:
                cursor_tempo = menor_ts - 1

            lote_num += 1
            # Pausa leve para respeitar taxa do gateway da corretora
            await asyncio.sleep(0.08)

        await self.conector.close()

        # Tratamento e consolidação segura dos dados
        if not todas_velas:
            self.logger.warning("Nenhuma vela foi obtida para o par %s.", par)
            return None

        df = pd.DataFrame(todas_velas)
        if "min" in df.columns and "low" not in df.columns:
            df = df.rename(columns={"min": "low", "max": "high"})

        if "from" not in df.columns:
            self.logger.warning("Dados retornados não contêm timestamp 'from' para %s.", par)
            return None

        colunas_padrao = ["from", "open", "high", "low", "close", "volume"]
        cols_existentes = [c for c in colunas_padrao if c in df.columns]
        df = df[cols_existentes].drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)

        # Adiciona coluna de data formatada em ISO
        df["datetime"] = pd.to_datetime(df["from"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

        pasta_saida = Path("data")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        par_limpo = par.replace("/", "_").replace("-", "_")
        caminho_csv = pasta_saida / f"candles_{par_limpo}_{self.timeframe}_{self.dias}d.csv"

        df.to_csv(caminho_csv, index=False)
        self.logger.info("✅ Coleta finalizada: %d velas gravadas em '%s'", len(df), caminho_csv)
        self.logger.info(
            "Período coberto: de %s a %s",
            df["datetime"].iloc[0], df["datetime"].iloc[-1]
        )
        return caminho_csv


async def collect_missing_pairs(
    dias: int = 90,
    timeframe: Optional[str] = None,
    pares: Optional[List[str]] = None,
) -> List[Path]:
    """Coleta histórico para todos os pares configurados que ainda não possuem dataset local."""
    pasta_data = Path("data")
    cfg = get_config()
    tf = (timeframe or cfg.trading.timeframe).upper()
    pares_alvo = pares or (list(cfg.trading.pares_moedas) if cfg.trading.pares_moedas else ["EURUSD", "EURJPY", "AUDUSD", "AUDUSD-OTC", "EURJPY-OTC", "USDCHF-OTC", "EURUSD-OTC"])
    arquivos_gerados: List[Path] = []

    for par in pares_alvo:
        par_limpo = par.replace("/", "_").replace("-", "_")
        esperado = pasta_data / f"candles_{par_limpo}_{tf}_{dias}d.csv"
        if esperado.exists() and esperado.stat().st_size > 30000:
            logging.info("⏩ Dataset já existe para %s (%s). Pulando.", par, esperado.name)
            arquivos_gerados.append(esperado)
            continue

        coletor = CandleCollector(dias=dias, timeframe=tf)
        res = await coletor.collect(par=par)
        if res:
            arquivos_gerados.append(res)
        await asyncio.sleep(1.0)

    return arquivos_gerados


async def main() -> None:
    """Ponto de entrada CLI para coleta de velas."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = get_config()
    dias = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    pares = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else None
    timeframe = sys.argv[3].upper() if len(sys.argv) > 3 else cfg.trading.timeframe

    if pares:
        for p in pares:
            coletor = CandleCollector(dias=dias, timeframe=timeframe)
            await coletor.collect(par=p.strip())
            await asyncio.sleep(1.0)
    else:
        await collect_missing_pairs(dias=dias, timeframe=timeframe)


if __name__ == "__main__":
    asyncio.run(main())
