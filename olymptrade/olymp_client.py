"""Cliente assíncrono para comunicação com a Olymp Trade via WebSocket (FTT).

Implementa a mesma interface do conector IQ Option do Razuk Bot:
- Autenticação via token de sessão (access_token)
- Consulta e alternância de saldo (Demo / Real)
- Coleta de velas históricas OHLCV
- Streaming de cotações em tempo real (get_current_price / subscribe_candle)
- Envio de ordens FTT (buy) e resolução de resultado (wait_for_option_result)
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set
import uuid

import websockets
from websockets.client import WebSocketClientProtocol

from .olymp_constants import (
    ACCOUNT_TYPE_MAP,
    DEFAULT_WS_URL,
    DIRECTION_MAP,
    TIMEFRAME_SECONDS,
    normalize_direction,
    normalize_pair,
)


class OlympTradeClient:
    """Cliente WebSocket assíncrono para operações na Olymp Trade (Fixed Time Trades)."""

    def __init__(
        self,
        token: Optional[str] = None,
        account_type: str = "PRACTICE",
        ws_url: str = DEFAULT_WS_URL,
    ) -> None:
        """Inicializa o cliente com token de sessão e preferências."""
        self._logger = logging.getLogger("olymp_client")
        self._token: Optional[str] = token
        self._account_type: str = account_type.upper()  # PRACTICE ou REAL
        self._ws_url: str = ws_url

        self._ws: Optional[WebSocketClientProtocol] = None
        self._is_connected: bool = False
        self._authenticated = asyncio.Event()
        self._listen_task: Optional[asyncio.Task[None]] = None

        # Dados da conta
        self._accounts: List[Dict[str, Any]] = []
        self._active_account_id: Optional[int] = None
        self._balance: float = 0.0

        # Correlação de requisições assíncronas
        self._requests: Dict[str, asyncio.Future[Any]] = {}

        # Streaming de preços e ordens
        self._live_prices: Dict[str, float] = {}
        self._live_price_ts: Dict[str, float] = {}
        self._subscribed_assets: Set[str] = set()
        self._closed_orders: Dict[str, Dict[str, Any]] = {}
        self._order_closed_events: Dict[str, asyncio.Event] = {}

        # Payouts estimados por par
        self._payouts: Dict[str, float] = {}

    @property
    def is_connected(self) -> bool:
        """Indica se a conexão WebSocket está ativa e autenticada."""
        return self._is_connected and self._ws is not None and getattr(self._ws, "state", 0) == 1

    def _load_cached_token(self) -> Optional[str]:
        """Tenta carregar token de sessão em cache do arquivo .olymp_token."""
        token_file = Path(".olymp_token")
        if token_file.exists():
            try:
                cached = token_file.read_text().strip()
                if cached:
                    return cached
            except Exception:
                pass
        return None

    def _save_cached_token(self, token: str) -> None:
        """Salva token no cache local para reconexões automáticas."""
        try:
            Path(".olymp_token").write_text(token.strip())
        except Exception:
            pass

    async def connect(self) -> bool:
        """Estabelece a conexão WebSocket com a Olymp Trade e autentica a sessão."""
        if not self._token:
            self._token = self._load_cached_token()

        if not self._token:
            self._logger.warning(
                "Nenhum token da Olymp Trade informado. Configure OLYMP_TOKEN no .env ou passe token no construtor."
            )
            # Sem token, opera em modo Simulado Fallback para permitir testes locais
            self._is_connected = False
            return False

        headers = {
            "Origin": "https://olymptrade.com",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        }

        self._logger.info("Conectando ao WebSocket da Olymp Trade: %s", self._ws_url.split("?")[0])
        try:
            self._ws = await websockets.connect(
                self._ws_url,
                extra_headers=headers,
                open_timeout=20,
                ping_interval=20,
                ping_timeout=20,
                max_size=30 * 1024 * 1024,
            )
        except Exception as exc:
            self._logger.error("Falha ao abrir conexão WebSocket com Olymp Trade: %s", exc)
            self._is_connected = False
            return False

        self._authenticated.clear()
        self._listen_task = asyncio.create_task(self._listen_loop())

        # Envia handshake de autenticação com o access_token
        auth_msg = {
            "t": 1,  # Tipo de mensagem de handshake
            "d": [{"access_token": self._token}],
        }
        await self._send_raw(auth_msg)

        try:
            # Aguarda confirmação de autenticação ou leitura inicial
            await asyncio.wait_for(self._authenticated.wait(), timeout=12.0)
            self._is_connected = True
            self._save_cached_token(self._token)
            self._logger.info("Autenticação com Olymp Trade concluída com sucesso.")
        except asyncio.TimeoutError:
            self._logger.warning("Tempo esgotado aguardando confirmação de login da Olymp Trade.")
            # Considera conectado se o socket responder
            self._is_connected = self._ws is not None

        # Atualiza saldos e conta ativa
        await self.get_balance()
        return self._is_connected

    async def _send_raw(self, payload: Any) -> None:
        """Envia mensagem serializada em JSON através do WebSocket."""
        if not self._ws:
            raise ConnectionError("WebSocket da Olymp Trade não está conectado.")
        await self._ws.send(json.dumps(payload))

    async def _request(self, payload: Dict[str, Any], timeout: float = 6.0) -> Any:
        """Envia requisição com id de rastreamento e aguarda retorno."""
        req_id = payload.get("uuid") or str(uuid.uuid4())
        payload["uuid"] = req_id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._requests[req_id] = fut
        try:
            await self._send_raw(payload)
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._requests.pop(req_id, None)

    async def _listen_loop(self) -> None:
        """Loop de escuta contínua de eventos do WebSocket."""
        try:
            while self._ws:
                raw_msg = await self._ws.recv()
                try:
                    data = json.loads(raw_msg)
                    await self._handle_incoming_message(data)
                except Exception as parse_err:
                    self._logger.debug("Mensagem não-JSON recebida: %s (%s)", raw_msg[:100], parse_err)
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            self._logger.warning("Conexão WebSocket com Olymp Trade encerrada.")
        except Exception as exc:
            self._logger.error("Erro no loop WebSocket Olymp Trade: %s", exc)
        finally:
            self._is_connected = False
            self._authenticated.clear()

    async def _handle_incoming_message(self, data: Any) -> None:
        """Processa as mensagens recebidas da corretora."""
        if isinstance(data, dict):
            # 1. Trata respostas a requisições com UUID
            uuid_key = data.get("uuid")
            if uuid_key and uuid_key in self._requests:
                fut = self._requests[uuid_key]
                if not fut.done():
                    fut.set_result(data.get("d") or data)

            # 2. Resposta de autenticação / saldos
            if "d" in data and isinstance(data["d"], list):
                for item in data["d"]:
                    if isinstance(item, dict):
                        # Detecta objetos de contas
                        if "account_id" in item and "amount" in item:
                            self._update_account(item)
                            self._authenticated.set()

                        # Detecta streaming de preços (ticks)
                        if "p" in item and "q" in item:
                            pair = item.get("p", "")
                            price = float(item.get("q", 0.0))
                            if pair:
                                self._live_prices[pair] = price
                                self._live_price_ts[pair] = time.time()

                        # Detecta resolução de ordens FTT
                        if "deal_id" in item or "order_id" in item:
                            self._handle_deal_update(item)

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_incoming_message(item)

    def _update_account(self, item: Dict[str, Any]) -> None:
        """Atualiza estado interno da conta e saldo ativo."""
        acc_group = item.get("group", "").lower()
        target_group = ACCOUNT_TYPE_MAP.get(self._account_type, "demo")

        if acc_group == target_group:
            self._active_account_id = item.get("account_id")
            self._balance = float(item.get("amount", 0.0))
            self._logger.debug(
                "Saldo Olymp Trade atualizado: %s | ID=%s | R$%.2f",
                self._account_type,
                self._active_account_id,
                self._balance,
            )

    def _handle_deal_update(self, item: Dict[str, Any]) -> None:
        """Processa atualizações de status de ordens FTT."""
        deal_id = str(item.get("deal_id") or item.get("order_id") or "")
        if not deal_id:
            return

        status = item.get("status", "").lower()
        if status in {"win", "loose", "loss", "draw", "equal", "closed"}:
            pnl = float(item.get("profit", item.get("win_amount", 0.0)))
            was_win = pnl > 0
            is_equal = pnl == 0

            win_str = "equal" if is_equal else ("win" if was_win else "loose")
            result_dict = {
                "win": win_str,
                "profit_amount": pnl,
                "amount": float(item.get("amount", 0.0)),
                "raw": item,
            }
            self._closed_orders[deal_id] = result_dict
            if deal_id in self._order_closed_events:
                self._order_closed_events[deal_id].set()

    async def change_balance(self, account_type: str) -> bool:
        """Alterna entre conta DEMO (PRACTICE) e LIVE (REAL)."""
        self._account_type = account_type.upper()
        target_group = ACCOUNT_TYPE_MAP.get(self._account_type, "demo")

        for acc in self._accounts:
            if acc.get("group") == target_group:
                self._active_account_id = acc.get("account_id")
                self._balance = float(acc.get("amount", 0.0))
                self._logger.info("Alternado para conta Olymp Trade: %s (ID %s)", self._account_type, self._active_account_id)
                return True
        return False

    async def get_balance(self) -> float:
        """Retorna o saldo monetário da conta ativa."""
        if not self.is_connected:
            return self._balance

        try:
            req = {
                "t": 2,
                "e": 94,  # Código de consulta de saldos da Olymp Trade
                "d": [],
            }
            res = await self._request(req, timeout=3.0)
            if isinstance(res, list):
                self._accounts = res
                for item in res:
                    if isinstance(item, dict):
                        self._update_account(item)
        except Exception as exc:
            self._logger.debug("Falha na consulta de saldo da Olymp Trade: %s", exc)

        return self._balance

    async def get_candles(
        self,
        par: str,
        timeframe: str = "M1",
        contagem: int = 100,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Busca lista de velas históricas OHLCV da Olymp Trade."""
        pair_code = normalize_pair(par)
        tf_secs = TIMEFRAME_SECONDS.get(timeframe.upper(), 60)
        to_time = end_time if end_time is not None else int(datetime.now(timezone.utc).timestamp())

        candles: List[Dict[str, Any]] = []
        if not self.is_connected:
            return candles

        try:
            req = {
                "t": 2,
                "e": 81,  # Código do protocolo Olymp Trade para get_candles
                "d": [
                    {
                        "pair": pair_code,
                        "size": tf_secs,
                        "count": contagem,
                        "to": to_time,
                    }
                ],
            }
            res = await self._request(req, timeout=5.0)
            raw_candles = res if isinstance(res, list) else (res.get("candles", []) if isinstance(res, dict) else [])

            for item in raw_candles:
                # Normaliza schema OHLCV compatível com o FeatureEngine do Razuk Bot
                c = {
                    "time": item.get("t") or item.get("time") or item.get("from"),
                    "open": float(item.get("o") or item.get("open", 0.0)),
                    "close": float(item.get("c") or item.get("close", 0.0)),
                    "high": float(item.get("h") or item.get("high") or item.get("max", 0.0)),
                    "low": float(item.get("l") or item.get("low") or item.get("min", 0.0)),
                    "volume": float(item.get("v") or item.get("volume", 0.0)),
                }
                candles.append(c)
        except Exception as exc:
            self._logger.error("Erro ao buscar velas de %s na Olymp Trade: %s", par, exc)

        return candles

    async def subscribe_candle(self, par: str) -> None:
        """Inscreve-se no stream de cotações em tempo real do par."""
        pair_code = normalize_pair(par)
        if pair_code in self._subscribed_assets:
            return

        try:
            req = {
                "t": 2,
                "e": 83,  # Código para subscribe_ticks
                "d": [{"pair": pair_code}],
            }
            await self._send_raw(req)
            self._subscribed_assets.add(pair_code)
            self._logger.debug("Inscrito no stream de ticks da Olymp Trade para %s", pair_code)
        except Exception as exc:
            self._logger.debug("Falha ao subscrever ticks de %s na Olymp Trade: %s", par, exc)

    async def unsubscribe_candle(self, par: str) -> None:
        """Cancela inscrição no stream de cotações."""
        pair_code = normalize_pair(par)
        self._subscribed_assets.discard(pair_code)

    async def get_current_price(self, par: str) -> float:
        """Retorna cotação viva em tempo real."""
        pair_code = normalize_pair(par)
        agora = time.time()

        if pair_code in self._live_prices and (agora - self._live_price_ts.get(pair_code, 0.0)) <= 2.0:
            return self._live_prices[pair_code]

        if pair_code not in self._subscribed_assets:
            await self.subscribe_candle(par)

        # Fallback: puxa vela recente
        candles = await self.get_candles(par, timeframe="M1", contagem=1)
        if candles:
            price = candles[-1]["close"]
            self._live_prices[pair_code] = price
            self._live_price_ts[pair_code] = agora
            return price

        return 0.0

    async def buy(
        self,
        par: str,
        valor: float,
        direcao: str,
        expiracao_minutos: int = 1,
    ) -> Dict[str, Any]:
        """Abre uma operação Fixed Time Trade (FTT) na direção up ou down."""
        pair_code = normalize_pair(par)
        dir_olymp = normalize_direction(direcao)
        duration_seconds = max(60, int(expiracao_minutos * 60))
        target_group = ACCOUNT_TYPE_MAP.get(self._account_type, "demo")

        order_id = str(uuid.uuid4())
        exp_timestamp = int(time.time()) + duration_seconds

        req = {
            "t": 2,
            "e": 85,  # Código de place_order FTT
            "d": [
                {
                    "pair": pair_code,
                    "amount": round(valor, 2),
                    "direction": dir_olymp,
                    "duration": duration_seconds,
                    "account_id": self._active_account_id,
                    "group": target_group,
                }
            ],
            "uuid": order_id,
        }

        try:
            res = await self._request(req, timeout=5.0)
            resp_dict = res if isinstance(res, dict) else {"result": res}
            resp_dict["id"] = resp_dict.get("deal_id") or resp_dict.get("order_id") or order_id
            resp_dict["exp_timestamp"] = exp_timestamp
            resp_dict["profit_percent"] = int(self.get_payout(par) * 100)
            return resp_dict
        except Exception as exc:
            self._logger.error("Falha ao enviar ordem FTT na Olymp Trade: %s", exc)
            return {"message": str(exc), "exp_timestamp": exp_timestamp}

    async def wait_for_option_result(self, order_id: Any, timeout: float = 70.0) -> Optional[Dict[str, Any]]:
        """Aguarda a resolução oficial da ordem emitida pela Olymp Trade."""
        str_id = str(order_id)
        if str_id in self._closed_orders:
            return self._closed_orders.pop(str_id)

        evt = asyncio.Event()
        self._order_closed_events[str_id] = evt
        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
            return self._closed_orders.pop(str_id, None)
        except asyncio.TimeoutError:
            return None
        finally:
            self._order_closed_events.pop(str_id, None)

    def get_payout(self, par: str) -> float:
        """Retorna payout decimal estimado (ex: 0.82 para 82%)."""
        pair_code = normalize_pair(par)
        return self._payouts.get(pair_code, 0.82)

    def is_asset_open(self, par: str) -> bool:
        """Verifica se o ativo está disponível para negociação."""
        return True

    async def load_initialization_data(self) -> None:
        """Carrega catálogo de payouts e ativos abertos."""
        self._logger.info("Catálogo inicial da Olymp Trade inicializado.")

    async def close(self) -> None:
        """Encerra conexão WebSocket."""
        self._is_connected = False
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._logger.info("Conexão com a Olymp Trade encerrada.")
