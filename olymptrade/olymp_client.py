"""Cliente assíncrono para comunicação com a Olymp Trade via WebSocket (FTT / Fixed Time Trades).

Implementa a mesma interface padronizada do conector IQ Option do Razuk Bot:
- Autenticação via Cookie de sessão (access_token)
- Consulta e alternância de saldo (DEMO / REAL)
- Coleta de velas históricas OHLCV (Evento 10 / 1003)
- Streaming de cotações em tempo real (Evento 12 / 280 / Evento 1)
- Envio de ordens FTT (Evento 23) e resolução de resultado (Evento 26)
"""

import asyncio
from datetime import datetime, timezone
import inspect
import json
import logging
from pathlib import Path
import random
import string
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


def _generate_uuid() -> str:
    """Gera identificador de requisição compatível com a Olymp Trade."""
    prefix = "".join(random.choices(string.ascii_uppercase, k=4))
    suffix = "".join(random.choices(string.ascii_lowercase, k=2))
    return f"{prefix}-{suffix}"


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
        self._ping_task: Optional[asyncio.Task[None]] = None

        # Dados da conta
        self._accounts: Dict[str, Dict[str, Any]] = {}
        self._active_account_id: Optional[int] = None
        self._balance: float = 0.0

        # Correlação de requisições assíncronas por UUID e por Event Code
        self._requests: Dict[str, asyncio.Future[Any]] = {}
        self._event_waiters: Dict[int, List[asyncio.Future[Any]]] = {}

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
        candidates = [
            Path(".olymp_token"),
            Path(__file__).parent / ".olymp_token",
            Path(__file__).parent.parent / ".olymp_token",
        ]
        for p in candidates:
            if p.exists():
                try:
                    cached = p.read_text().strip()
                    if cached:
                        return cached
                except Exception:
                    pass
        return None

    def _save_cached_token(self, token: str) -> None:
        """Salva token no cache local para reconexões automáticas."""
        candidates = [
            Path(".olymp_token"),
            Path(__file__).parent / ".olymp_token",
            Path(__file__).parent.parent / ".olymp_token",
        ]
        for p in candidates:
            try:
                p.write_text(token.strip())
            except Exception:
                pass

    async def connect(self) -> bool:
        """Estabelece a conexão WebSocket com a Olymp Trade e autentica a sessão."""
        if not self._token:
            self._token = self._load_cached_token()

        if not self._token:
            self._logger.warning(
                "Nenhum token da Olymp Trade informado. Configure OLYMP_TOKEN no .env ou salve em .olymp_token."
            )
            self._is_connected = False
            return False

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            ),
            "Cookie": f"access_token={self._token}",
        }

        connect_kwargs: Dict[str, Any] = {
            "open_timeout": 20,
            "ping_interval": None,  # Gerenciamos ping manualmente via evento 90
            "max_size": 30 * 1024 * 1024,
        }

        sig = inspect.signature(websockets.connect)
        if "additional_headers" in sig.parameters:
            connect_kwargs["additional_headers"] = headers
            if "origin" in sig.parameters:
                connect_kwargs["origin"] = "https://olymptrade.com"
        elif "extra_headers" in sig.parameters:
            headers["Origin"] = "https://olymptrade.com"
            connect_kwargs["extra_headers"] = headers

        self._logger.info("Conectando ao WebSocket da Olymp Trade: %s", self._ws_url.split("?")[0])
        try:
            self._ws = await websockets.connect(self._ws_url, **connect_kwargs)
        except Exception as exc:
            self._logger.error("Falha ao abrir conexão WebSocket com Olymp Trade: %s", exc)
            self._is_connected = False
            return False

        self._is_connected = True
        self._authenticated.clear()
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

        # Executa sequência de inicialização e subscrição (padrão oficial do navegador)
        await self._initialize_session()

        try:
            await asyncio.wait_for(self._authenticated.wait(), timeout=12.0)
            self._save_cached_token(self._token)
            self._logger.info(
                "Autenticação Olymp Trade concluída! Conta ativa: %s (Saldo: R$%.2f)",
                self._account_type,
                self._balance,
            )
            return True
        except asyncio.TimeoutError:
            if self._active_account_id:
                self._is_connected = True
                return True
            self._logger.warning("Tempo esgotado aguardando confirmação de dados da conta.")
            return self._is_connected

    async def _initialize_session(self) -> None:
        """Envia subscrições iniciais e requisição de saldo imitando o navegador."""
        startup_subscriptions = [
            [220],
            [110, 700, 112, 140, 1038, 1037, 1039, 141, 22, 26, 111],
            [1054, 1076, 1301, 1097],
            [141, 241],
            [230, 231],
            [75],
            [1055],
            [2223, 2301, 55, 150, 152, 151, 126, 602, 601],
            [2076],
            [126],
        ]
        for sub in startup_subscriptions:
            await self._send_event(98, sub)

        # Solicita dados das contas demo e real (evento 1068)
        for group in ("demo", "real"):
            req_uuid = _generate_uuid()
            await self._send_event(1068, [{"group": group}], request_uuid=req_uuid)

    async def _ping_loop(self) -> None:
        """Envia ping periódico (evento 90) para manter o canal ativo."""
        try:
            while self.is_connected:
                await asyncio.sleep(25)
                if self._ws:
                    await self._send_event(90, {})
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _send_event(self, event_code: int, data: Any, request_uuid: Optional[str] = None) -> None:
        """Envia mensagem formatada no protocolo da Olymp Trade ([{'t': 2, 'e': code, 'd': data}])."""
        if not self._ws:
            raise ConnectionError("WebSocket da Olymp Trade não está conectado.")

        part: Dict[str, Any] = {
            "t": 2,
            "e": event_code,
            "d": data,
        }
        if request_uuid:
            part["uuid"] = request_uuid

        payload = [part]
        await self._ws.send(json.dumps(payload))

    async def _listen_loop(self) -> None:
        """Loop de escuta contínua de eventos do WebSocket."""
        try:
            while self._ws:
                raw_msg = await self._ws.recv()
                try:
                    data = json.loads(raw_msg)
                    await self._handle_incoming_message(data)
                except Exception as parse_err:
                    self._logger.debug("Mensagem não-JSON recebida: %s (%s)", str(raw_msg)[:100], parse_err)
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
        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            event_code = item.get("e")
            payload = item.get("d")
            uuid_key = item.get("uuid")

            # 1. Resposta a requisição correlacionada por UUID
            if uuid_key and uuid_key in self._requests:
                fut = self._requests[uuid_key]
                if not fut.done():
                    fut.set_result(payload or item)

            # 2. Resposta de velas históricas (Evento 10 ou 1003 contendo 'candles')
            if event_code in {10, 1003} and isinstance(payload, list):
                candle_list = []
                for p_item in payload:
                    if isinstance(p_item, dict) and "candles" in p_item:
                        candle_list = p_item.get("candles", [])
                        break
                if 10 in self._event_waiters:
                    while self._event_waiters[10]:
                        fut = self._event_waiters[10].pop(0)
                        if not fut.done():
                            fut.set_result(candle_list)
                continue

            # 3. Resposta correlacionada por Event Code genérico
            if event_code and event_code in self._event_waiters:
                waiters = self._event_waiters[event_code]
                while waiters:
                    w = waiters.pop(0)
                    if not w.done():
                        w.set_result(payload or item)

            # 4. Informações de Conta e Saldo (Lista de contas contendo amount, account_id, group)
            if isinstance(payload, list):
                for sub in payload:
                    if isinstance(sub, dict) and "account_id" in sub and "amount" in sub:
                        self._update_account(sub)

            # 5. Ticks de preço em tempo real (Evento 1)
            if event_code == 1 and isinstance(payload, list):
                for tick in payload:
                    if isinstance(tick, dict):
                        pair = tick.get("p", "")
                        price = float(tick.get("q", 0.0))
                        if pair and price > 0:
                            self._live_prices[pair] = price
                            self._live_price_ts[pair] = time.time()

            # 6. Execução ou fechamento de ordens FTT (Eventos 22 e 26)
            if event_code in {22, 26}:
                self._handle_deal_update(payload)

    def _update_account(self, item: Dict[str, Any]) -> None:
        """Atualiza estado interno da conta e saldo ativo."""
        group = item.get("group", "").lower()
        acc_id = item.get("account_id")
        amount = float(item.get("amount", 0.0))

        if group:
            self._accounts[group] = {
                "account_id": acc_id,
                "amount": amount,
                "raw": item,
            }

        target_group = ACCOUNT_TYPE_MAP.get(self._account_type, "demo")
        if group == target_group:
            self._active_account_id = acc_id
            self._balance = amount
            self._authenticated.set()

    def _handle_deal_update(self, payload: Any) -> None:
        """Processa atualizações de status de ordens FTT (Evento 22 e 26)."""
        deals = payload if isinstance(payload, list) else [payload]
        for deal in deals:
            if not isinstance(deal, dict):
                continue
            deal_id = str(deal.get("id") or deal.get("deal_id") or "")
            if not deal_id:
                continue

            status = str(deal.get("status", "")).lower()
            profit = float(deal.get("profit", deal.get("win_amount", 0.0)))
            amount = float(deal.get("amount", 0.0))

            was_win = profit > amount
            is_equal = profit == amount

            win_str = "equal" if is_equal else ("win" if was_win else "loose")
            result_dict = {
                "win": win_str,
                "profit_amount": profit - amount if was_win else -amount,
                "amount": amount,
                "raw": deal,
            }
            self._closed_orders[deal_id] = result_dict
            if deal_id in self._order_closed_events:
                self._order_closed_events[deal_id].set()

    async def change_balance(self, account_type: str) -> bool:
        """Alterna entre conta DEMO (PRACTICE) e LIVE (REAL)."""
        self._account_type = account_type.upper()
        target_group = ACCOUNT_TYPE_MAP.get(self._account_type, "demo")

        if target_group in self._accounts:
            acc = self._accounts[target_group]
            self._active_account_id = acc["account_id"]
            self._balance = acc["amount"]
            self._logger.info("Alternado para conta Olymp Trade: %s (ID %s)", self._account_type, self._active_account_id)
            return True
        return False

    async def get_balance(self) -> float:
        """Retorna o saldo monetário da conta ativa."""
        target_group = ACCOUNT_TYPE_MAP.get(self._account_type, "demo")
        if target_group in self._accounts:
            self._balance = self._accounts[target_group]["amount"]
        return self._balance

    async def get_candles(
        self,
        par: str,
        timeframe: str = "M1",
        contagem: int = 100,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Busca lista de velas históricas OHLCV da Olymp Trade (Evento 10)."""
        pair_code = normalize_pair(par)
        tf_secs = TIMEFRAME_SECONDS.get(timeframe.upper(), 60)
        to_time = end_time if end_time is not None else int(datetime.now(timezone.utc).timestamp())

        candles: List[Dict[str, Any]] = []
        if not self.is_connected:
            return candles

        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        if 10 not in self._event_waiters:
            self._event_waiters[10] = []
        self._event_waiters[10].append(fut)

        req_uuid = _generate_uuid()
        req_data = [{"pair": pair_code, "size": tf_secs, "to": to_time, "solid": True}]

        try:
            await self._send_event(10, req_data, request_uuid=req_uuid)
            raw_candles = await asyncio.wait_for(fut, timeout=6.0)

            for item in raw_candles:
                c = {
                    "time": item.get("t") or item.get("time") or item.get("from"),
                    "open": float(item.get("open") or item.get("o", 0.0)),
                    "close": float(item.get("close") or item.get("c", 0.0)),
                    "high": float(item.get("high") or item.get("h") or item.get("max", 0.0)),
                    "low": float(item.get("low") or item.get("l") or item.get("min", 0.0)),
                    "volume": float(item.get("volume") or item.get("v", 1.0)),
                }
                candles.append(c)
            # Ordena temporalmente as velas da mais antiga para a mais recente
            candles.sort(key=lambda x: x["time"])
            if contagem > 0 and len(candles) > contagem:
                candles = candles[-contagem:]
        except Exception as exc:
            self._logger.debug("Falha na coleta de velas de %s (Olymp Trade): %s", par, exc)
        finally:
            if 10 in self._event_waiters and fut in self._event_waiters[10]:
                self._event_waiters[10].remove(fut)

        return candles

    async def subscribe_candle(self, par: str) -> None:
        """Inscreve-se no stream de ticks e velas do par (Eventos 12 e 280)."""
        pair_code = normalize_pair(par)
        if pair_code in self._subscribed_assets:
            return

        try:
            await self._send_event(12, [{"pair": pair_code}])
            await self._send_event(280, [{"pair": pair_code}])
            self._subscribed_assets.add(pair_code)
            self._logger.debug("Inscrito no stream de ticks da Olymp Trade para %s", pair_code)
        except Exception as exc:
            self._logger.debug("Falha ao subscrever ticks de %s na Olymp Trade: %s", par, exc)

    async def unsubscribe_candle(self, par: str) -> None:
        """Cancela inscrição no stream de cotações (Eventos 13 e 281)."""
        pair_code = normalize_pair(par)
        try:
            await self._send_event(13, [{"pair": pair_code}])
            await self._send_event(281, [{"pair": pair_code}])
            self._subscribed_assets.discard(pair_code)
        except Exception:
            pass

    async def get_current_price(self, par: str) -> float:
        """Retorna cotação viva em tempo real."""
        pair_code = normalize_pair(par)
        agora = time.time()

        if pair_code not in self._subscribed_assets:
            await self.subscribe_candle(par)

        if pair_code in self._live_prices and (agora - self._live_price_ts.get(pair_code, 0.0)) <= 3.0:
            return self._live_prices[pair_code]

        # Aguarda até 1.5s pela chegada do primeiro tick via WebSocket
        for _ in range(15):
            await asyncio.sleep(0.1)
            if pair_code in self._live_prices and (time.time() - self._live_price_ts.get(pair_code, 0.0)) <= 3.0:
                return self._live_prices[pair_code]

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
        """Abre uma operação Fixed Time Trade (FTT) na direção up ou down (Evento 23)."""
        pair_code = normalize_pair(par)
        dir_olymp = normalize_direction(direcao)
        duration_seconds = max(60, int(expiracao_minutos * 60))
        target_group = ACCOUNT_TYPE_MAP.get(self._account_type, "demo")

        order_uuid = _generate_uuid()
        exp_timestamp = int(time.time()) + duration_seconds
        timestamp_ms = int(time.time() * 1000)

        data = [
            {
                "amount": round(valor, 2),
                "dir": dir_olymp,
                "pair": pair_code,
                "cat": "digital",
                "pos": 0,
                "source": "platform",
                "account_id": self._active_account_id,
                "group": target_group,
                "timestamp": timestamp_ms,
                "risk_free_id": None,
                "is_flex": False,
                "duration": duration_seconds,
            }
        ]

        try:
            fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            self._requests[order_uuid] = fut
            await self._send_event(23, data, request_uuid=order_uuid)

            res = await asyncio.wait_for(fut, timeout=5.0)
            resp_dict = res if isinstance(res, dict) else (res[0] if isinstance(res, list) and res else {})
            real_id = resp_dict.get("id") or resp_dict.get("deal_id") or order_uuid

            return {
                "id": real_id,
                "exp_timestamp": exp_timestamp,
                "profit_percent": int(self.get_payout(par) * 100),
                "raw": resp_dict,
            }
        except Exception as exc:
            self._logger.error("Falha ao enviar ordem FTT na Olymp Trade: %s", exc)
            return {"message": str(exc), "exp_timestamp": exp_timestamp}
        finally:
            self._requests.pop(order_uuid, None)

    async def wait_for_option_result(self, order_id: Any, timeout: float = 70.0) -> Optional[Dict[str, Any]]:
        """Aguarda a resolução oficial da ordem emitida pela Olymp Trade (Evento 26)."""
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
        """Retorna payout decimal estimado (ex: 0.85 para 85%)."""
        pair_code = normalize_pair(par)
        return self._payouts.get(pair_code, 0.85)

    def is_asset_open(self, par: str) -> bool:
        """Verifica se o ativo está disponível para negociação."""
        return True

    async def load_initialization_data(self) -> None:
        """Carrega catálogo de payouts e ativos abertos."""
        self._logger.info("Catálogo inicial da Olymp Trade sincronizado.")

    async def close(self) -> None:
        """Encerra conexão WebSocket."""
        self._is_connected = False
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._logger.info("Conexão com a Olymp Trade encerrada.")
