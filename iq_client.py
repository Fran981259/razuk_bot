"""Cliente assíncrono para comunicação com a API da IQ Option via WebSocket e REST.

# GOLDCODE EXEMPTION: Protocolo único e coeso de mensageria WebSocket da corretora IQ Option.
Implementa autenticação (REST/SSID), streaming WebSocket, troca de saldo,
coleta de velas históricas e envio de ordens.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set
import uuid

import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol

from config import ConexaoConfig
from iq_constants import (
    BALANCE_TYPE_PRACTICE,
    BALANCE_TYPE_REAL,
    OPTION_BINARY,
    OPTION_TURBO,
    TIMEFRAME_SECONDS,
    get_active_id,
)


class IqOptionClient:
    """Cliente assíncrono para operações na corretora IQ Option."""

    def __init__(self, config: ConexaoConfig) -> None:
        """Inicializa o cliente com as configurações fornecidas."""
        self._config = config
        self._logger = logging.getLogger("iq_client")
        self._ws: Optional[WebSocketClientProtocol] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._ssid: Optional[str] = config.ssid
        self._requests: Dict[str, asyncio.Future[Any]] = {}
        self._profile: Dict[str, Any] = {}
        self._balance: float = 0.0
        self._balance_id: Optional[int] = None
        self._authenticated = asyncio.Event()
        self._listen_task: Optional[asyncio.Task[None]] = None
        self._is_connected: bool = False
        self._payouts: Dict[int, int] = {}
        self._open_actives: Set[int] = set()
        self._closed_options: Dict[int, Dict[str, Any]] = {}
        self._option_closed_events: Dict[int, asyncio.Event] = {}
        self._live_prices: Dict[int, float] = {}
        self._live_price_ts: Dict[int, float] = {}
        self._subscribed_candles: Set[int] = set()
        self._ws_error_count: int = 0

    @property
    def is_connected(self) -> bool:
        """Indica se a conexão WebSocket está ativa e autenticada."""
        return self._is_connected and self._ws is not None and getattr(self._ws, "state", 0) == 1

    async def _authenticate_rest(self) -> str:
        """Realiza autenticação via endpoint REST para obter o token SSID com retentativas."""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=25, connect=12)
            self._session = aiohttp.ClientSession(timeout=timeout)

        payload = {"identifier": self._config.email, "password": self._config.password}
        for tentativa in range(1, 4):
            try:
                async with self._session.post(self._config.corretora_url, json=payload) as resp:
                    if resp.status != 200:
                        raise ConnectionError(f"Falha no login REST (HTTP {resp.status}). Verifique credenciais.")
                    data = await resp.json()
                    ssid = data.get("data", {}).get("ssid") or resp.cookies.get("ssid")
                    if not ssid:
                        raise ConnectionError("SSID não retornado no login REST da IQ Option.")
                    return str(ssid.value if hasattr(ssid, "value") else ssid)
            except Exception as exc:
                if tentativa == 3:
                    raise ConnectionError(f"Falha no login REST após 3 tentativas: {exc}") from exc
                self._logger.warning("Tentativa %d de login REST falhou (%s). Retentando em 3s...", tentativa, exc)
                await asyncio.sleep(3)

    async def connect(self) -> bool:
        """Estabelece a conexão WebSocket e autentica o usuário com cache de SSID."""
        ssid_file = Path(".session_ssid")

        # SSID expirado: 3+ erros consecutivos de WS → força novo login REST
        if self._ws_error_count >= 3:
            self._logger.warning(
                "SSID possivelmente expirado (%d erros WS consecutivos). Forçando novo login REST...",
                self._ws_error_count,
            )
            self._ssid = None
            self._ws_error_count = 0
            ssid_file.unlink(missing_ok=True)

        if not self._ssid and ssid_file.exists():
            try:
                cached = ssid_file.read_text().strip()
                if cached:
                    self._ssid = cached
                    self._logger.info("Utilizando token de sessão SSID em cache.")
            except Exception:
                pass

        if not self._ssid:
            self._logger.info("Obtendo novo token de sessão SSID via REST...")
            self._ssid = await self._authenticate_rest()
            try:
                ssid_file.write_text(self._ssid)
            except Exception:
                pass

        self._logger.info("Conectando ao WebSocket da IQ Option: %s", self._config.ws_url)
        self._ws = await websockets.connect(
            self._config.ws_url, open_timeout=20, ping_interval=20, ping_timeout=20, max_size=50 * 1024 * 1024
        )
        self._authenticated.clear()
        self._listen_task = asyncio.create_task(self._listen_loop())

        await self._send("ssid", self._ssid)
        await self._send("setOptions", {"sendResults": True})
        try:
            await asyncio.wait_for(self._authenticated.wait(), timeout=12.0)
        except asyncio.TimeoutError as exc:
            self._ssid = None
            if ssid_file.exists():
                ssid_file.unlink(missing_ok=True)
            await self.close()
            raise TimeoutError("Tempo esgotado aguardando autenticação WebSocket da IQ Option.") from exc

        await self.change_balance(self._config.account_type)
        await self.load_initialization_data()
        self._is_connected = True
        self._ws_error_count = 0  # auth OK → limpa contador
        self._logger.info("Conexão e autenticação concluídas com sucesso.")
        return True

    async def _send(self, name: str, msg: Any, req_id: Optional[str] = None) -> None:
        """Envia mensagem serializada em JSON através do WebSocket."""
        if not self._ws:
            raise ConnectionError("WebSocket não está conectado.")
        envelope: Dict[str, Any] = {"name": name, "msg": msg}
        if req_id:
            envelope["request_id"] = req_id
        await self._ws.send(json.dumps(envelope))

    async def _request(self, name: str, msg: Any, timeout: float = 10.0) -> Any:
        """Envia requisição com ID exclusivo e aguarda o future de resposta."""
        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._requests[req_id] = future
        try:
            await self._send(name, msg, req_id=req_id)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._requests.pop(req_id, None)

    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """Processa mensagens individuais recebidas pelo WebSocket."""
        name = data.get("name")
        req_id = str(data.get("request_id") or "")
        msg = data.get("msg")

        if req_id in self._requests and not self._requests[req_id].done():
            self._requests[req_id].set_result(msg)

        if name == "heartbeat":
            await self._send("heartbeat", msg)
        elif name == "profile":
            self._update_profile(msg)
            self._authenticated.set()
        elif name in {"balances", "balance-changed"}:
            if isinstance(msg, dict):
                current = msg.get("current_balance", {})
                if current.get("id") == self._balance_id:
                    self._balance = float(current.get("amount", self._balance))
            elif isinstance(msg, list):
                self._update_balances(msg)
        elif name in {"option-closed", "socket-option-closed"}:
            opt = msg if isinstance(msg, dict) else {}
            opt_id = opt.get("id") or opt.get("option_id")
            if opt_id is not None:
                try:
                    int_id = int(opt_id)
                    self._closed_options[int_id] = opt
                    evt = self._option_closed_events.get(int_id)
                    if evt:
                        evt.set()
                except Exception:
                    pass
        elif name in {"candle-generated", "candles", "live-deal"}:
            if isinstance(msg, dict):
                aid = msg.get("active_id")
                close_p = msg.get("close", msg.get("price"))
                if aid is not None and close_p is not None:
                    try:
                        int_aid = int(aid)
                        self._live_prices[int_aid] = float(close_p)
                        self._live_price_ts[int_aid] = time.time()
                    except (ValueError, TypeError):
                        pass

    def _update_balances(self, balances: List[Dict[str, Any]]) -> None:
        """Atualiza saldo ativo a partir de lista de saldos."""
        tipo_alvo = BALANCE_TYPE_PRACTICE if self._config.account_type == "PRACTICE" else BALANCE_TYPE_REAL
        for item in balances:
            if item.get("type") == tipo_alvo:
                self._balance = float(item.get("amount", 0.0))
                self._balance_id = item.get("id")
                break

    def _update_profile(self, profile_data: Dict[str, Any]) -> None:
        """Atualiza informações do perfil e saldo ativo."""
        self._profile = profile_data or {}
        balances = self._profile.get("balances", [])
        self._update_balances(balances)

    async def _listen_loop(self) -> None:
        """Loop de escuta assíncrono para mensagens WebSocket."""
        try:
            while self._ws:
                raw_msg = await self._ws.recv()
                await self._handle_message(json.loads(raw_msg))
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            self._ws_error_count += 1
        except Exception as exc:
            self._logger.error("Erro no loop de mensagens WebSocket: %s", exc)
            self._ws_error_count += 1
        finally:
            self._is_connected = False
            self._authenticated.clear()
            for req_id, fut in list(self._requests.items()):
                if not fut.done():
                    fut.set_exception(ConnectionError("WebSocket encerrado."))
            self._requests.clear()

    async def change_balance(self, account_type: str) -> bool:
        """Alterna o saldo ativo da conta entre PRACTICE (demo) e REAL."""
        tipo_int = BALANCE_TYPE_PRACTICE if account_type.upper() == "PRACTICE" else BALANCE_TYPE_REAL
        payload = {"name": "change-balance-type", "version": "2.0", "body": {"type": tipo_int}}
        await self._send("sendMessage", payload)
        self._config.account_type = "PRACTICE" if tipo_int == BALANCE_TYPE_PRACTICE else "REAL"
        self._update_profile(self._profile)
        return True

    async def get_balance(self) -> float:
        """Retorna o saldo monetário da conta atualmente ativa."""
        try:
            payload = {"name": "get-balances", "version": "1.0"}
            res = await self._request("sendMessage", payload, timeout=2.5)
            if isinstance(res, list):
                self._update_balances(res)
        except Exception:
            pass
        return self._balance

    async def get_candles(
        self,
        par: str,
        timeframe: str = "M1",
        contagem: int = 100,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Busca lista de velas históricas do par informado com suporte a paginação."""
        active_id = get_active_id(par)
        duracao = TIMEFRAME_SECONDS.get(timeframe.upper(), 60)
        agora_epoch = end_time if end_time is not None else int(datetime.now(timezone.utc).timestamp())
        payload = {
            "name": "get-candles",
            "version": "2.0",
            "body": {"active_id": active_id, "size": duracao, "count": contagem, "to": agora_epoch},
        }
        res = await self._request("sendMessage", payload)
        candles = list(res) if isinstance(res, list) else (res.get("candles", []) if isinstance(res, dict) else [])
        for c in candles:
            if "min" in c and "low" not in c:
                c["low"] = c["min"]
            if "max" in c and "high" not in c:
                c["high"] = c["max"]
        return candles

    async def subscribe_candle(self, par: str) -> None:
        """Inscreve-se no feed de streaming candle-generated em tempo real."""
        try:
            aid = get_active_id(par)
            if aid not in self._subscribed_candles:
                payload = {
                    "name": "candle-generated",
                    "params": {
                        "routingFilters": {
                            "active_id": aid,
                            "size": 1,
                        }
                    },
                }
                await self._send("subscribeMessage", payload)
                self._subscribed_candles.add(aid)
                self._logger.debug("Inscrito no stream de candle-generated para %s (ID %d).", par, aid)
        except Exception as exc:
            self._logger.debug("Falha ao subscrever candle de %s: %s", par, exc)

    async def unsubscribe_candle(self, par: str) -> None:
        """Cancela inscrição no feed candle-generated."""
        try:
            aid = get_active_id(par)
            if aid in self._subscribed_candles:
                payload = {
                    "name": "candle-generated",
                    "params": {
                        "routingFilters": {
                            "active_id": aid,
                            "size": 1,
                        }
                    },
                }
                await self._send("unsubscribeMessage", payload)
                self._subscribed_candles.discard(aid)
                self._logger.debug("Cancelada subscrição candle para %s (ID %d).", par, aid)
        except Exception as exc:
            self._logger.debug("Falha ao cancelar subscrição candle de %s: %s", par, exc)

    async def get_current_price(self, par: str) -> float:
        """Retorna cotação em tempo real (stream WebSocket ativo ou vela viva recente)."""
        try:
            aid = get_active_id(par)
            agora = time.time()
            # 1. Se recebemos cotação recente via stream WebSocket nos últimos 2 segundos:
            if aid in self._live_prices and (agora - self._live_price_ts.get(aid, 0.0)) <= 2.0:
                return self._live_prices[aid]

            # 2. Garante que estamos inscritos no stream de velas em tempo real
            if aid not in self._subscribed_candles:
                await self.subscribe_candle(par)

            # 3. Busca a vela M1 mais recente para atualização viva imediata
            candles = await self.get_candles(par, timeframe="M1", contagem=1)
            if candles:
                price = float(candles[-1].get("close", 0.0))
                self._live_prices[aid] = price
                self._live_price_ts[aid] = agora
                return price
        except Exception as exc:
            self._logger.debug("Falha ao obter preço em tempo real de %s: %s", par, exc)
        return 0.0

    async def buy(self, par: str, valor: float, direcao: str, expiracao_minutos: int = 1) -> Dict[str, Any]:
        """Abre uma ordem de opção binária/turbo na direção CALL ou PUT."""
        active_id = get_active_id(par)
        direcao_limpa = direcao.strip().lower()
        if direcao_limpa not in {"call", "put"}:
            raise ValueError("Direção da ordem deve ser 'call' ou 'put'.")

        agora = int(time.time())
        if expiracao_minutos >= 60:
            secs = 3600
            resto = agora % secs
            exp_timestamp = (agora - resto) + (2 * secs if (secs - resto) < 120 else secs)
            option_type = OPTION_BINARY
        elif expiracao_minutos >= 30:
            secs = 1800
            resto = agora % secs
            exp_timestamp = (agora - resto) + (2 * secs if (secs - resto) < 90 else secs)
            option_type = OPTION_BINARY
        elif expiracao_minutos >= 15:
            secs = 900
            resto = agora % secs
            exp_timestamp = (agora - resto) + (2 * secs if (secs - resto) < 60 else secs)
            option_type = OPTION_BINARY
        elif expiracao_minutos >= 5:
            resto_5m = agora % 300
            # Expiração de 5 minutos alinhada ao fechamento da vela M5 (múltiplo de 300s)
            exp_timestamp = (agora - resto_5m) + (600 if (300 - resto_5m) < 45 else 300)
            option_type = OPTION_TURBO
        else:
            resto = agora % 60
            add_sec = 60 if resto >= 30 else 0
            exp_timestamp = (agora - resto) + add_sec + (expiracao_minutos * 60)
            if exp_timestamp - agora < 10:
                exp_timestamp += 60
            option_type = OPTION_TURBO

        profit_pct = self._payouts.get(active_id, 88)
        payload = {
            "name": "binary-options.open-option", "version": "1.0",
            "body": {
                "user_balance_id": self._balance_id, "active_id": active_id,
                "option_type_id": option_type, "direction": direcao_limpa,
                "expired": exp_timestamp, "price": valor, "profit_percent": profit_pct,
            },
        }
        res = await self._request("sendMessage", payload)
        if isinstance(res, dict) and "actual_commission" in str(res):
            comm = res.get("result", {}).get("actual_commission", {}).get("commission")
            if comm is not None:
                new_profit = 100 - int(comm)
                self._payouts[active_id] = new_profit
                payload["body"]["profit_percent"] = new_profit
                self._logger.info("Ajustando profit_percent para %d%% e reexecutando ordem...", new_profit)
                res = await self._request("sendMessage", payload)

        resultado = dict(res) if isinstance(res, dict) else {"result": res}
        resultado["exp_timestamp"] = exp_timestamp
        return resultado

    def get_payout(self, par: str) -> float:
        """Retorna o payout decimal do par (ex: 0.85 para 85%)."""
        try:
            active_id = get_active_id(par)
            return self._payouts.get(active_id, 85) / 100.0
        except Exception:
            return 0.85

    def is_asset_open(self, par: str) -> bool:
        """Verifica se o par está atualmente aberto e com negociação autorizada."""
        if not self._open_actives:
            return True
        try:
            aid = get_active_id(par)
            return aid in self._open_actives
        except KeyError:
            return False

    async def load_initialization_data(self) -> None:
        """Carrega catálogo oficial de ativos e payouts da corretora com checagem de horário."""
        try:
            payload = {"name": "get-initialization-data", "version": "3.0"}
            data = await self._request("sendMessage", payload, timeout=5.0)
            if not isinstance(data, dict):
                return
            now = int(time.time())
            abertos: Set[int] = set()
            for cat in ("turbo", "binary"):
                actives = data.get(cat, {}).get("actives", {})
                for aid_str, item in actives.items():
                    try:
                        aid = int(aid_str)
                    except (ValueError, TypeError):
                        continue
                    enabled = bool(item.get("enabled", False))
                    suspended = bool(item.get("is_suspended", False))
                    schedule = item.get("schedule", [])
                    in_sched = any(s <= now <= e for s, e in schedule) if schedule else True
                    if enabled and not suspended and in_sched:
                        abertos.add(aid)
                    commission = item.get("option", {}).get("profit", {}).get("commission", 15)
                    self._payouts[aid] = 100 - int(commission)
            self._open_actives = abertos
            self._logger.info("Catálogo IQ sincronizado: %d ativos abertos e payouts atualizados.", len(self._open_actives))
        except Exception as exc:
            self._logger.warning("Não foi possível carregar catálogo da IQ Option: %s", exc)



    async def wait_for_option_result(self, order_id: Any, timeout: float = 70.0) -> Optional[Dict[str, Any]]:
        """Aguarda a mensagem oficial option-closed enviada pela corretora para a ordem."""
        try:
            int_id = int(order_id)
        except (ValueError, TypeError):
            return None

        if int_id in self._closed_options:
            return self._closed_options.pop(int_id)

        evt = asyncio.Event()
        self._option_closed_events[int_id] = evt
        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
            return self._closed_options.pop(int_id, None)
        except asyncio.TimeoutError:
            return None
        finally:
            self._option_closed_events.pop(int_id, None)

    async def close(self) -> None:
        """Fecha WebSocket, cancela tarefas e encerra a sessão HTTP."""
        self._is_connected = False
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._logger.info("Conexão com a IQ Option encerrada.")
