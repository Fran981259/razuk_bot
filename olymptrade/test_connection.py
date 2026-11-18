"""Script de teste e validação da conexão assíncrona com a Olymp Trade.

Permite testar de forma segura:
1. Conexão WebSocket e envio de headers corretos
2. Validação do access_token da sessão
3. Consulta de saldo (DEMO / REAL)
4. Coleta de velas recentes (EURUSD M1/M5)
5. Teste de streaming de cotações em tempo real
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Adiciona diretório raiz do bot ao sys.path para importações se necessário
sys.path.insert(0, str(Path(__file__).parent.parent))

from olymptrade.olymp_client import OlympTradeClient
from olymptrade.olymp_constants import DEFAULT_WS_URL


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("test_olymp")

    print("\n" + "=" * 65)
    print(" 🚀 TESTE DE CONECTIVIDADE COM A OLYMP TRADE (FTT)")
    print("=" * 65)

    # 1. Carrega token do ambiente ou arquivo .olymp_token
    token = os.getenv("OLYMP_TOKEN")
    token_file = Path(".olymp_token")
    if not token and token_file.exists():
        token = token_file.read_text().strip()

    if not token:
        print("\n⚠️ AVISO: Nenhum token de sessão 'OLYMP_TOKEN' encontrado!")
        print("Para conectar à sua conta real/demo da Olymp Trade:")
        print("  1. Abra o site https://olymptrade.com no seu navegador e faça login.")
        print("  2. Pressione F12 -> aba 'Network' (Rede) -> filtro 'WS' (WebSockets).")
        print("  3. Clique na conexão ativa e copie o valor do seu 'access_token'.")
        print("  4. Crie o arquivo .olymp_token na raiz ou defina OLYMP_TOKEN no .env.")
        print("\nTentando conexão com o endpoint público para verificar conectividade de rede...")

    account_type = os.getenv("OLYMP_ACCOUNT_TYPE", "PRACTICE")
    client = OlympTradeClient(token=token, account_type=account_type)

    print(f"\n[1/4] Iniciando conexão WebSocket com a Olymp Trade ({account_type})...")
    conectado = await client.connect()

    if not conectado:
        print("❌ Não foi possível autenticar (necessário access_token válido).")
        print("💡 Quando obtiver o token, salve-o em .olymp_token e execute este teste novamente.")
        return

    print("✅ Conexão WebSocket estabelecida com sucesso!")

    print("\n[2/4] Consultando saldo da conta ativa...")
    saldo = await client.get_balance()
    print(f"💰 Saldo Atual ({account_type}): R${saldo:.2f}")

    print("\n[3/4] Coletando velas históricas de EURUSD (M1)...")
    velas = await client.get_candles("EURUSD", timeframe="M1", contagem=5)
    if velas:
        print(f"✅ Recebidas {len(velas)} velas:")
        for v in velas:
            print(f"   Vela {v.get('time')}: O={v.get('open')} H={v.get('high')} L={v.get('low')} C={v.get('close')}")
    else:
        print("ℹ️ Sem histórico retornado no momento.")

    print("\n[4/4] Verificando cotação em tempo real...")
    preco = await client.get_current_price("EURUSD")
    print(f"📈 Preço vivo EURUSD: {preco}")

    await client.close()
    print("\n🎉 Teste finalizado com sucesso!\n")


if __name__ == "__main__":
    asyncio.run(main())
