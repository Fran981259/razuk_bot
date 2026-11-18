# 🏛️ Conector Olymp Trade (Fixed Time Trades - FTT)

Módulo dedicado e isolado para integração do **Razuk Bot** com a corretora **Olymp Trade**.

---

## 📁 Estrutura da Pasta

```
olymptrade/
├── __init__.py               → Exportações do pacote
├── olymp_constants.py        → URLs de WebSocket, mapeamento de pares e timeframes
├── olymp_client.py           → Cliente assíncrono WebSocket (OlympTradeClient)
├── test_connection.py        → Script CLI para testar login, saldo e cotações
├── .env.olymptrade.example   → Modelo de variáveis de ambiente da Olymp Trade
└── README.md                 → Este manual de integração
```

---

## 🔑 Como Obter o seu Token de Acesso (`access_token`)

A Olymp Trade possui proteção rigorosa por Cloudflare, impedindo login direto por usuário e senha via código. A forma mais estável e segura é utilizar o **Token de Sessão** do seu próprio navegador:

1. Abra o navegador (Google Chrome, Brave, Edge ou Firefox).
2. Acesse [olymptrade.com](https://olymptrade.com) e faça login na sua conta normalmente.
3. Abra as Ferramentas do Desenvolvedor pressionando **F12** (ou clique com botão direito -> *Inspecionar*).
4. Vá para a aba **Network** (Rede) e selecione o filtro **WS** (WebSockets).
5. Recarregue a página (`F5`).
6. Clique na conexão WebSocket ativa (geralmente chamada `otp` ou `v6`).
7. Na sub-aba **Payload** ou **Messages**, localize a mensagem de handshake inicial contendo `"access_token"`, ou copie o valor do token na URL/Headers.
8. Salve o token:
   - Crie um arquivo chamado `.olymp_token` na raiz do projeto contendo apenas o token; **OU**
   - Defina no seu arquivo `.env`: `OLYMP_TOKEN=seu_token_aqui`.

---

## 🧪 Como Testar a Conexão

Execute o script de teste isolado no terminal:

```bash
python3 olymptrade/test_connection.py
```

O teste executará:
1. Conexão WebSocket com os headers oficiais de navegador.
2. Handshake de autenticação com o seu `access_token`.
3. Consulta e exibição do saldo (DEMO / REAL).
4. Coleta de 5 velas de teste de `EURUSD`.
5. Leitura de preço vivo em tempo real.

---

## ⚙️ Compatibilidade com o Razuk Bot

A classe `OlympTradeClient` foi desenhada para seguir **exatamente a mesma interface** do `IqOptionClient`:
- `connect()` / `close()` / `is_connected`
- `get_balance()` / `change_balance()`
- `get_candles(par, timeframe, contagem)`
- `subscribe_candle(par)` / `get_current_price(par)`
- `buy(par, valor, direcao, expiracao_minutos)`
- `wait_for_option_result(order_id, timeout)`
- `get_payout(par)`

Isso garante que toda a inteligência do Razuk Bot (Machine Learning LightGBM, EMA Wave 34, MicroSniper e Gestão de Risco) pode operar na Olymp Trade sem misturar ou comprometer o funcionamento da IQ Option.
