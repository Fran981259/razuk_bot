# ⚡ RAZUK BOT QUANT — Documentação Oficial & Guia de Arquitetura

Bot autônomo de trading quantitativo para Opções Binárias e Forex (IQ Option) operando em **M5**. Combina filtros estatísticos via **Machine Learning (LightGBM)**, aprendizado contínuo adaptativo (**Online Learning**), gestão de capital com **Critério de Kelly Fracionário amortecido por CVaR**, confluência institucional de **Smart Money Concepts (SMC)**, métricas de **Volatilidade de Parkinson/Garman-Klass** e dashboards em tempo real (**Desktop Tkinter Dark Mode** e **Terminal Rich TUI**).

---

## 📋 Sumário
1. [Visão Geral & Estado do Projeto](#-visão-geral--estado-do-projeto)
2. [Arquitetura do Sistema](#-arquitetura-do-sistema)
3. [Módulos do Projeto](#-módulos-do-projeto)
4. [Estratégia Quantitativa & Confluência SMC](#-estratégia-quantitativa--confluência-smc)
5. [Radar de Pares & Catálogo Dinâmico](#-radar-de-pares--catálogo-dinâmico)
6. [Gestão de Risco & Risco de Cauda (CVaR)](#-gestão-de-risco--risco-de-cauda-cvar)
7. [Instalação e Configuração](#-instalação-e-configuração)
8. [Como Executar](#-como-executar)
9. [Direção Final do Bot (Roadmap de Produção)](#-direção-final-do-bot-roadmap-de-produção)
10. [Instruções para IAs & LLMs](#-instruções-para-ias--llms)

---

## 🎯 Visão Geral & Estado do Projeto

O **Razuk Bot** opera com autonomia total em servidores Linux/VPS. Ele elimina o ruído emocional combinando 4 camadas:
1. **Estrutura Técnica & Maré**: EMA Wave 34 (Raghee Horner), alinhamento em cascata (9, 21, 50, 200), suporte/resistência fractal e **Choppiness Index (CHOP)** para detecção de rompimento de consolidação.
2. **Confluência Institucional Smart Money Concepts (SMC)**: Mapeamento de **Order Blocks (OB)** e **Fair Value Gaps (FVG)** com detecção de mitigação institucional.
3. **Filtro Estatístico de Machine Learning (LightGBM)**: Avaliação de um vetor de **14 Alpha Factors** na virada de cada vela M5, autorizando apenas operações com $P(\text{Win}) \ge \text{ML\_THRESHOLD}$.
4. **Aprendizado Contínuo com Quality Gate (MLE Workflow)**: A cada 5 operações reais, o modelo re-treina em segundo plano e só é promovido se for aprovado em validação *Out-of-Sample* (holdout 15%), mantendo backup de rollback.

### 📊 Estado Atual de Desenvolvimento:
* **Fase**: Validação e Estabilização em Conta Practice (Demo) no timeframe M5.
* **Base de Dados**: 161.000 velas históricas M5 (90 dias em 7 pares) + 570 trades reais coletados (schema v2, 14 features).
* **Modelo Ativo**: LightGBM v2 treinado em M5 com 14 features, AUC CV 0.5051, retreinado em 20/10/2026 (`models/lgbm_model.joblib`).
* **Conector**: Driver assíncrono WebSocket nativo (`iq_client.py`) com sincronização de catálogo e payloads de 50MB.
* **Online Learner**: Ativo — `record_trade()` corrigido para respeitar header do CSV existente; retreino dispara a cada 5 trades ou 30 min.
* **Retreino por Target Real**: `retrain_real_target.py` ativado automaticamente (scheduled task a cada 30 min) ao atingir 1.500 trades reais.

---

## 🏛️ Arquitetura do Sistema

```text
┌──────────────────────────────────────────────────────────────────┐
│                   THREAD PRINCIPAL (Interface)                   │
│  [gui_dashboard.py] Janela Desktop Dark Mode (atualização 1Hz)   │
│  [ui_dashboard.py]  Dashboard Terminal Responsivo via Rich       │
└─────────────────────────────────▲────────────────────────────────┘
                                  │ (comunicação thread-safe)
┌─────────────────────────────────┴────────────────────────────────┐
│                THREAD ASSÍNCRONA WORKER (asyncio)                │
│  [main.py] BotOrchestrator: sincronização na virada de vela M5   │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────────┐ │
│  │    radar.py     │───▶│   features.py   │───▶│  ml_core.py   │ │
│  │(Score / Catálogo│    │  (16 Alphas)    │    │  (LightGBM)   │ │
│  └────────┬────────┘    └────────┬────────┘    └───────┬───────┘ │
│           │                      │                     │         │
│           │             ┌────────▼────────┐            │         │
│           │             │ smc_detector.py │            ▼         │
│           │             │ (Order Blocks)  │     [P(Win) >= 52%]  │
│           │             └────────┬────────┘            │         │
│           ▼                      │                     ▼         │
│  ┌─────────────────┐             ▼             ┌───────────────┐ │
│  │  iq_client.py   │◀── [Confluência SMC] ◀─── │    risk.py    │ │
│  │ (WebSocket/REST)│                           │ (Kelly + CVaR)│ │
│  └────────┬────────┘                           └───────────────┘ │
│           │ (resultado option-closed)                            │
│           ▼                                                      │
│  ┌─────────────────┐                                             │
│  │online_learning.p│ ──▶ [Quality Gate ML / Rollback] ──▶ ml_core │
│  └─────────────────┘                                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📦 Módulos do Projeto

| Arquivo | Responsabilidade Técnica |
| :--- | :--- |
| [main.py](file:///home/razuk/Área de trabalho/razuk bot/main.py) | Orquestrador assíncrono (`BotOrchestrator`): virada de vela M5, telemetria hot-path e despacho de ordens. |
| [radar.py](file:///home/razuk/Área de trabalho/razuk bot/radar.py) | Scanner de alta velocidade, catálogo dinâmico de ativos abertos (OTC e regular), quarentena de perdas e **cache de velas** (`Dict[str, list]`) — elimina double fetch por ciclo. |
| [features.py](file:///home/razuk/Área de trabalho/razuk bot/features.py) | Motor quantitativo vetorizado (`FeatureEngine v2`): 14 Alpha Factors calculados em ~8ms (`feat_inside_bar` e `feat_ema_alignment` removidos no Passo 1). |
| [trend_sr.py](file:///home/razuk/Área de trabalho/razuk bot/trend_sr.py) | Detector de tendência por médias móveis (9,21,50,200), suporte/resistência e Choppiness Index (CHOP). |
| [smc_detector.py](file:///home/razuk/Área de trabalho/razuk bot/smc_detector.py) | Módulo Smart Money Concepts (Vibe-Trading): rastreador de Order Blocks e Fair Value Gaps (FVG). |
| [volatility_models.py](file:///home/razuk/Área de trabalho/razuk bot/volatility_models.py) | Modelos de amplitude de Parkinson e Garman-Klass com detecção de regimes e filtro de exaustão. |
| [quant_factors.py](file:///home/razuk/Área de trabalho/razuk bot/quant_factors.py) | Fatores matemáticos Alpha101 (Kakushadze) e Microsoft Qlib para momentum e absorção institucional. Disponível para uso futuro como filtro de confirmação (desacoplado do loop principal). |
| [risk.py](file:///home/razuk/Área de trabalho/razuk bot/risk.py) | Gerenciador financeiro: Critério de Kelly Fracionário amortecido dinamicamente por cauda de risco. |
| [cvar_risk.py](file:///home/razuk/Área de trabalho/razuk bot/cvar_risk.py) | Cálculo de Value-at-Risk (VaR) e Conditional VaR (Expected Shortfall) de trades recentes. |
| [online_learning.py](file:///home/razuk/Área de trabalho/razuk bot/online_learning.py) | Aprendizado adaptativo contínuo com Quality Gate Out-of-Sample (15%) e snapshots de rollback. `record_trade()` corrigido para alinhar colunas com o header existente do CSV. |
| [retrain_real_target.py](file:///home/razuk/Área de trabalho/razuk bot/retrain_real_target.py) | Retreino com target real (`is_win`) do `live_trades.csv`. Ativado automaticamente pela scheduled task ao atingir 1.500 amostras. Quality Gate: AUC CV > 0.50. |
| [telemetry.py](file:///home/razuk/Área de trabalho/razuk bot/telemetry.py) | Rastreador de latência de hot-path em nanossegundos/milissegundos com métricas p50/p95/p99. |
| [errors.py](file:///home/razuk/Área de trabalho/razuk bot/errors.py) | Hierarquia de erros tipados de corretora, cálculo e inferência (`BrokerOrderRefusalError`, etc.). |
| [ml_core.py](file:///home/razuk/Área de trabalho/razuk bot/ml_core.py) | Motor preditivo LightGBM com inferência calibrada $P(\text{Win})$ e hot-reload não-bloqueante. |
| [ema_wave.py](file:///home/razuk/Área de trabalho/razuk bot/ema_wave.py) | Indicador proprietário EMA Wave 34 (High/Close/Low) com Clock Filter analógico (Raghee Horner). |
| [gui_dashboard.py](file:///home/razuk/Área de trabalho/razuk bot/gui_dashboard.py) | Interface desktop nativa Dark Mode (Tkinter) com feed de ordens, quarentena e status de IA. |
| [iq_client.py](file:///home/razuk/Área de trabalho/razuk bot/iq_client.py) | Driver assíncrono WebSocket/REST nativo para IQ Option com suporte a WebSocket frames de 50MB. |
| [iq_constants.py](file:///home/razuk/Área de trabalho/razuk bot/iq_constants.py) | Mapeamento centralizado de Active IDs de Opções Turbo/Binárias (18xx) e OTC oficiais. |
| [razuk-bot.md](file:///home/razuk/Área de trabalho/razuk bot/razuk-bot.md) | Project Map (Waze): mapa de navegação rápido e atualizado para qualquer IA/LLM. |

---

## 🎯 Estratégia Quantitativa & Confluência SMC

1. **Gatilho de Rompimento de Lateralização (CHOP + S/R Breakout)**:
   - Se o relógio analógico estiver neutro, mas o Choppiness Index acusar saída de congestão ($\text{CHOP} < 58.0$) com rompimento de suporte/resistência, o bot autoriza o trade a favor do rompimento.
2. **Mitigação Institucional SMC**:
   - Quando o preço recua para testar um Order Block ativo de alta ou baixa, o [smc_detector.py](file:///home/razuk/Área%20de%20trabalho/razuk%20bot/smc_detector.py) sinaliza confluência com as instituições.
3. **Filtro de Exaustão de Volatilidade de Garman-Klass**:
   - Caso a volatilidade da vela recente supere em 2.5x a média histórica (`EXTREME_EXHAUSTION`), a ordem é bloqueada para prevenir compras no topo ou vendas no fundo.

---

## 🛡️ Gestão de Risco & Risco de Cauda (CVaR)

Configurado em [risk.py](file:///home/razuk/Área%20de%20trabalho/razuk%20bot/risk.py) e [cvar_risk.py](file:///home/razuk/Área%20de%20trabalho/razuk%20bot/cvar_risk.py):
* **Critério de Kelly Amortecido por CVaR**:
  $$f^* = \text{Damping}_{\text{CVaR}} \times 0.15 \times \left( \frac{p \cdot (b + 1) - 1}{b} \right)$$
  *(Se a cauda recente de perdas indicar risco desproporcional $\text{CVaR} > 5\%$, o lote é reduzido automaticamente)*.
* **Stop Loss Diário**: Trava automática caso o drawdown acumulado atinja o teto (`MAX_DAILY_DRAWDOWN_PCT=0.05`).
* **Quarentena Inteligente por Par**: Pares com 2 derrotas consecutivas entram em pausa de 10 minutos no radar.

---

## ⚙️ Instalação e Execução

```bash
# 1. Ativar o ambiente virtual e instalar dependências
source .venv/bin/activate
pip install -r requirements.txt

# 2. Executar a bateria de testes de conformidade
python3 test_bot.py

# 3a. Iniciar com Interface Gráfica Desktop Dark Mode (padrão)
python3 main.py

# 3b. Iniciar em modo headless — VPS/servidor sem display (usa Terminal TUI Rich)
HEADLESS=1 python3 main.py
```

---

## 🤖 Instruções para IAs & LLMs (GOLDCODE Standard)

1. **Consulte sempre o [razuk-bot.md](file:///home/razuk/Área de trabalho/razuk bot/razuk-bot.md) antes de qualquer ação**: Ele é o GPS obrigatório do projeto.
2. **Teto estrito de 300 linhas por arquivo**: Nenhum arquivo novo ou modificado deve ultrapassar 300 linhas (salvo exceções marcadas).
3. **Fidelidade de Ativos**: Nunca force conversões artificiais entre ativos regulares e OTC.
4. **Preserve a assincronicidade**: Nunca execute operações bloqueantes na thread principal do `main.py`.

---
---
## 📋 Changelog de Otimizações (Fases 1–3)

| Fase | Tarefa | Arquivo | Impacto |
|------|--------|---------|---------|
| 1 | `ML_THRESHOLD` 0.52 → **0.55** | `.env`, `config.py` | Sai do EV negativo (breakeven ≈ 54.1% para payout 85%) |
| 1 | Remover imports duplicados e campos TMZ legados | `main.py`, `config.py` | Código limpo, sem dead config |
| 2 | Migrar `live_trades.csv` schema v1 → v2 | `data/live_trades.csv` | Online Learner reativado (552 trades históricos) |
| 2 | Suporte `HEADLESS=1` | `main.py` | Compatível com VPS/servidor sem X11 |
| 3 | Cache de velas no Radar (`Dict[str, list]`) | `radar.py`, `main.py` | −272 ms/ciclo (elimina double fetch WebSocket) |
| 3 | Remover `quant_alphas` morto | `main.py` | −14 ms/vela de CPU desperdiçado |
| 4 | **Bug crítico**: fallback `feat_ema_alignment` em `main.py` | `main.py` | Corrigido bloqueio de 52 min sem entradas — direção agora derivada diretamente do `ClockState` |
| 4 | Remover `feat_inside_bar` e `feat_ema_alignment` de `FEATURE_COLUMNS` | `features.py` | FeatureEngine v2: 14 features (de 16); modelo retreinado em 160.980 amostras |
| 4 | `record_trade()` respeita header existente do CSV | `online_learning.py` | Elimina corrupção de linhas (19 campos vs 21) no `live_trades.csv` |
| 4 | Correção de 12 linhas corrompidas no `live_trades.csv` | `data/live_trades.csv` | 570 trades válidos agora disponíveis para retreino |
| 4 | Validação holdout real: `live_trades.csv` vs modelo | diagnóstico | AUC 0.47 no holdout real — proxy target histórico não generaliza para opções binárias |
| 4 | `retrain_real_target.py` + scheduled task 30 min | `retrain_real_target.py` | Retreino automático com `is_win` ao atingir 1.500 trades reais |

---
⚡ GOLDCODE rules applied — global_rules.md v3.0
