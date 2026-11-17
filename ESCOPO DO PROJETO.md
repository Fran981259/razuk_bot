# 📋 ESCOPO DO PROJETO (PROJECT CHARTER)
## Bot Autônomo de Trading Quantitativo com Machine Learning para Opções Binárias

---

### 1. RESUMO EXECUTIVO E OBJETIVOS
* **Propósito**: Solução algorítmica em Python operando de forma 100% autônoma via WebSockets nativos com a corretora IQ Option em gráficos M5.
* **Mecanismo de Decisão**: Price Action, confluência institucional de **Smart Money Concepts (SMC)**, **EMA Wave 34 (Raghee Horner)**, **Choppiness Index (CHOP)** e indicadores quantitativos como gatilhos primários, filtrados por **Machine Learning (LightGBM)** ($P(\text{Win}) \ge \text{ML\_THRESHOLD}$).
* **Expectativa Matemática**: Elevar a taxa de acerto (*Win Rate*) para a faixa de **55% a 65%**, superando o ponto de equilíbrio (*breakeven* de 54,05% para payouts de 85%).
* **Preservação de Capital**: **Critério de Kelly Fracionário amortecido por CVaR (Expected Shortfall)**, limiar fixo de risco por trade (1% a 2%) e travas automatizadas de *Stop Loss* diário e quarentena por par.

---

### 2. ARQUITETURA DE SOFTWARE E MÓDULOS

O sistema opera em pipeline sequencial desacoplado e assíncrono (`asyncio`):

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. CONFIGURAÇÃO & CONEXÃO (config.py + iq_client.py) [CONCLUÍDO]        │
│    Carrega .env, gerencia catálogo dinâmico de 113+ ativos e WS de 50MB.│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. ENGINE QUANTITATIVA & SMC (features.py + smc_detector.py) [CONCLUÍDO]│
│    Calcula 16 Alphas vetorizados em ~8ms e detecta Order Blocks/FVG.    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. GESTÃO DE RISCO DE CAUDA (risk.py + cvar_risk.py) [CONCLUÍDO]        │
│    Kelly Fracionário amortecido dinamicamente pelo CVaR das operações. │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. MACHINE LEARNING & QUALITY GATE (ml_core.py + online_learning.py)    │
│    Inferência P(Win) e retreino adaptativo com validação Out-of-Sample. │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. ORQUESTRADOR, TELEMETRIA & UI (main.py + telemetry.py) [CONCLUÍDO]   │
│    Virada de vela M5, telemetria hot-path e execução com parada suave.  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 3. ESPECIFICAÇÃO DAS ESTRATÉGIAS TÉCNICAS E QUANTITATIVAS

1. **FeatureEngine v2.1 Otimizada**:
   * Cálculo em bloco com dicionário de arrays sem fragmentação de memória Pandas (latência reduzida em 40%).
   * 16 variáveis tratadas (Alpha Factors de momentum, RSI, MACD, ROC, CCI normalizado, Bollinger %B e proporções geométricas de pavio).
2. **Smart Money Concepts (SMC)**:
   * Detecção de **Order Blocks (OB)** e **Fair Value Gaps (FVG)** institucionais para autorização de trades na mitigação de liquidez.
3. **Estrutura de Mercado & Rompimento de Lateralização**:
   * Médias móveis (9, 21, 50, 200), suporte/resistência fractal e **Choppiness Index (CHOP)** para identificação de rompimentos autênticos fora da consolidação.
4. **Volatilidade de Amplitude de Parkinson & Garman-Klass**:
   * Medição estocástica de volatilidade com detecção e bloqueio de regimes de sobre-extensão extrema (`EXTREME_EXHAUSTION`).

---

### 4. PIPELINE DE MACHINE LEARNING E ADAPTAÇÃO CONTÍNUA

* **Algoritmo**: `LightGBM` (LGBMClassifier).
* **Alvo (Target)**: Variável binária (`1` para Win e `0` para Loss na expiração M5).
* **Treinamento Supervisionado**: Validação sequencial cronológica (`TimeSeriesSplit`).
* **Quality Gate de ML (MLE Workflow)**: A cada 5 operações reais, o modelo re-treina em segundo plano ancorado no histórico de 90 dias com peso exponencial de recência. O novo modelo só é promovido se superar o modelo anterior em 15% de holdout temporal estrito (*Out-of-Sample*), mantendo snapshot de rollback.

---

### 5. GESTÃO DE RISCO E MATEMÁTICA FINANCEIRA

* **Kelly Fracionário com Damping CVaR**:
  $$f^* = \text{Damping}_{\text{CVaR}} \times 0.15 \times \left( \frac{p \cdot (b + 1) - 1}{b} \right)$$
* **Teto de Risco**: Limite máximo de **1% a 2%** do capital por trade.
* **Stop Loss Diário**: Bloqueio operacional automático ao atingir **5% de drawdown** na banca.
* **Quarentena Inteligente por Par**: 2 perdas seguidas colocam o par em pausa de 10 minutos no radar.

---

### 6. STATUS DOS MÓDULOS E CRONOGRAMA

| Fase | Módulo | Arquivos Envolvidos | Status Atual |
| :--- | :--- | :--- | :--- |
| **Fase 1** | **Configuração & Conector Oficial** | `config.py`, `iq_client.py`, `iq_constants.py` | ✅ Concluído |
| **Fase 2** | **Engine de Features v2.1 Vetorizada** | `features.py`, `quant_factors.py` | ✅ Concluído |
| **Fase 3** | **Estrutura, SMC & Volatilidade** | `trend_sr.py`, `smc_detector.py`, `volatility_models.py` | ✅ Concluído |
| **Fase 4** | **Gerenciador de Risco & CVaR** | `risk.py`, `cvar_risk.py` | ✅ Concluído |
| **Fase 5** | **ML, Online Learning & Quality Gate** | `ml_core.py`, `online_learning.py`, `train_ml.py` | ✅ Concluído |
| **Fase 6** | **Telemetria de Hot Path & Erros Tipados** | `telemetry.py`, `errors.py` | ✅ Concluído |
| **Fase 7** | **Orquestrador, Radar & UI Dark Mode** | `main.py`, `radar.py`, `gui_dashboard.py` | ✅ Concluído |
| **Fase 8** | **Validação em Conta Demo (M5)** | `live_trades.csv`, checkpoints | 🟡 Em Andamento |
| **Fase 9** | **Operação em Conta Real (Produção)** | Transição para REAL + Webhook Telegram | ⏳ Planejado |

---
⚡ GOLDCODE rules applied — global_rules.md v3.0
