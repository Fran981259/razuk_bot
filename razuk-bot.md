# 🗺️ razuk-bot — Project Map

> Project Map: read razuk-bot.md at root before any action.

## Stack
- Runtime: Python 3.14 (Linux x86_64)
- Machine Learning: LightGBM (LGBMClassifier), scikit-learn, joblib
- Data & Math: pandas, numpy
- Configuration: pydantic-settings, python-dotenv
- Network & Realtime: aiohttp, websockets
- UI & Visualization: Rich (Terminal TUI), Tkinter (Desktop GUI Dark Mode)
- Indicators: Pine Script v5 (TradingView), Lua (IQ Option native script)

## Folder Structure
/                   → Raiz do projeto
  conector/         → Pacote conector com a corretora IQ Option
  data/             → Datasets históricos OHLCV (90d), live_trades.csv e checkpoints
  models/           → Modelos serializados de IA (lgbm_model.joblib)

## Key Files & Modules
razuk-bot.md        → Project Map (Waze) para navegação ágil de qualquer IA/LLM
README.md           → Documentação técnica oficial e guia de arquitetura
main.py             → Ponto de entrada e orquestrador assíncrono (BotOrchestrator)
gui_dashboard.py    → Janela gráfica Desktop Dark Mode nativa em Tkinter
ui_dashboard.py     → Dashboard TUI visual no terminal em tempo real via Rich
iq_client.py        → Conector assíncrono WebSocket/REST nativo para IQ Option
iq_constants.py     → Mapeamento de pares, Active IDs e opções Turbo/Binárias
radar.py            → Scanner assíncrono dinâmico, Score Multicritério, Quarentena e cache de velas (Dict[str,list])
features.py         → Motor FeatureEngine v2 (14 Alpha Factors — feat_inside_bar e feat_ema_alignment removidos no Passo 1)
trend_sr.py         → Detector de estrutura: EMAs (9,21,50,200), Suporte/Resistência e Choppiness Index (CHOP)
ema_wave.py         → Indicador técnico EMA Wave 34 (High/Close/Low) e Clock Filter
ml_core.py          → Motor MLCore de inferência preditiva P(Win) em tempo real
online_learning.py  → Mecanismo adaptativo de aprendizado contínuo (5 trades / 30 min)
risk.py             → Gerenciador RiskManager (Kelly Fracionário, Drawdown, Cooldown, CVaR)
cvar_risk.py        → Gestão de risco de cauda: CVaR (Expected Shortfall) e amortecimento Kelly
smc_detector.py     → Motor Smart Money Concepts: Order Blocks e Fair Value Gaps (FVG)
volatility_models.py→ Modelos de volatilidade intradiária de amplitude (Parkinson e Garman-Klass)
quant_factors.py    → Fatores Alpha101 (Kakushadze) e Microsoft Qlib — desacoplado do loop principal, disponível para filtro futuro
telemetry.py        → Telemetria de latência p50/p95/p99 do Hot Path (latency-critical-systems)
errors.py           → Hierarquia de erros tipados de corretora e execução (error-handling)
train_ml.py         → Pipeline de treino supervisionado com TimeSeriesSplit (target proxy: próxima vela)
retrain_real_target.py → Retreino com target real (is_win) do live_trades.csv — ativado automaticamente ao atingir 1.500 amostras
collect_candles.py  → Coletor paginado de histórico retroativo de velas OHLCV
backtest_data.py    → Módulo de reamostragem M5 para M15 e tipos estatísticos
backtest_m15.py     → Backtest comparativo Out-of-Sample M15 (Técnica Pura vs Técnica + ML)
backtest_mtf_data.py→ Reamostragem concorrente M5, M15, M30 e H1
backtest_mtf_compare.py → Disputa MTF Independente vs Cascata Macro H1
test_bot.py         → Bateria de testes de ponta a ponta do fluxo assíncrono
dummy_conector.py   → Conector mock fallback para testes offline e simulações
.env.example        → Modelo de variáveis de ambiente com documentação dos parâmetros
.env                → Configurações ativas do usuário (protegido no .gitignore)
requirements.txt    → Dependências Python do projeto

## Database / Persistence
- data/candles_*_M5_90d.csv: 161.000 velas históricas M5 cobrindo 90 dias (7 pares: EURUSD, EURJPY, AUDUSD, AUDUSD-OTC, EURJPY-OTC, USDCHF-OTC, EURUSD-OTC)
- data/candles_*_M1_90d.csv: 348.855 velas históricas M1 legadas (AUDUSD-OTC, EURJPY-OTC, EURUSD-OTC, USDCHF-OTC)
- data/live_trades.csv: Registro contínuo de 570+ operações reais com snapshot das 14 features (schema v2, corrigido 2026-10-20 — 12 linhas com 19 campos reparadas para 21)
- data/checkpoint_before.json: Snapshot temporal de rollback do modelo v1
- data/checkpoint_after.json: Snapshot temporal de validação do modelo com Alpha Factors v2
- models/lgbm_model.joblib: Bundle LightGBM v2 — 14 features, AUC CV 0.5051, retreinado 2026-10-20 (160.980 amostras, target proxy)
- models/lgbm_model_rollback.joblib: Backup para rollback automático (mle-workflow, online learner)
- models/lgbm_model_proxy_backup.joblib: Backup gerado antes de cada retreino por retrain_real_target.py

## Regras Arquiteturais Fundamentais
1. Fidelidade de Ativo: Não há conversão forçada entre mercado Normal e OTC. O ativo analisado é estritamente o ativo operado.
2. Quarentena Inteligente: Pares com 2 perdas seguidas entram em quarentena de 10 min no radar.
3. Cooldown de Sequência: Desativado temporariamente para testes (COOLDOWN_MINUTES_AFTER_LOSS_STREAK=0). Reativar com 60 min para conta real.
5b. Modo Headless: `HEADLESS=1 python3 main.py` substitui AppWindow (Tkinter) por TerminalUI (Rich) e roda asyncio direto na thread principal — compatível com VPS sem X11.
4. Hot-Reload de IA: O modelo treina em background assíncrono e recarrega em memória sem interromper o loop.
5. Operação em M15: Timeframe padrão migrado e calibrado para 15 minutos (M15) com expiração estrita de 15 minutos (900s) alinhada ao fechamento da vela e opção binária institucional.
6. Catálogo Dinâmico e Ativos Abertos: O radar e o conector validam em tempo real via get-initialization-data se os pares estão abertos (enabled=True, is_suspended=False) e utilizam os Active IDs de Opções (18xx) e OTC oficiais.
7. Quality Gate de ML (mle-workflow): Retreinos adaptativos avaliam acurácia out-of-sample (holdout 15%) e rejeitam candidatos degradados, salvando snapshot de rollback. O `retrain_real_target.py` usa AUC CV > 0.50 como gate adicional.
8. Telemetria de Hot Path (latency-critical-systems): Rastreamento de latência em milissegundos de coleta, extração de features, inferência e envio de ordens.
9. Benchmark Loop & Gate Numérico (benchmark-optimization-loop): Otimizações numéricas exigem portão de equivalência numérica (diff=0.0) e medição de redução de latência.
10. Confluência Institucional SMC & Risco CVaR (Vibe-Trading): Entradas ganham confluência por mitigação de Order Blocks, filtram exaustão de Garman-Klass e amortecem Kelly via CVaR.

## Direção Final do Bot (Roadmap de Produção)
- [x] Conexão WebSocket assíncrona oficial de alta velocidade
- [x] Motor de 16 Alpha Factors com EMA Wave 34
- [x] Detector de Estrutura de Mercado: Médias Móveis (9,21,50,200) e Suporte/Resistência
- [x] Timeframe M15 nativo com sincronismo de vela de 900s e expiração de 15M (Assertividade 57.25% no backtest)
- [x] Inferência LightGBM com limiar de P(Win)
- [x] Aprendizado contínuo adaptativo (Online Learning) com retreino não-bloqueante
- [x] Radar dinâmico com score multicritério, verificação de ativos abertos, quarentena de pares e cache de velas (Dict[str,list])
- [x] Dashboards em tempo real (Desktop Tkinter Dark Mode e Terminal Rich)
- [x] Modo Headless (HEADLESS=1) para execução em VPS sem display
- [x] ML_THRESHOLD calibrado em 0.505 para M15 (+4.93% Win Rate e Profit Factor 1.14)
- [x] Online Learner reativado — live_trades.csv migrado para schema v2
- [x] Bug fix: fallback feat_ema_alignment removido de main.py (causava 52 min sem entradas)
- [x] record_trade() corrigido — alinha colunas com header existente do CSV
- [x] Retreino supervisionado com target real (retrain_real_target.py) concluído com 1.332 amostras reais filtradas (Accuracy Val: 61.0%, P(Win)@0.55: 66.2%, 26 features) e ativo em produção na VPS
- [x] Migração de Produção para M15 concluída e validada na VPS
- [x] Confluência em Cascata MTF (Macro H1 -> Gatilho M15) vencedora no backtest (59.40% WR, Profit Factor 1.24) promovida para produção
- [ ] Integrar quant_factors como filtro de confirmação (vibe_alpha101_12 contra absorção institucional)
- [ ] Ativação do Critério de Kelly automático para dimensionamento dinâmico
- [ ] Transição segura para Conta REAL com reativação das travas estritas
- [x] Deploy em VPS de produção (Ubuntu 24.04 na 13.140.149.158 via systemd + auto-restart + backup horário)
- [x] Filtro estrito de pares lucrativos (AUDUSD-OTC, EURJPY, USDJPY, USDCHF-OTC) e bloqueio de horas ruins expandido (2h, 4h, 13h, 18h, 21h, 22h, 23h UTC / 23h, 01h, 10h, 15h, 18h, 19h, 20h BRT)
- [x] Bug fix Headless: implementados sync_active_pairs e update_learning em TerminalUI (desbloqueia análises na VPS)
- [x] Pausa de Fim de Semana Automática: Repouso de proteção contra OTC de sábado/domingo (Sex 18:00 BRT até Dom 20:00 BRT)
- [x] Soberania do Machine Learning fixada no CerebroQuant: ML com 63.0% default e piso inquebrável de 45%
- [x] Motor MicroSniper & Telemetria em Tempo Real: Pullback estrutural C1 + Rejeição Tipo B + Janela 1/4 + Revalidação Ativa
- [x] Bug fix MicroSniper Telemetria: Subscrição dinâmica de streaming `candle-generated` no WebSocket e eliminação de cache estático no `get_current_price`
- [x] Resolução Nativa Multi-Timeframe M15: Coleta direta de 100 velas nativas por timeframe, eliminando erro de DataFrame curto (33 < 85)

## Last updated: 2026-11-05 (Retreino de Modelo com Target Real `is_win`, 26 Features e Deploy na VPS)

