# Razuk Bot — Quick Brief

**O que é**: Bot autônomo de opções binárias/Forex na IQ Option, timeframe M5, LightGBM + Kelly/CVaR + online learning.
**Código real**: `~/Área de trabalho/razuk bot/` (fora do segundo_cerebro)
**Status**: Validação em conta demo. 570 trades reais coletados. AUC CV 0.5051. ML_THRESHOLD = 0.55.
**Próximo passo**: Acumular até 1.500 trades para ativar retreino com target real (`retrain_real_target.py`).

## Regras
- Antes de qualquer ação: leia `razuk-bot.md` (Project Map) na raiz do projeto de código
- Máx 300 linhas/arquivo (GOLDCODE)
- Nunca operações bloqueantes na thread principal de `main.py`

## Para trabalhar
- Contexto completo → `razuk-bot.md` (no diretório de código)
- Arquitetura → `README.md`
- Online learning → `online_learning.py` + `retrain_real_target.py`
- Dados vivos → `data/live_trades.csv`

## ⚠️ Não fazer
- Não executar operações bloqueantes em `main.py` (async puro)
- Não misturar ativos regulares e OTC no catálogo
- Não confundir as features v1 (16) com v2 (14) — `feat_inside_bar` e `feat_ema_alignment` foram removidos
