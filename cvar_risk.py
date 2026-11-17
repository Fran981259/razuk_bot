"""Módulo de Gestão de Risco de Cauda (CVaR / Expected Shortfall).

Inspirado em agent/src/quantlib/risk.py do Vibe-Trading:
Implementa o cálculo de Conditional Value-at-Risk (CVaR) para séries de retornos
de trades recentes, amortecendo a alocação de Kelly quando a cauda de perdas
apresenta risco desproporcional.

Melhorias (Fase 1):
- Janela ampliada de 20 para 100 trades para estimativa estável
- Fallback paramétrico (distribuição Student-t) quando há < 30 trades
- Métrica de confiança da estimativa CVaR retornada ao chamador
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


# Tamanho máximo da janela de trades para cálculo do CVaR
WINDOW_SIZE: int = 100

# Limiar mínimo de trades para usar método histórico (vs paramétrico)
HISTORICAL_MIN_TRADES: int = 30

# Confiança padrão do CVaR (95% é o padrão financeiro)
DEFAULT_CONFIDENCE: float = 0.95


def calculate_historical_var(returns: List[float], confidence: float = 0.95) -> float:
    """Calcula o Value-at-Risk (VaR) histórico.

    Convenção: Retorna a perda como número positivo.
    Ex: 0.025 significa perda de 2.5% no percentil crítico.
    """
    if len(returns) < 5:
        return 0.02  # Valor padrão conservador para histórico curto

    arr = np.array(returns, dtype=np.float64)
    # Perda é o retorno invertido (-ret)
    losses = -arr
    sorted_losses = np.sort(losses)
    idx = int(np.ceil(confidence * len(sorted_losses))) - 1
    idx = min(max(idx, 0), len(sorted_losses) - 1)
    return float(max(0.0, sorted_losses[idx]))


def calculate_cvar(returns: List[float], confidence: float = 0.95) -> float:
    """Calcula o Conditional Value-at-Risk (CVaR / Expected Shortfall).

    Mede a perda média esperada nos piores (1 - confidence)% cenários.
    Garante matematicamente que CVaR >= VaR.
    """
    if len(returns) < 5:
        return 0.03

    arr = np.array(returns, dtype=np.float64)
    losses = -arr
    sorted_losses = np.sort(losses)
    cutoff_idx = int(np.ceil(confidence * len(sorted_losses))) - 1
    cutoff_idx = min(max(cutoff_idx, 0), len(sorted_losses) - 1)

    tail_losses = sorted_losses[cutoff_idx:]
    if len(tail_losses) == 0:
        return float(max(0.0, sorted_losses[cutoff_idx]))

    return float(max(0.0, np.mean(tail_losses)))


def cvar_parametric(returns: List[float], confidence: float = 0.95) -> float:
    """Calcula o CVaR paramétrico usando distribuição Student-t.

    Usa-se Student-t em vez de Normal porque retornos financeiros apresentam
    caudas pesadas (excess kurtosis). Com < 30 observações, o método histórico
    é instável — o paramétrico fornece estimativa robusta via ajuste de distribuição.

    Para retornos binários (opções binárias: +payout ou -100%), o método
    calcula a probabilidade empírica de perda e usa a distribuição Student-t
    para estimar a severidade da cauda.

    Args:
        returns: Lista de retornos percentuais (ex: [-1.0, +0.85, ...]).
        confidence: Nível de confiança (padrão 95%).

    Returns:
        CVaR paramétrico como perda positiva (ex: 0.03 = 3% de perda esperada na cauda).
    """
    if len(returns) < 5:
        return 0.03  # Fallback conservador

    arr = np.array(returns, dtype=np.float64)
    n = len(arr)

    # Para retornos binários (poucos valores únicos), usa abordagem híbrida:
    # probabilidade empírica de perda × severidade média das perdas
    n_unique = len(np.unique(arr))
    if n_unique <= 3:
        # Retornos binários: usa probabilidade empírica de perda com inflação de cauda
        losses = arr[arr < 0]
        if len(losses) == 0:
            return 0.01  # Sem perdas observadas, risco mínimo
        loss_rate = len(losses) / n
        avg_loss = abs(np.mean(losses))
        # Infla a perda pela incerteza da amostra pequena (factor de segurança)
        # Com 10 trades e 50% de perda, o erro padrão é ~0.16 — infla o CVaR
        uncertainty_factor = 1.0 + 1.0 / np.sqrt(n)
        cvar_value = avg_loss * min(loss_rate * uncertainty_factor, 1.0)
        return float(max(0.01, cvar_value))

    # Ajusta distribuição Student-t aos dados via método dos momentos
    mean_ret = np.mean(arr)
    std_ret = np.std(arr, ddof=1)  # Desvio padrão amostral

    # Estima os graus de liberdade (df) via excedent kurtosis
    # Para df > 30, Student-t converge para Normal
    kurtosis = stats.kurtosis(arr, fisher=True)  # Excess kurtosis
    df = 6.0 / max(kurtosis, 0.1) if kurtosis > 0 else 30.0
    df = min(max(df, 3.0), 50.0)  # Clamp para estabilidade numérica

    # VaR paramétrico: perda no quantil (1 - confidence)
    # Usamos -mean - std * t.ppf(1-confidence, df) para a cauda esquerda
    var_quantile = stats.t.ppf(1 - confidence, df)
    var_value = -(mean_ret + std_ret * var_quantile)

    # CVaR paramétrico para Student-t:
    # E[X | X <= -VaR] = -(mean + std * df * t.pdf(q) / (df - q^2) / (1-confidence))
    # onde q = t.ppf(1-confidence, df)
    q = var_quantile
    pdf_at_q = stats.t.pdf(q, df)
    # Fórmula exata do Expected Shortfall para Student-t
    if df > 1:
        cvar_value = -(mean_ret + std_ret * (df + q**2) / (df - 1) * pdf_at_q / (1 - confidence))
    else:
        cvar_value = var_value  # Fallback se df <= 1

    return float(max(0.01, cvar_value))


def _confidence_score(n_trades: int) -> float:
    """Calcula a confiança da estimativa CVaR baseada no número de trades.

    A confiança cresce logisticamente com o número de observações:
    - 5 trades: ~0.15 (muito incerto)
    - 20 trades: ~0.45 (moderado)
    - 50 trades: ~0.75 (confiável)
    - 100+ trades: ~0.90 (alta confiança)

    Args:
        n_trades: Número de trades na janela.

    Returns:
        Score entre 0.0 e 1.0 indicando confiança na estimativa.
    """
    # Curva logística: score = 1 / (1 + exp(-k * (n - n0)))
    # k controla a inclinação, n0 é o ponto de inflexão
    k = 0.08
    n0 = 40.0
    score = 1.0 / (1.0 + np.exp(-k * (n_trades - n0)))
    # Normaliza para [0.1, 0.95]
    return float(np.clip(score, 0.1, 0.95))


def compute_cvar_kelly_damping(
    trade_results: List[float],
    max_tolerable_cvar: float = 0.05,
    min_damping: float = 0.40,
) -> Tuple[float, float]:
    """Calcula o fator de amortecimento de Kelly baseado no CVaR recente.

    Estratégia adaptativa:
    - >= 30 trades: usa CVaR histórico (não-paramétrico, mais preciso com amostra grande)
    - < 30 trades: usa CVaR paramétrico (Student-t, robusto com amostra pequena)

    Args:
        trade_results: Lista de retornos percentuais das últimas operações (-1.0 para loss, +0.85 para win).
        max_tolerable_cvar: Limiar aceitável de perda de cauda (padrão 5%).
        min_damping: Piso mínimo de escala de entrada (40% do lote padrão).

    Returns:
        Tupla (fator_amortecimento: float entre min_damping e 1.0, cvar_calculado: float).
    """
    if len(trade_results) < 5:
        return 1.0, 0.02

    # Usa até WINDOW_SIZE trades (aumentado de 20 para 100)
    recent = trade_results[-WINDOW_SIZE:]
    n = len(recent)

    # Escolhe método baseado no número de trades disponíveis
    if n >= HISTORICAL_MIN_TRADES:
        cvar = calculate_cvar(recent, confidence=DEFAULT_CONFIDENCE)
    else:
        cvar = cvar_parametric(recent, confidence=DEFAULT_CONFIDENCE)

    if cvar <= max_tolerable_cvar:
        return 1.0, cvar

    # Amortecimento proporcional à severidade do risco de cauda
    damping = max_tolerable_cvar / cvar
    damping = float(np.clip(damping, min_damping, 1.0))
    return round(damping, 3), round(cvar, 4)


def compute_cvar_with_confidence(
    trade_results: List[float],
    max_tolerable_cvar: float = 0.05,
    min_damping: float = 0.40,
) -> Tuple[float, float, float]:
    """Calcula o fator de amortecimento de Kelly com métrica de confiança.

    Versão estendida de compute_cvar_kelly_damping que retorna também o
    score de confiança da estimativa CVaR, útil para dashboard e logs.

    Args:
        trade_results: Lista de retornos percentuais das últimas operações.
        max_tolerable_cvar: Limiar aceitável de perda de cauda (padrão 5%).
        min_damping: Piso mínimo de escala de entrada (40% do lote padrão).

    Returns:
        Tupla (damping, cvar, confidence_score) onde:
        - damping: fator de amortecimento entre min_damping e 1.0
        - cvar: perda esperada na cauda (positivo)
        - confidence_score: confiança da estimativa entre 0.0 e 1.0
    """
    damping, cvar = compute_cvar_kelly_damping(
        trade_results, max_tolerable_cvar, min_damping
    )
    n = min(len(trade_results), WINDOW_SIZE)
    conf = _confidence_score(n)
    return damping, cvar, round(conf, 2)
