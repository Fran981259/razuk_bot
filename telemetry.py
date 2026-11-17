"""Módulo de telemetria de latência e calibração ML para caminhos críticos.

Implementado conforme a skill 'latency-critical-systems'.
Mede tempos de resposta da corretora, cálculo de features, inferência LightGBM
e tempo de trânsito de ordens, computando estatísticas p50/p95/p99.

Melhorias (Fase 1):
- Curva de confiabilidade (reliability curve) para monitorar calibração do ML
- Métrica de slope da curva para validar que P(Win) calibrado reflete realidade
"""

from collections import deque
import logging
import time
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class LatencyTracker:
    """Monitor de latência em tempo real para o caminho crítico do robô."""

    def __init__(self, max_history: int = 100) -> None:
        self._logger = logging.getLogger("telemetry")
        self._latencies: Dict[str, Deque[float]] = {
            "candles": deque(maxlen=max_history),
            "features": deque(maxlen=max_history),
            "inference": deque(maxlen=max_history),
            "order": deque(maxlen=max_history),
            "total": deque(maxlen=max_history),
        }

    def record(self, segment: str, elapsed_ms: float) -> None:
        """Registra a duração em milissegundos de um segmento operacional."""
        if segment in self._latencies:
            self._latencies[segment].append(elapsed_ms)

    def get_percentiles(self, segment: str) -> Dict[str, float]:
        """Calcula p50, p95 e p99 para o segmento informado."""
        vals = sorted(self._latencies.get(segment, []))
        if not vals:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
        n = len(vals)
        p50 = vals[int(n * 0.50)]
        p95 = vals[min(int(n * 0.95), n - 1)]
        p99 = vals[min(int(n * 0.99), n - 1)]
        return {"p50": round(p50, 1), "p95": round(p95, 1), "p99": round(p99, 1), "count": n}

    def log_hot_path(
        self,
        par: str,
        t_candles_ms: float,
        t_features_ms: float,
        t_ml_ms: float,
        t_order_ms: Optional[float] = None,
    ) -> None:
        """Registra e emite log formatado dos tempos do ciclo operacional."""
        self.record("candles", t_candles_ms)
        self.record("features", t_features_ms)
        self.record("inference", t_ml_ms)
        total_ms = t_candles_ms + t_features_ms + (t_order_ms or 0.0)
        self.record("total", total_ms)

        order_str = f" | Ordem: {t_order_ms:.1f}ms" if t_order_ms is not None else ""
        self._logger.info(
            "⚡ [HOT-PATH] %s | Velas: %.1fms | Features: %.1fms | ML: %.1fms%s | Total: %.1fms",
            par, t_candles_ms, t_features_ms, t_ml_ms, order_str, total_ms,
        )


def reliability_curve(
    predicted_probs: List[float],
    actual_outcomes: List[int],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Calcula a curva de confiabilidade (reliability curve) do modelo ML.

    Divide as previsões em bins e compara a probabilidade média prevista
    com a taxa de acerto real em cada bin. Um modelo bem calibrado tem
    slope próximo de 1.0 na regressão linear entre bins.

    Args:
        predicted_probs: Probabilidades previstas P(Win) pelo modelo.
        actual_outcomes: Resultados reais (1 = win, 0 = loss).
        n_bins: Número de bins para a curva (padrão 10).

    Returns:
        Dicionário com:
        - slope: inclinação da regressão linear (1.0 = calibração perfeita)
        - intercept: intercepto da regressão (0.0 = calibração perfeita)
        - calibration_error: erro médio absoluto entre previsto e real
        - n_samples: número de amostras usadas
        - bin_details: lista de (bin_center, predicted, actual) por bin
    """
    if len(predicted_probs) < 10 or len(actual_outcomes) < 10:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "calibration_error": 1.0,
            "n_samples": len(predicted_probs),
            "bin_details": [],
        }

    probs = np.array(predicted_probs, dtype=np.float64)
    outcomes = np.array(actual_outcomes, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_predicted = []
    bin_actual = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        if np.sum(mask) > 0:
            bin_centers.append((low + high) / 2.0)
            bin_predicted.append(np.mean(probs[mask]))
            bin_actual.append(np.mean(outcomes[mask]))

    if len(bin_centers) < 2:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "calibration_error": 1.0,
            "n_samples": len(probs),
            "bin_details": [],
        }

    # Regressão linear: actual = slope * predicted + intercept
    bc = np.array(bin_centers)
    bp = np.array(bin_predicted)
    ba = np.array(bin_actual)

    # Pondera pelo número de amostras em cada bin
    weights = np.ones(len(bc))
    slope, intercept = np.polyfit(bp, ba, 1, w=weights)

    # Calibration error (MAE entre previsto e real por bin)
    calibration_error = float(np.mean(np.abs(bp - ba)))

    bin_details = list(zip(bc.tolist(), bp.tolist(), ba.tolist()))

    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "calibration_error": round(calibration_error, 4),
        "n_samples": len(probs),
        "bin_details": bin_details,
    }
