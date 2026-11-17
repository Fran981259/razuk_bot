"""CerebroQuant — Orquestrador central de votos com pesos adaptativos.

Quatro agentes especialistas emitem scores [0, 1] para a direção proposta.
O Cérebro combina via Σ(score_i × peso_i) e decide se entra.
Pesos se adaptam após cada trade: recompensa quem acertou, penaliza quem errou.
Agente com peso abaixo de SILENCIO_LIMIAR é temporariamente silenciado.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_LOG = logging.getLogger("cerebro_quant")

# ── Tipos ─────────────────────────────────────────────────────────────────

@dataclass
class VotoAgente:
    nome: str
    score: float   # [0, 1] — confiança na direção proposta
    ativo: bool = True


@dataclass
class DecisaoCerebro:
    entrar: bool
    score_final: float
    votos: Dict[str, float]
    pesos: Dict[str, float]
    direcao: Optional[str]
    motivo: str


# ── Agentes Especialistas ─────────────────────────────────────────────────

class AgenteTendencia:
    """A1 — Força de tendência: ADX, CHOP e slope da EMA21."""
    nome = "tendencia"

    def votar(self, features_df, wave_sig, smc_res, trend_res, vol_rep,
              direcao: Optional[str], **_) -> VotoAgente:
        if features_df.empty or not direcao:
            return VotoAgente(self.nome, 0.0)

        u = features_df.iloc[-1]
        adx        = float(u.get("feat_adx", 0.0))
        ema_slope  = float(u.get("feat_ema21_slope", 0.0))
        chop       = getattr(trend_res, "chop_index", 55.0)

        adx_score   = min(adx / 40.0, 1.0)
        chop_score  = max(0.0, min(1.0, 1.0 - (chop - 38.2) / (61.8 - 38.2)))
        slope_ok    = (ema_slope > 0 and direcao == "CALL") or \
                      (ema_slope < 0 and direcao == "PUT")
        slope_score = min(abs(ema_slope) * 500.0, 1.0) if slope_ok else 0.0

        score = 0.50 * adx_score + 0.30 * chop_score + 0.20 * slope_score
        return VotoAgente(self.nome, round(score, 4))


class AgenteVolatilidade:
    """A2 — Segurança de entrada: pavio, Bollinger, Z-Score e regime GK."""
    nome = "volatilidade"

    def votar(self, features_df, wave_sig, smc_res, trend_res, vol_rep,
              direcao: Optional[str], **_) -> VotoAgente:
        if features_df.empty or not direcao:
            return VotoAgente(self.nome, 0.0)

        u      = features_df.iloc[-1]
        pct_b  = float(u.get("feat_bollinger_pct_b", 0.5))
        pavio  = float(u.get("feat_upper_shadow", 0.0)) + float(u.get("feat_lower_shadow", 0.0))
        zscore = float(u.get("feat_zscore", 0.0))

        is_safe    = getattr(vol_rep, "is_safe_for_trend", True)
        vol_score  = 1.0 if is_safe else 0.1

        if direcao == "CALL":
            bb_score = max(0.0, 1.0 - (pct_b - 0.7) / 0.3) if pct_b > 0.7 else 1.0
        else:
            bb_score = max(0.0, pct_b / 0.3) if pct_b < 0.3 else 1.0

        pavio_score  = max(0.0, 1.0 - pavio * 2.0)
        zscore_ok    = (zscore < 0.70 and direcao == "CALL") or \
                       (zscore > -0.70 and direcao == "PUT")
        zscore_score = 1.0 if zscore_ok else 0.35

        score = 0.30 * vol_score + 0.25 * bb_score + \
                0.25 * pavio_score + 0.20 * zscore_score
        return VotoAgente(self.nome, round(score, 4))


class AgenteInstitucional:
    """A3 — Confluência SMC + alinhamento de tendência + COG."""
    nome = "institucional"

    def votar(self, features_df, wave_sig, smc_res, trend_res, vol_rep,
              direcao: Optional[str], **_) -> VotoAgente:
        if not direcao:
            return VotoAgente(self.nome, 0.0)

        try:
            from trend_sr import TrendDirection
            td       = getattr(trend_res, "direction", None)
            trend_ok = (td == TrendDirection.ALTA  and direcao == "CALL") or \
                       (td == TrendDirection.BAIXA and direcao == "PUT")
            trend_score = 0.9 if trend_ok else 0.3
        except Exception:
            trend_score = 0.5

        inst = getattr(smc_res, "institutional_signal", None)
        smc_score = 0.9 if inst == direcao else (0.5 if inst is None else 0.1)

        cog_score = 0.5
        if not features_df.empty:
            cog = float(features_df.iloc[-1].get("feat_cog", 0.0))
            cog_ok    = (cog > 0 and direcao == "CALL") or (cog < 0 and direcao == "PUT")
            cog_score = 0.85 if cog_ok else 0.25

        score = 0.35 * trend_score + 0.40 * smc_score + 0.25 * cog_score
        return VotoAgente(self.nome, round(score, 4))


class AgenteML:
    """A4 — Probabilidade estatística: P(Win) do LightGBM."""
    nome = "ml"

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold

    def votar(self, features_df, wave_sig, smc_res, trend_res, vol_rep,
              direcao: Optional[str], prob_win: float = 0.0, **_) -> VotoAgente:
        if not direcao or prob_win == 0.0:
            return VotoAgente(self.nome, 0.0)

        if prob_win < self.threshold:
            score = (prob_win / self.threshold) * 0.40
        else:
            score = 0.50 + (prob_win - self.threshold) / (1.0 - self.threshold) * 0.50

        return VotoAgente(self.nome, round(score, 4))


class AgenteHistorico:
    """A5 — Memória Empírica: k-NN no live_trades.csv.

    Antes de cada trade, encontra os K trades históricos mais similares por
    cosine similarity do vetor de features e retorna o WR empírico desses
    vizinhos como score. Responde: 'da última vez que o mercado estava assim
    na mesma direção, o bot ganhou?'

    Fica neutro (0.5) durante cold start (< MIN_AMOSTRAS trades no CSV).
    """
    nome = "historico"
    _K               = 15     # vizinhos mais próximos a consultar
    _MIN_AMOSTRAS    = 80     # abaixo disso retorna neutro — dados insuficientes
    _MIN_VIZINHOS    = 5      # mínimo de vizinhos na mesma direção para confiar
    _REFRESH_INTERVALO = 60.0 # recarrega CSV no máximo 1x por minuto
    _CSV = Path(__file__).parent / "data" / "live_trades.csv"

    def __init__(self) -> None:
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._dirs: Optional[np.ndarray] = None
        self._cols: List[str] = []
        self._ultimo_load: float = 0.0
        self._ultimo_mtime: float = 0.0

    def _carregar(self) -> None:
        """Recarrega o CSV se o arquivo mudou ou expirou o intervalo de cache."""
        try:
            from features import FEATURE_COLUMNS
            mtime = self._CSV.stat().st_mtime if self._CSV.exists() else 0.0
            agora = time.monotonic()
            if (agora - self._ultimo_load < self._REFRESH_INTERVALO
                    and mtime == self._ultimo_mtime):
                return
            if not self._CSV.exists():
                return
            df = pd.read_csv(self._CSV)
            cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            df = df[df["is_win"].isin([0, 1])].dropna(
                subset=cols + ["is_win", "direction"]
            )
            if len(df) < self._MIN_AMOSTRAS:
                return
            self._X    = df[cols].values.astype(float)
            self._y    = df["is_win"].values.astype(int)
            self._dirs = df["direction"].values
            self._cols = cols
            self._ultimo_load  = agora
            self._ultimo_mtime = mtime
            _LOG.debug("AgenteHistorico: CSV recarregado (%d linhas)", len(df))
        except Exception as exc:
            _LOG.debug("AgenteHistorico._carregar falhou: %s", exc)

    def votar(self, features_df, wave_sig, smc_res, trend_res, vol_rep,
              direcao: Optional[str], **_) -> VotoAgente:
        if not direcao or features_df.empty:
            return VotoAgente(self.nome, 0.5)

        self._carregar()

        if self._X is None or len(self._X) < self._MIN_AMOSTRAS:
            return VotoAgente(self.nome, 0.5)  # cold start: neutro

        try:
            cols_ok = [c for c in self._cols if c in features_df.columns]
            q = features_df.iloc[-1][cols_ok].values.astype(float)
            q = np.nan_to_num(q, nan=0.0)

            # Filtra apenas trades na mesma direção
            mask  = self._dirs == direcao
            X_dir = np.nan_to_num(self._X[mask][:, :len(cols_ok)], nan=0.0)
            y_dir = self._y[mask]

            if len(X_dir) < self._MIN_VIZINHOS:
                return VotoAgente(self.nome, 0.5)

            # Cosine similarity
            norm_X = np.linalg.norm(X_dir, axis=1)
            norm_q = np.linalg.norm(q)
            if norm_q < 1e-9:
                return VotoAgente(self.nome, 0.5)
            cos = X_dir.dot(q) / (norm_X * norm_q + 1e-9)

            # Top K vizinhos
            k    = min(self._K, len(X_dir))
            idx  = np.argpartition(cos, -k)[-k:]
            wr   = float(y_dir[idx].mean())

            _LOG.debug(
                "AgenteHistorico: k=%d, direção=%s, WR empírico=%.1f%%",
                k, direcao, wr * 100,
            )
            return VotoAgente(self.nome, round(wr, 4))

        except Exception as exc:
            _LOG.debug("AgenteHistorico.votar falhou: %s", exc)
            return VotoAgente(self.nome, 0.5)


# ── Cérebro Central ───────────────────────────────────────────────────────

class CerebroQuant:
    """Orquestrador de votos com pesos adaptativos e recompensa por acerto."""

    _PESOS_DEFAULT: Dict[str, float] = {
        "tendencia":     0.095,
        "volatilidade":  0.063,
        "institucional": 0.136,
        "ml":            0.630,
        "historico":     0.076,
    }
    THRESHOLD_ENTRADA = 0.52   # score mínimo ponderado para autorizar entrada
    RECOMPENSA        = 0.020  # incremento de peso por acerto
    PENALIDADE        = 0.030  # decremento de peso por erro
    SILENCIO_LIMIAR   = 0.08   # peso mínimo — abaixo disso o agente é logado como fraco
    _PESOS_FILE       = Path(__file__).parent / "data" / "brain_weights.json"

    def __init__(self, threshold_ml: float = 0.55) -> None:
        self.agentes = {
            "tendencia":     AgenteTendencia(),
            "volatilidade":  AgenteVolatilidade(),
            "institucional": AgenteInstitucional(),
            "ml":            AgenteML(threshold=threshold_ml),
            "historico":     AgenteHistorico(),
        }
        self.pesos: Dict[str, float] = self._carregar_pesos()
        self._ultimo_votos: List[Tuple[str, float]] = []

    # ── I/O de pesos ──────────────────────────────────────────────────────

    def _carregar_pesos(self) -> Dict[str, float]:
        try:
            if self._PESOS_FILE.exists():
                saved = json.loads(self._PESOS_FILE.read_text())
                if all(k in saved for k in self._PESOS_DEFAULT):
                    _LOG.info("Pesos do Cérebro restaurados: %s", saved)
                    return saved
        except Exception:
            pass
        return dict(self._PESOS_DEFAULT)

    def _salvar_pesos(self) -> None:
        try:
            self._PESOS_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._PESOS_FILE.write_text(json.dumps(self.pesos, indent=2))
        except Exception as e:
            _LOG.warning("Falha ao salvar pesos do Cérebro: %s", e)

    def _normalizar(self) -> None:
        total = sum(self.pesos.values())
        if total > 0:
            self.pesos = {k: round(v / total, 5) for k, v in self.pesos.items()}

    # ── Decisão ───────────────────────────────────────────────────────────

    def decidir(
        self,
        features_df,
        wave_sig,
        smc_res,
        trend_res,
        vol_rep,
        direcao: Optional[str],
        prob_win: float,
    ) -> DecisaoCerebro:
        """Coleta votos de todos os agentes e decide se autoriza a entrada."""
        votos: Dict[str, float] = {}

        for nome, agente in self.agentes.items():
            v = agente.votar(
                features_df, wave_sig, smc_res, trend_res, vol_rep,
                direcao=direcao, prob_win=prob_win,
            )
            votos[nome] = v.score
            if v.score < self.SILENCIO_LIMIAR:
                _LOG.debug("Agente '%s' com score muito baixo: %.3f", nome, v.score)

        score = sum(votos[k] * self.pesos[k] for k in votos)
        entrar = direcao is not None and score >= self.THRESHOLD_ENTRADA

        motivo = (
            f"score={score:.3f} ({'ENTRA' if entrar else 'PASSA'}) | " +
            " | ".join(
                f"{k}={votos[k]:.2f}(w={self.pesos[k]:.2f})"
                for k in votos
            )
        )

        self._ultimo_votos = list(votos.items())

        return DecisaoCerebro(
            entrar=entrar,
            score_final=round(score, 4),
            votos=votos,
            pesos=dict(self.pesos),
            direcao=direcao,
            motivo=motivo,
        )

    # ── Recompensa ────────────────────────────────────────────────────────

    def recompensar(self, is_win: bool) -> None:
        """Atualiza pesos após resultado. Recompensa acerto, penaliza erro."""
        if not self._ultimo_votos:
            return

        for nome, score in self._ultimo_votos:
            voto_positivo = score >= 0.50
            acertou       = voto_positivo == is_win
            delta         = self.RECOMPENSA if acertou else -self.PENALIDADE
            piso          = 0.45 if nome == "ml" else 0.05
            self.pesos[nome] = max(piso, self.pesos[nome] + delta)

        self._normalizar()
        self._salvar_pesos()

        _LOG.info(
            "🧠 Cérebro atualizado (%s) → %s",
            "WIN ✓" if is_win else "LOSS ✗",
            self.status(),
        )

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> str:
        return " | ".join(f"{k}={v:.3f}" for k, v in self.pesos.items())
