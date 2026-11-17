"""Módulo de Inferência em Tempo Real (MLCore).

Carrega o artefato {model, scaler, feature_names} gerado pelo MLTrainer e
estima a probabilidade de vitória P(Win) a partir do vetor de features
calculado pela FeatureEngine. Opera em modo fallback quando não há modelo.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

from config import MachineLearningConfig
from features import FEATURE_COLUMNS


class MLCore:
    """Núcleo de inferência estatística preditiva em tempo real.

    Suporta direções CALL e PUT. Para PUT, retorna 1.0 - P(CALL) como
    proxy da probabilidade de queda, conforme a assimetria do modelo.

    Suporta modelos dedicados por par: se existir
    ``models/lgbm_{par_normalizado}_calibrated.joblib`` (ex: lgbm_audusd_otc_calibrated),
    ele é carregado na primeira chamada para aquele par e reutilizado em cache.
    Caso contrário, usa o modelo global.
    """

    def __init__(
        self,
        config: Optional[MachineLearningConfig] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Inicializa e carrega o modelo serializado de forma resiliente.

        Args:
            config: Configurações de Machine Learning. Se None, usa padrão.
            model_path: Caminho alternativo explícito para o artefato .joblib.
        """
        self._logger = logging.getLogger("ml_core")
        cfg = config or MachineLearningConfig()
        self._model_path = Path(model_path or cfg.model_path)
        self._threshold: float = cfg.ml_threshold

        # Componentes do artefato global (carregados via _carregar_modelo)
        self._model = None
        self._scaler = None
        self._feature_names: list = list(FEATURE_COLUMNS)

        # Cache de modelos dedicados por par: {par: (model, scaler, feature_names) | None}
        # None = já tentou carregar e não encontrou (evita I/O repetido)
        self._pair_cache: dict = {}

        self._carregar_modelo()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True se o modelo e o scaler estão disponíveis para inferência."""
        return self._model is not None and self._scaler is not None

    @property
    def threshold(self) -> float:
        """Limiar mínimo de probabilidade configurado para autorizar entradas."""
        return self._threshold

    # ------------------------------------------------------------------
    # Carregamento resiliente
    # ------------------------------------------------------------------

    def _carregar_modelo(self) -> bool:
        """Carrega o artefato .joblib do disco de forma resiliente.

        Em caso de arquivo ausente ou corrompido, define modo fallback
        (is_ready=False) sem lançar exceção e sem travar o bot.

        Returns:
            True se o modelo foi carregado com sucesso, False caso contrário.
        """
        if not self._model_path.exists():
            self._logger.warning(
                "Artefato '%s' não encontrado. Execute train_ml.py para treinar o modelo. "
                "MLCore operando em modo FALLBACK (sem predições).",
                self._model_path,
            )
            return False

        try:
            artefato = joblib.load(self._model_path)

            # Suporte a artefato dicionário {model, scaler, feature_names}
            if isinstance(artefato, dict):
                self._model = artefato["model"]
                self._scaler = artefato["scaler"]
                self._feature_names = artefato.get("feature_names", list(FEATURE_COLUMNS))
            else:
                # Compatibilidade retroativa com artefato simples (só o modelo)
                self._logger.warning(
                    "Artefato sem scaler detectado — usando StandardScaler identidade."
                )
                self._model = artefato
                self._scaler = None

            self._logger.info(
                "Modelo carregado com sucesso: '%s' | Features: %d | Threshold: %.2f",
                self._model_path,
                len(self._feature_names),
                self._threshold,
            )
            return True

        except Exception as exc:
            self._logger.error(
                "Erro ao carregar artefato '%s': %s. Modo FALLBACK ativo.",
                self._model_path,
                exc,
            )
            self._model = None
            self._scaler = None
            return False

    def reload_model(self) -> bool:
        """Recarrega o artefato do disco (útil após re-treino sem reiniciar o bot).

        Returns:
            True se o recarregamento foi bem-sucedido.
        """
        self._pair_cache.clear()
        return self._carregar_modelo()

    # ------------------------------------------------------------------
    # Modelos por par
    # ------------------------------------------------------------------

    def _carregar_par(self, par: str) -> Optional[tuple]:
        """Tenta carregar modelo dedicado para `par`. Retorna (model, scaler, feature_names) ou None."""
        if par in self._pair_cache:
            return self._pair_cache[par]

        nome_arquivo = "lgbm_" + par.lower().replace("-", "_") + "_calibrated.joblib"
        caminho = self._model_path.parent / nome_arquivo
        if not caminho.exists():
            self._pair_cache[par] = None
            return None

        try:
            art = joblib.load(caminho)
            if isinstance(art, dict):
                entry = (art["model"], art["scaler"], art.get("feature_names", list(FEATURE_COLUMNS)))
            else:
                entry = (art, None, list(FEATURE_COLUMNS))
            self._pair_cache[par] = entry
            self._logger.info("Modelo dedicado carregado para %s: %s", par, caminho.name)
            return entry
        except Exception as exc:
            self._logger.warning("Falha ao carregar modelo para %s: %s", par, exc)
            self._pair_cache[par] = None
            return None

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_row(par: Optional[str], timestamp: Optional[float], top_pairs: list) -> dict:
        """Constrói as features de contexto (hora/dow + one-hot de par) para inferência."""
        import math
        ts = timestamp if timestamp is not None else __import__("time").time()
        dt = __import__("datetime").datetime.utcfromtimestamp(ts)
        hora = dt.hour
        dow  = dt.weekday()
        ctx: dict = {
            "hora_sin": math.sin(2 * math.pi * hora / 24),
            "hora_cos": math.cos(2 * math.pi * hora / 24),
            "dow_sin":  math.sin(2 * math.pi * dow / 7),
            "dow_cos":  math.cos(2 * math.pi * dow / 7),
        }
        for p in top_pairs:
            col = f"par_{p.lower().replace('-', '_')}"
            ctx[col] = 1.0 if par == p else 0.0
        return ctx

    def predict_win_probability(
        self,
        features_df: pd.DataFrame,
        direction: str = "CALL",
        par: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """Estima a probabilidade de vitória P(Win) para a vela atual.

        Para CALL: retorna P(Win) diretamente.
        Para PUT : retorna 1.0 - P(CALL) como proxy de queda.

        Em modo FALLBACK (modelo não carregado), retorna 0.0 sem travar.

        Args:
            features_df: DataFrame com as variáveis 'feat_*' da FeatureEngine.
            direction: "CALL" (alta) ou "PUT" (baixa).
            par: Par operado (ex: "EURUSD-OTC") — usado para contexto e modelo dedicado.
            timestamp: Unix timestamp da vela — usado para features de hora/dia.

        Returns:
            Probabilidade entre 0.0 e 1.0.
        """
        if not self.is_ready:
            self._logger.warning(
                "MLCore em FALLBACK — modelo não disponível. Retornando P(Win)=0.0."
            )
            return 0.0

        if features_df.empty:
            self._logger.warning("DataFrame de features vazio. Retornando P(Win)=0.0.")
            return 0.0

        # Seleciona modelo dedicado por par (se existir) ou global
        par_entry = self._carregar_par(par) if par else None
        if par_entry is not None:
            modelo, scaler, feature_names = par_entry
        else:
            modelo, scaler, feature_names = self._model, self._scaler, self._feature_names

        # Usa apenas a última linha (vela mais recente)
        linha = features_df.iloc[[-1]].copy()

        # Injeta features de contexto se o modelo foi treinado com elas
        context_cols = [c for c in feature_names if c not in FEATURE_COLUMNS]
        if context_cols:
            top_pairs = []
            # Extrai top_pairs do artefato se disponível
            try:
                art = joblib.load(self._model_path)
                top_pairs = art.get("top_pairs", []) if isinstance(art, dict) else []
            except Exception:
                pass
            ctx = self._build_context_row(par, timestamp, top_pairs)
            for col, val in ctx.items():
                if col in context_cols:
                    linha[col] = val
            # Garante que todas as colunas de contexto existam
            for col in context_cols:
                if col not in linha.columns:
                    linha[col] = 0.0

        faltantes = set(feature_names) - set(linha.columns)
        if faltantes:
            self._logger.error("Features ausentes: %s. Retornando P(Win)=0.0.", faltantes)
            return 0.0

        linha = linha[feature_names]

        # Aplica normalização do scaler treinado
        if scaler is not None:
            linha_sc = scaler.transform(linha)
        else:
            linha_sc = linha.to_numpy()

        # LightGBM predict_proba → [[P(Loss), P(Win)]]
        probabilidades = modelo.predict_proba(linha_sc)
        prob_call = float(probabilidades[0, 1])

        direcao = direction.upper()
        if direcao == "PUT":
            prob_win = round(1.0 - prob_call, 4)
        else:
            prob_win = round(prob_call, 4)

        self._logger.debug(
            "Inferência | Direção=%s | P(CALL)=%.4f | P(Win)=%.4f | Threshold=%.2f | Sinal=%s",
            direcao,
            prob_call,
            prob_win,
            self._threshold,
            "✅ OPERAR" if prob_win >= self._threshold else "⏭ SKIP",
        )
        return prob_win


# ----------------------------------------------------------------------
# Teste sintético — valida carregamento, fallback e inferência
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 65)
    print("MLCore — Teste de Integração Sintético")
    print("=" * 65)

    # --- [1] Modo Fallback (modelo inexistente) ---
    print("\n--- [1] Fallback: modelo inexistente ---")
    core_fallback = MLCore(model_path="models/_nao_existe.joblib")
    assert not core_fallback.is_ready, "FALHA: deveria estar em fallback"
    prob = core_fallback.predict_win_probability(pd.DataFrame())
    print(f"  is_ready={core_fallback.is_ready} | P(Win) fallback={prob:.4f}")
    assert prob == 0.0, "FALHA: fallback deve retornar 0.0"

    # --- [2] Treino sintético e carregamento ---
    print("\n--- [2] Treinamento e carregamento de artefato real ---")
    from train_ml import MLTrainer
    from sklearn.preprocessing import StandardScaler as SS
    from lightgbm import LGBMClassifier as LGBM

    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_artefato = Path(tmpdir) / "test_model.joblib"

        trainer = MLTrainer()
        trainer._model_path = caminho_artefato
        X, y = trainer.preparar_dataset(trainer._carregar_ohlcv(None))

        scaler = SS()
        X_sc = scaler.fit_transform(X)
        modelo = LGBM(n_estimators=50, verbose=-1, random_state=42)
        modelo.fit(X_sc, y)

        joblib.dump({"model": modelo, "scaler": scaler, "feature_names": list(FEATURE_COLUMNS)}, caminho_artefato)

        core = MLCore(model_path=caminho_artefato)
        assert core.is_ready, "FALHA: modelo deveria estar carregado"

        # Gera uma linha de features sintéticas para inferência
        rng = np.random.default_rng(seed=7)
        close = 1.1000 + np.cumsum(rng.normal(0, 0.0002, 250))
        df_inf = pd.DataFrame({
            "open": close - 0.0001,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "volume": rng.integers(500, 5000, 250).astype(float),
        })
        from features import FeatureEngine
        feats = FeatureEngine().calculate_features(df_inf)

        prob_call = core.predict_win_probability(feats, direction="CALL")
        prob_put = core.predict_win_probability(feats, direction="PUT")

        print(f"  is_ready={core.is_ready}")
        print(f"  P(Win) CALL = {prob_call:.4f}")
        print(f"  P(Win) PUT  = {prob_put:.4f}")
        print(f"  CALL + PUT  = {prob_call + prob_put:.4f}  (deve ser ≈ 1.0)")

        assert abs(prob_call + prob_put - 1.0) < 1e-4, "FALHA: CALL + PUT deve ser 1.0"
        assert 0.0 <= prob_call <= 1.0, "FALHA: prob fora do intervalo"

    print("\n" + "=" * 65)
    print("  ✅ TODOS OS TESTES PASSARAM — MLCore operacional.")
    print("=" * 65 + "\n")
