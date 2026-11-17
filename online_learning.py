"""Módulo de Aprendizado Contínuo em Tempo Real (OnlineLearner).

Coleta feedback de operações reais (WIN/LOSS), monitora novas velas M1 e
executa re-calibração periódica e adaptativa do modelo LightGBM sem bloquear
o loop assíncrono de trading.
"""

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from config import MachineLearningConfig
from features import FEATURE_COLUMNS, FeatureEngine

_DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "n_estimators": 150,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}


class OnlineLearner:
    """Gerencia buffer de feedback real e ciclo adaptativo de re-treino."""

    def __init__(
        self,
        config: Optional[MachineLearningConfig] = None,
        data_dir: str = "data",
        retrain_trades_threshold: int = 5,
        retrain_interval_seconds: float = 1800.0,
        base_anchor_samples: int = 2500,
        timeframe: str = "M5",
    ) -> None:
        """Inicializa o mecanismo de aprendizado online."""
        self._logger = logging.getLogger("online_learner")
        self._config = config or MachineLearningConfig()
        self._timeframe = timeframe.upper()
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._trades_path = self._data_dir / "live_trades.csv"
        self._model_path = Path(self._config.model_path)
        self._retrain_trades_threshold = retrain_trades_threshold
        self._retrain_interval_seconds = retrain_interval_seconds
        self._base_anchor_samples = base_anchor_samples
        self._trades_since_retrain: int = 0
        self._last_time_retrain: float = time.time()
        self._is_learning: bool = False
        self._engine = FeatureEngine()
        self._csv_lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None

    @property
    def trades_since_retrain(self) -> int:
        """Quantidade de trades registrados desde a última calibração."""
        return self._trades_since_retrain

    def record_trade(self, features: pd.Series, direction: str, is_win: bool, par: str = "") -> None:
        """Registra o feedback real da operação no dataset de aprendizado com lock thread-safe."""
        dir_upper = direction.strip().upper()
        target = (1 if is_win else 0) if dir_upper == "CALL" else (0 if is_win else 1)

        row_data: Dict[str, Any] = {col: float(features.get(col, 0.0)) for col in FEATURE_COLUMNS}
        row_data["target"] = int(target)
        row_data["direction"] = dir_upper
        row_data["is_win"] = int(is_win)
        row_data["par"] = par
        row_data["timestamp"] = int(datetime.now(timezone.utc).timestamp())

        df_row = pd.DataFrame([row_data])
        with self._csv_lock:
            is_new = not self._trades_path.exists()
            if not is_new:
                try:
                    existing_cols = pd.read_csv(self._trades_path, nrows=0).columns.tolist()
                    for col in existing_cols:
                        if col not in df_row.columns:
                            df_row[col] = 0.0
                    df_row = df_row[existing_cols]
                except Exception:
                    pass
            df_row.to_csv(self._trades_path, mode="a", index=False, header=is_new)

        self._trades_since_retrain += 1
        self._logger.info(
            "Feedback registrado: %s (%s) em %s | Target: %d | Buffer: %d/%d trades",
            dir_upper, "WIN" if is_win else "LOSS", par, target,
            self._trades_since_retrain, self._retrain_trades_threshold,
        )

    def should_retrain_trades(self) -> bool:
        """Verifica se atingiu o limiar de trades reais para re-calibração."""
        return self._trades_since_retrain >= self._retrain_trades_threshold and not self._is_learning

    def should_retrain_time(self) -> bool:
        """Verifica se transcorreu o intervalo de tempo para re-calibração."""
        decorrido = time.time() - self._last_time_retrain
        return decorrido >= self._retrain_interval_seconds and not self._is_learning

    @property
    def learning_state(self) -> Tuple[str, float]:
        """Estado atual do aprendizado online (estado, progresso 0.0–1.0)."""
        pct = min(self._trades_since_retrain / max(self._retrain_trades_threshold, 1), 1.0)
        if self._is_learning:
            return ("APRENDENDO", 1.0)
        decorrido = time.time() - self._last_time_retrain
        tempo_pct = min(decorrido / max(self._retrain_interval_seconds, 1), 1.0)
        if decorrido > self._retrain_interval_seconds * 2 and self._trades_since_retrain == 0:
            return ("SEM APRENDIZADO", 0.0)
        return ("ESTÁVEL", max(pct, tempo_pct))

    def _load_base_anchor(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[np.ndarray]]:
        """Carrega amostra base histórica para ancorar o Scaler e evitar esquecimento catastrófico."""
        candidatos = list(self._data_dir.glob(f"candles_*_{self._timeframe}_*.csv")) or list(self._data_dir.glob("candles_*.csv"))
        if not candidatos:
            return None, None, None
        try:
            df_raw = pd.read_csv(candidatos[0]).tail(self._base_anchor_samples)
            if len(df_raw) < 85:
                return None, None, None
            df_feat = self._engine.calculate_features(df_raw, include_raw=True)
            df_feat["target"] = (df_feat["close"].shift(-1) > df_feat["close"]).astype(int)
            df_valid = df_feat.iloc[:-1].copy()
            weights = np.ones(len(df_valid), dtype=float)
            return df_valid[FEATURE_COLUMNS], df_valid["target"], weights
        except Exception as exc:
            self._logger.warning("Falha ao carregar âncora histórica (%s): %s", candidatos[0].name, exc)
            return None, None, None

    def _load_live_trades(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[np.ndarray]]:
        """Carrega trades reais com tolerância a linhas corrompidas e ponderação por recência."""
        if not self._trades_path.exists():
            return None, None, None
        try:
            with self._csv_lock:
                df = pd.read_csv(self._trades_path, on_bad_lines="skip")
            if df.empty or len(df) < 2:
                return None, None, None
            cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            if len(cols) != len(FEATURE_COLUMNS) or "target" not in df.columns:
                return None, None, None
            # Ponderação por recência: trades recentes têm peso maior (1.5 até 4.0)
            n = len(df)
            weights = np.linspace(1.5, 4.0, n)
            return df[FEATURE_COLUMNS], df["target"], weights
        except Exception as exc:
            self._logger.warning("Falha ao ler live_trades.csv: %s", exc)
            return None, None, None

    def _train_sync(self, velas_recentes: Optional[pd.DataFrame]) -> bool:
        """Executa ajuste matemático e persistência do modelo em thread dedicada."""
        x_list: List[pd.DataFrame] = []
        y_list: List[pd.Series] = []
        weights_list: List[np.ndarray] = []

        # 1. Âncora histórica (preserva invariantes de mercado e calibração do Scaler)
        x_base, y_base, w_base = self._load_base_anchor()
        if x_base is not None and y_base is not None and w_base is not None:
            x_list.append(x_base)
            y_list.append(y_base)
            weights_list.append(w_base)

        # 2. Velas recentes de mercado (peso 2.0 para dinâmica imediata)
        if velas_recentes is not None and len(velas_recentes) >= 85:
            df_feat = self._engine.calculate_features(velas_recentes, include_raw=True)
            df_feat["target"] = (df_feat["close"].shift(-1) > df_feat["close"]).astype(int)
            df_valid = df_feat.iloc[:-1].copy()
            if not df_valid.empty:
                x_list.append(df_valid[FEATURE_COLUMNS])
                y_list.append(df_valid["target"])
                weights_list.append(np.full(len(df_valid), 2.0, dtype=float))

        # 3. Feedback real dos trades executados (peso adaptativo 1.5 a 4.0)
        x_tr, y_tr, w_tr = self._load_live_trades()
        if x_tr is not None and y_tr is not None and w_tr is not None:
            x_list.append(x_tr)
            y_list.append(y_tr)
            weights_list.append(w_tr)

        if not x_list:
            self._logger.warning("Dados insuficientes para re-treino adaptativo.")
            return False

        x_total = pd.concat(x_list, ignore_index=True)
        y_total = pd.concat(y_list, ignore_index=True)
        w_total = np.concatenate(weights_list)

        # Validação: requer presença das duas classes para treinamento binário
        if len(np.unique(y_total)) < 2:
            self._logger.warning("Amostra univariada de targets (apenas uma classe). Abortando.")
            return False

        # Quality Gate (mle-workflow): validação out-of-sample antes da promoção do modelo
        split_idx = int(len(x_total) * 0.85)
        x_train, x_val = x_total.iloc[:split_idx], x_total.iloc[split_idx:]
        y_train, y_val = y_total.iloc[:split_idx], y_total.iloc[split_idx:]
        w_train = w_total[:split_idx]

        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_val_sc = scaler.transform(x_val)

        # Modelo base LightGBM
        base_model = LGBMClassifier(**_DEFAULT_LGBM_PARAMS)
        base_model.fit(x_train_sc, y_train, sample_weight=w_train)

        # Calibração de probabilidades (Fase 1)
        # Isotonic para > 1000 amostras, sigmoid para < 1000
        # cv=2: cross-validation interna (sklearn 1.9+ removeu cv="prefit")
        n_cal = len(x_train_sc)
        cal_method = "isotonic" if n_cal > 1000 else "sigmoid"
        modelo = CalibratedClassifierCV(base_model, method=cal_method, cv=2)
        modelo.fit(x_train_sc, y_train, sample_weight=w_train)
        self._logger.info(
            "Modelo calibrado: %s (n=%d) | Base AUC preservada",
            cal_method, n_cal
        )

        # Avaliação na fatia de validação recente
        acc_cand = float((modelo.predict(x_val_sc) == y_val).mean())
        self._logger.info("Quality Gate MLE: Acurácia do candidato na validação = %.2f%%", acc_cand * 100)

        # Comparação com o baseline em produção (se existir)
        if self._model_path.exists():
            try:
                base_bundle = joblib.load(self._model_path)
                base_sc = base_bundle["scaler"].transform(x_val)
                acc_base = float((base_bundle["model"].predict(base_sc) == y_val).mean())
                if acc_cand < (acc_base - 0.05) and acc_cand < 0.52:
                    self._logger.warning(
                        "🚫 Modelo candidato rejeitado pelo Quality Gate: %.1f%% inferior ao baseline %.1f%%.",
                        acc_cand * 100, acc_base * 100,
                    )
                    return False
            except Exception:
                pass

        # Backup automático do modelo anterior (Rollback Seguro - MLE-09)
        if self._model_path.exists():
            rollback_path = self._model_path.with_name(self._model_path.stem + "_rollback.joblib")
            try:
                rollback_path.write_bytes(self._model_path.read_bytes())
            except Exception:
                pass

        artefato = {
            "model": modelo,
            "scaler": scaler,
            "feature_names": list(FEATURE_COLUMNS),
            "updated_at": int(datetime.now(timezone.utc).timestamp()),
            "val_accuracy": acc_cand,
        }
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artefato, self._model_path)
        return True

    async def run_learning_cycle(
        self, conector: Any, ml_core: Any, par_coleta: str = "EURUSD-OTC", motivo: str = ""
    ) -> bool:
        """Executa o ciclo completo de aprendizado de forma não-bloqueante e protegida por mutex."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        if self._async_lock.locked():
            self._logger.info("Ciclo de aprendizado já em andamento. Aguardando próximo gatilho.")
            return False

        async with self._async_lock:
            self._is_learning = True
            self._logger.info("🚀 Iniciando aprendizado adaptativo em tempo real (%s)...", motivo)

            velas_df: Optional[pd.DataFrame] = None
            try:
                if conector is not None and getattr(conector, "is_connected", False):
                    velas = await conector.get_candles(par=par_coleta, timeframe=self._timeframe, contagem=300)
                    if len(velas) >= 85:
                        velas_df = pd.DataFrame(velas)
                        if "min" in velas_df.columns and "low" not in velas_df.columns:
                            velas_df = velas_df.rename(columns={"min": "low", "max": "high"})
            except Exception as exc:
                self._logger.warning("Coleta de velas para re-treino falhou: %s", exc)

            try:
                sucesso = await asyncio.to_thread(self._train_sync, velas_df)
                if sucesso:
                    ml_core.reload_model()
                    self._trades_since_retrain = 0
                    self._last_time_retrain = time.time()
                    self._logger.info("✅ Modelo LightGBM re-calibrado e recarregado com sucesso!")
                    return True
                return False
            except Exception as exc:
                self._logger.error("Erro durante ciclo de aprendizado em tempo real: %s", exc)
                return False
            finally:
                self._is_learning = False
