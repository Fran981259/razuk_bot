"""Pipeline de treinamento e validação temporal do modelo LightGBM (MLTrainer).

Executa validação cruzada TimeSeriesSplit sem vazamento de dados futuros,
normaliza features com StandardScaler, exibe métricas e persiste o artefato.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import get_config
from features import FEATURE_COLUMNS, FeatureEngine

logger = logging.getLogger("train_ml")

# Hiperparâmetros anti-overfitting (fixos conforme especificação)
_LGBM_PARAMS: Dict = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "n_estimators": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}


class MLTrainer:
    """Orquestrador do pipeline de treinamento com validação temporal rigorosa."""

    def __init__(self, config=None) -> None:
        """Inicializa o treinador com configurações do bot.

        Args:
            config: Instância de BotConfig. Se None, carrega via get_config().
        """
        try:
            self._cfg = config or get_config()
            tf_min = 5 if getattr(self._cfg.trading, "timeframe", "M1").upper() == "M5" else 1
            self._expiration_candles: int = max(1, self._cfg.trading.expiration_minutes // tf_min)
            self._model_path: Path = Path(self._cfg.machine_learning.model_path)
            self._ml_threshold: float = self._cfg.machine_learning.ml_threshold
        except Exception:
            logger.warning("Config indisponível — usando valores padrão.")
            self._expiration_candles = 1
            self._model_path = Path("models/lgbm_model.joblib")
            self._ml_threshold = 0.65

        self._engine = FeatureEngine()

    # ------------------------------------------------------------------
    # Preparação do dataset
    # ------------------------------------------------------------------

    def preparar_dataset(
        self, df: pd.DataFrame, expiration_candles: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Processa OHLCV via FeatureEngine e gera a variável alvo binária.

        Alvo: 1 (Win) se Close_futuro > Close_atual, 0 (Loss) caso contrário.

        Args:
            df: DataFrame com colunas ['open', 'high', 'low', 'close', 'volume'].
            expiration_candles: Número de velas até o fechamento da operação.

        Returns:
            Tupla (X, y) — features com prefixo 'feat_' e target binário.
        """
        df_feat = self._engine.calculate_features(df, include_raw=True)

        # Target: fechamento da vela de expiração vs fechamento atual
        alvo = (df_feat["close"].shift(-expiration_candles) > df_feat["close"]).astype(int)
        df_feat["target"] = alvo

        # Remove última(s) linha(s) sem target válido
        df_valid = df_feat.iloc[:-expiration_candles].copy()

        X = df_valid[FEATURE_COLUMNS]
        y = df_valid["target"]

        logger.info(
            "Dataset preparado: %d amostras | %d features | Win rate: %.1f%%",
            len(X),
            len(FEATURE_COLUMNS),
            y.mean() * 100,
        )
        return X, y

    # ------------------------------------------------------------------
    # Treinamento e validação
    # ------------------------------------------------------------------

    def treinar_modelo(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_path: Optional[str] = None,
        n_splits: int = 5,
    ) -> Path:
        """Treina com TimeSeriesSplit, normaliza com StandardScaler e salva o artefato.

        O arquivo salvo em disco é um dicionário com chaves:
          - 'model'         → LGBMClassifier treinado
          - 'scaler'        → StandardScaler ajustado ao dataset completo
          - 'feature_names' → Lista canônica de features

        Args:
            X: Matriz de features calculadas.
            y: Vetor target (0 ou 1).
            model_path: Destino do artefato. Se None, usa MODEL_PATH da config.
            n_splits: Número de folds temporais para validação cruzada.

        Returns:
            Path do artefato serializado.
        """
        destino = Path(model_path or self._model_path)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        auc_scores: List[float] = []
        precisoes_threshold: List[float] = []
        matrizes = []

        print("\n" + "=" * 65)
        print("VALIDAÇÃO CRUZADA TEMPORAL — TimeSeriesSplit")
        print("=" * 65)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Normalização sem vazamento: scaler ajustado apenas no treino
            scaler_fold = StandardScaler()
            X_tr_sc = scaler_fold.fit_transform(X_tr)
            X_val_sc = scaler_fold.transform(X_val)

            clf = LGBMClassifier(**_LGBM_PARAMS)
            clf.fit(X_tr_sc, y_tr)

            prob_val = clf.predict_proba(X_val_sc)[:, 1]
            auc = roc_auc_score(y_val, prob_val)
            cm = confusion_matrix(y_val, (prob_val >= 0.5).astype(int))

            # Precisão apenas nas previsões acima do limiar de alta confiança
            mascara_conf = prob_val >= self._ml_threshold
            prec_threshold = (
                float(np.mean(y_val.to_numpy()[mascara_conf] == 1))
                if mascara_conf.any()
                else 0.0
            )

            auc_scores.append(auc)
            precisoes_threshold.append(prec_threshold)
            matrizes.append(cm)

            print(
                f"  Fold {fold}/{n_splits} → ROC-AUC: {auc:.4f} | "
                f"Precisão (P ≥ {self._ml_threshold:.2f}): {prec_threshold:.2%} | "
                f"Amostras: {len(y_val)}"
            )

        # Treinamento final sobre o dataset completo
        scaler_final = StandardScaler()
        X_sc = scaler_final.fit_transform(X)
        modelo_final = LGBMClassifier(**_LGBM_PARAMS)
        modelo_final.fit(X_sc, y)

        print("\n--- RESULTADOS CONSOLIDADOS ---")
        print(f"  Média ROC-AUC                  : {np.mean(auc_scores):.4f}")
        print(f"  Média Precisão (P ≥ {self._ml_threshold:.2f})    : {np.mean(precisoes_threshold):.2%}")
        print(f"  Última Matriz de Confusão (≥0.5):\n{matrizes[-1]}")

        # Persiste artefato como dicionário {model, scaler, feature_names}
        artefato = {
            "model": modelo_final,
            "scaler": scaler_final,
            "feature_names": list(FEATURE_COLUMNS),
        }
        destino.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artefato, destino)
        print(f"\n✅ Artefato salvo em: {destino.resolve()}")
        logger.info("Modelo serializado em '%s'", destino.resolve())

        return destino

    # ------------------------------------------------------------------
    # Pipeline orquestrado completo
    # ------------------------------------------------------------------

    def run_pipeline(self, caminho_csv: Optional[Union[str, Path]] = None) -> Path:
        """Executa o pipeline completo: carga → features → treino → salvamento."""
        df = self._carregar_ohlcv(caminho_csv)
        X, y = self.preparar_dataset(df, self._expiration_candles)
        return self.treinar_modelo(X, y)

    @staticmethod
    def _carregar_ohlcv(caminho: Optional[Union[str, Path]]) -> pd.DataFrame:
        """Carrega CSV OHLCV ou gera dataset sintético de 2 000 velas."""
        if caminho and Path(caminho).exists():
            df = pd.read_csv(caminho)
            df.columns = [c.strip().lower() for c in df.columns]
            logger.info("CSV carregado: %s (%d linhas)", caminho, len(df))
            return df

        candidatos = sorted(list(Path("data").glob("candles_*_M5_*.csv"))) or sorted(list(Path("data").glob("candles_*.csv")))
        if candidatos:
            dfs = []
            for f in candidatos:
                d = pd.read_csv(f)
                d.columns = [c.strip().lower() for c in d.columns]
                dfs.append(d)
            consolidado = pd.concat(dfs, ignore_index=True)
            logger.info("Consolidado %d CSVs reais (%d linhas)", len(dfs), len(consolidado))
            return consolidado

        logger.warning("Nenhum CSV fornecido — gerando dataset sintético de 2000 velas.")
        rng = np.random.default_rng(seed=42)
        n = 2_000
        retornos = rng.normal(0.00002, 0.0006, n)
        close = 1.1000 * np.exp(np.cumsum(retornos))
        spread = rng.uniform(0.0001, 0.0006, n)
        return pd.DataFrame(
            {
                "open": close + rng.uniform(-0.0003, 0.0003, n),
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": rng.integers(500, 15_000, n).astype(float),
            }
        )


# ----------------------------------------------------------------------
# Teste sintético de ponta a ponta
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 65)
    print("MLTrainer — Pipeline de Treinamento Sintético")
    print("=" * 65)

    trainer = MLTrainer()
    caminho_artefato = trainer.run_pipeline()

    # Verifica integridade do artefato salvo
    artefato = joblib.load(caminho_artefato)
    assert "model" in artefato, "FALHA: chave 'model' ausente"
    assert "scaler" in artefato, "FALHA: chave 'scaler' ausente"
    assert "feature_names" in artefato, "FALHA: chave 'feature_names' ausente"
    assert len(artefato["feature_names"]) == len(FEATURE_COLUMNS), "FALHA: feature_names incorreto"

    print(f"\n  Chaves do artefato : {list(artefato.keys())}")
    print(f"  Features no modelo : {len(artefato['feature_names'])}")
    print("\n" + "=" * 65)
    print("  ✅ Pipeline concluído — artefato validado com sucesso.")
    print("=" * 65)
