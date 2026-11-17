"""
Retreino com target real (is_win) do live_trades.csv.
Executado automaticamente quando o CSV atinge o limiar de amostras.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

BOT_DIR    = Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

from features import FEATURE_COLUMNS

CSV_PATH   = BOT_DIR / "data" / "live_trades.csv"
MODEL_PATH = BOT_DIR / "models" / "lgbm_model.joblib"
THRESHOLD  = 0.55
MIN_SAMPLES = 1000  # após filtro de pares/horas ruins; bruto precisa ~1300+
# AUC_GATE > 0.505 exige poder preditivo real acima do aleatório e do proxy v1 (0.5051)
AUC_GATE   = 0.505

# Pares com melhor win rate histórico (>= 50%) recebem flag one-hot
# EURUSD removido (WR 40.2%); USDJPY promovido (57.3% nas últimas 500)
TOP_PAIRS = ["AUDUSD-OTC", "EURJPY", "EURJPY-OTC", "USDJPY", "USDCHF-OTC", "EURUSD-OTC"]

# Pares e horas excluídos do treino — dados ruidosos degradam o sinal
_PARES_EXCLUIDOS   = {"EURUSD"}
_HORAS_BLOQUEADAS  = {2, 4, 13, 18, 21, 22, 23}

# Features contextuais calculadas de timestamp+par (não existem na FeatureEngine ao vivo)
CONTEXT_FEATURES = [
    "hora_sin", "hora_cos",   # hora do dia em codificação cíclica
    "dow_sin",  "dow_cos",    # dia da semana cíclico
] + [f"par_{p.lower().replace('-','_')}" for p in TOP_PAIRS]

ALL_TRAIN_FEATURES = FEATURE_COLUMNS + CONTEXT_FEATURES

LGBM_PARAMS = dict(
    max_depth=4,
    learning_rate=0.03,
    n_estimators=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)


def _adicionar_contexto(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features cíclicas de hora/dia e flags de par ao DataFrame."""
    dt = pd.to_datetime(df["timestamp"], unit="s")
    hora = dt.dt.hour
    dow  = dt.dt.dayofweek
    df = df.copy()
    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)
    for p in TOP_PAIRS:
        col = f"par_{p.lower().replace('-', '_')}"
        df[col] = (df["par"] == p).astype(float)
    return df


def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["direction"].isin(["CALL", "PUT"])].copy()
    df = df.dropna(subset=FEATURE_COLUMNS + ["is_win"])
    df = df[df["is_win"].isin([0, 1])].copy()

    # Remove pares com WR historicamente baixo — ensinam padrões errados
    n_antes = len(df)
    df = df[~df["par"].isin(_PARES_EXCLUIDOS)].copy()

    # Remove horas bloqueadas — entradas de baixo WR contaminam o sinal
    df = _adicionar_contexto(df)
    hora_utc = pd.to_datetime(df["timestamp"], unit="s").dt.hour
    df = df[~hora_utc.isin(_HORAS_BLOQUEADAS)].copy()

    n_depois = len(df)
    print(f"Dados filtrados: {n_antes} → {n_depois} (removidos {n_antes - n_depois} ruidosos)")
    return df.sort_values("timestamp").reset_index(drop=True)


def _scale_pos_weight(y: pd.Series) -> float:
    """Peso para balancear classes — LightGBM penaliza mais a classe minoritária."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def treinar(df: pd.DataFrame) -> dict:
    X = df[ALL_TRAIN_FEATURES]
    # target real: is_win (vitória efetiva do trade)
    y = df["is_win"].astype(int)

    spw = _scale_pos_weight(y)
    print(f"Balanço de classes: {(y==1).sum()} wins / {(y==0).sum()} losses → scale_pos_weight={spw:.3f}")

    tscv = TimeSeriesSplit(n_splits=3)
    aucs = []
    for tr, val in tscv.split(X):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_val, y_val = X.iloc[val], y.iloc[val]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue
        sc = StandardScaler()
        spw_fold = _scale_pos_weight(y_tr)
        clf = LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw_fold)
        clf.fit(sc.fit_transform(X_tr), y_tr)
        prob = clf.predict_proba(sc.transform(X_val))[:, 1]
        aucs.append(roc_auc_score(y_val, prob))

    mean_auc = float(np.mean(aucs)) if aucs else 0.0

    # Treino final nos primeiros 85%
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    dirs_val   = df["direction"].iloc[split_idx:].values
    is_win_val = df["is_win"].iloc[split_idx:].values

    sc_final  = StandardScaler()
    spw_final = _scale_pos_weight(y_train)
    clf_final = LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw_final)
    clf_final.fit(sc_final.fit_transform(X_train), y_train)

    prob_val     = clf_final.predict_proba(sc_final.transform(X_val))[:, 1]
    prob_win_val = np.where(dirs_val == "CALL", prob_val, 1.0 - prob_val)
    acc_val      = float((clf_final.predict(sc_final.transform(X_val)) == y_val).mean())

    mask_thr  = prob_win_val >= THRESHOLD
    n_thr     = int(mask_thr.sum())
    prec_thr  = float(is_win_val[mask_thr].mean()) if n_thr > 0 else 0.0

    return dict(
        model=clf_final,
        scaler=sc_final,
        mean_auc=mean_auc,
        acc_val=acc_val,
        prec_thr=prec_thr,
        n_thr=n_thr,
        n_samples=len(df),
    )


def main() -> None:
    if not CSV_PATH.exists():
        print("live_trades.csv não encontrado.")
        return

    df = carregar_dados()
    n = len(df)
    print(f"Amostras válidas: {n} | Mínimo: {MIN_SAMPLES}")

    if n < MIN_SAMPLES:
        print(f"Aguardando mais trades ({MIN_SAMPLES - n} faltando). Abortando.")
        return

    print(f"Iniciando retreino com {n} amostras reais...")
    resultado = treinar(df)

    print(f"AUC médio CV:  {resultado['mean_auc']:.4f}")
    print(f"Accuracy val:  {resultado['acc_val']*100:.1f}%")
    print(f"P(Win)@{THRESHOLD}: {resultado['prec_thr']*100:.1f}% (n={resultado['n_thr']})")

    if resultado["mean_auc"] <= AUC_GATE:
        print(f"Quality Gate FALHOU (AUC {resultado['mean_auc']:.4f} ≤ {AUC_GATE}). Modelo NÃO salvo.")
        return

    if resultado["acc_val"] < 0.55:
        print(f"Quality Gate FALHOU (Accuracy Val {resultado['acc_val']*100:.1f}% < 55%). Modelo NÃO salvo.")
        return

    # Backup do modelo anterior
    backup = MODEL_PATH.with_name("lgbm_model_proxy_backup.joblib")
    if MODEL_PATH.exists():
        import shutil
        shutil.copy(MODEL_PATH, backup)
        print(f"Backup salvo: {backup.name}")

    artefato = {
        "model":             resultado["model"],
        "scaler":            resultado["scaler"],
        "feature_names":     list(ALL_TRAIN_FEATURES),
        "base_feature_names": list(FEATURE_COLUMNS),
        "context_features":  list(CONTEXT_FEATURES),
        "top_pairs":         list(TOP_PAIRS),
        "updated_at":        int(datetime.now(timezone.utc).timestamp()),
        "val_accuracy":      resultado["acc_val"],
        "training_source":   "live_trades_real_target_v3_filtered",
        "n_samples":         resultado["n_samples"],
        "mean_auc_cv":       round(resultado["mean_auc"], 4),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, MODEL_PATH)
    print(f"Modelo salvo: {MODEL_PATH}")


if __name__ == "__main__":
    main()
