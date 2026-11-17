"""Backtest Comparativo: Features v3 (18) vs Features v4 Extended (23).

Avalia se as 7 novas features (RSI-7, ROC-3/10, ATR-ratio, BB-bandwidth,
volume-ratio, OBV-slope) melhoram o poder preditivo do LightGBM nos dados
históricos M5/M15 disponíveis.

Metodologia:
- Dados: candles M5 de 90 dias (7 pares)
- Target: próxima vela fecha acima/abaixo da abertura (binário)
- Validação: TimeSeriesSplit 5-fold (sem leakage temporal)
- Métricas: AUC-ROC, Precisão@threshold, WR simulado, n_trades
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

BOT_DIR = Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

from features import FEATURE_COLUMNS, FEATURE_COLUMNS_EXTENDED, FeatureEngine

# ── Configurações ─────────────────────────────────────────────────────────────
PARES = [
    "AUDUSD", "AUDUSD_OTC", "EURJPY", "EURJPY_OTC",
    "EURUSD_OTC", "USDCHF_OTC",
]
THRESHOLD  = 0.55
PAYOUT     = 0.85
STAKE      = 10.0
N_SPLITS   = 5
LGBM_PARAMS = dict(
    max_depth=4, learning_rate=0.03, n_estimators=200,
    subsample=0.8, colsample_bytree=0.7,
    min_child_samples=20, random_state=42, verbose=-1,
)

fe = FeatureEngine()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _carregar_m5(par: str) -> pd.DataFrame | None:
    path = BOT_DIR / "data" / f"candles_{par}_M5_90d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for rename in [{"min": "low", "max": "high"}, {"Min": "low", "Max": "high"}]:
        for old, new in rename.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return None
    return df.reset_index(drop=True)


def _build_dataset(df: pd.DataFrame, extended: bool):
    """Extrai features e target binário (próxima vela CALL/PUT)."""
    feats = fe.calculate_features(df, extended=extended)
    # Target: 1 se close[i+1] > open[i+1] (ganho CALL), 0 caso contrário
    # Alinha pelo índice original após dropna
    close_series = df["close"].reindex(feats.index)
    target = (close_series.shift(-1) > close_series).astype(int)

    mask = target.notna()
    X = feats.loc[mask]
    y = target.loc[mask].astype(int)
    return X, y


def _cv_metrics(X: pd.DataFrame, y: pd.Series):
    """TimeSeriesSplit CV → AUC médio + precisão@threshold."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs, precs, ns = [], [], []

    for tr_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue

        sc = StandardScaler()
        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(sc.fit_transform(X_tr), y_tr)
        prob = clf.predict_proba(sc.transform(X_val))[:, 1]

        aucs.append(roc_auc_score(y_val, prob))

        mask_thr = prob >= THRESHOLD
        n_thr = mask_thr.sum()
        if n_thr > 0:
            precs.append(float(y_val.values[mask_thr].mean()))
            ns.append(n_thr)

    return (
        float(np.mean(aucs)) if aucs else 0.0,
        float(np.mean(precs)) if precs else 0.0,
        int(np.mean(ns)) if ns else 0,
    )


def _walk_forward_wr(X: pd.DataFrame, y: pd.Series):
    """Walk-forward 70/30 — WR simulado e lucro/prejuízo em STAKE fixo."""
    split = int(len(X) * 0.70)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    if len(np.unique(y_tr)) < 2:
        return 0.0, 0.0, 0

    sc = StandardScaler()
    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(sc.fit_transform(X_tr), y_tr)
    prob = clf.predict_proba(sc.transform(X_te))[:, 1]

    mask = prob >= THRESHOLD
    n = mask.sum()
    if n == 0:
        return 0.0, 0.0, 0

    wr = float(y_te.values[mask].mean())
    lucro = wr * STAKE * PAYOUT - (1 - wr) * STAKE
    return wr * 100, lucro * n, int(n)


# ── Execução ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("BACKTEST COMPARATIVO: Features v3 (16) vs v4 Extended (23)")
    print(f"Threshold entrada: {THRESHOLD} | Payout: {PAYOUT*100:.0f}% | Stake: R${STAKE}")
    print(f"Breakeven WR: {1/(1+PAYOUT)*100:.1f}%")
    print("=" * 72)

    resultados = []

    for par in PARES:
        df = _carregar_m5(par)
        if df is None:
            print(f"  [SKIP] {par} — arquivo não encontrado")
            continue

        print(f"\n▶ {par} ({len(df)} velas M5)")

        row = {"par": par, "n_velas": len(df)}

        for label, extended in [("v3 (16 feat)", False), ("v4 (23 feat)", True)]:
            try:
                X, y = _build_dataset(df, extended)
                auc, prec, n_cv = _cv_metrics(X, y)
                wr_wf, lucro_wf, n_wf = _walk_forward_wr(X, y)

                print(
                    f"  {label:15s} | AUC={auc:.4f} | "
                    f"Prec@{THRESHOLD}={prec*100:.1f}% (n_cv≈{n_cv}) | "
                    f"WF WR={wr_wf:.1f}% n={n_wf} lucro={lucro_wf:+.1f}"
                )

                key = "v3" if "v3" in label else "v4"
                row[f"{key}_auc"]   = round(auc, 4)
                row[f"{key}_prec"]  = round(prec * 100, 1)
                row[f"{key}_wr_wf"] = round(wr_wf, 1)
                row[f"{key}_n_wf"]  = n_wf
                row[f"{key}_lucro"] = round(lucro_wf, 1)
            except Exception as e:
                print(f"  {label:15s} | ERRO: {e}")

        resultados.append(row)

    # ── Resumo global ─────────────────────────────────────────────────────────
    if not resultados:
        print("\nNenhum par processado.")
        return

    print("\n" + "=" * 72)
    print("RESUMO GLOBAL")
    print("=" * 72)

    df_res = pd.DataFrame(resultados)

    for col in ["v3_auc", "v4_auc", "v3_prec", "v4_prec", "v3_wr_wf", "v4_wr_wf"]:
        if col in df_res.columns:
            df_res[col] = pd.to_numeric(df_res[col], errors="coerce")

    cols_show = [c for c in
        ["par", "v3_auc", "v4_auc", "v3_prec", "v4_prec",
         "v3_wr_wf", "v4_wr_wf", "v3_n_wf", "v4_n_wf"]
        if c in df_res.columns]

    print(df_res[cols_show].to_string(index=False))

    if "v3_auc" in df_res and "v4_auc" in df_res:
        delta_auc  = (df_res["v4_auc"]   - df_res["v3_auc"]).mean()
        delta_prec = (df_res["v4_prec"]  - df_res["v3_prec"]).mean()
        delta_wr   = (df_res["v4_wr_wf"] - df_res["v3_wr_wf"]).mean()
        print(f"\n── Ganho médio v4 vs v3 ──")
        print(f"  ΔAUCmédio : {delta_auc:+.4f}")
        print(f"  ΔPrec@{THRESHOLD}: {delta_prec:+.1f}%")
        print(f"  ΔWR WF   : {delta_wr:+.1f}%")

        if delta_auc > 0.005:
            print("\n✅ v4 melhora o sinal — recomenda-se atualizar FEATURE_COLUMNS.")
        elif delta_auc > 0:
            print("\n⚠️  Melhora marginal. Mais dados recomendados antes de promover v4.")
        else:
            print("\n❌ v4 não melhora vs v3. Manter feature set atual.")

    print("\nBreakeven: WR ≥ 54.1% para lucro com payout 85%")


if __name__ == "__main__":
    main()
