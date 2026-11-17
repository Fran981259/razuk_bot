"""Backtest comparativo de timeframes: M5 vs M15 vs M30 vs H1.

Para cada par e cada TF:
- Reamostra velas M5 → TF alvo
- Calcula features (16 canônicas)
- Target binário: próxima vela fecha acima/abaixo do close atual
- TimeSeriesSplit 5-fold → AUC médio
- Walk-forward 70/30 → WR simulado, n_trades, lucro/prejuízo

Conclui com ranking dos TFs por AUC médio e WR médio.
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

from features import FEATURE_COLUMNS, FeatureEngine

# ── Config ────────────────────────────────────────────────────────────────────
PARES = [
    "AUDUSD", "AUDUSD_OTC",
    "EURJPY", "EURJPY_OTC",
    "EURUSD_OTC", "USDCHF_OTC",
]
TIMEFRAMES = {
    "M5":  300,
    "M15": 900,
    "M30": 1800,
    "H1":  3600,
}
THRESHOLD  = 0.55
PAYOUT     = 0.85
STAKE      = 10.0
N_SPLITS   = 5
MIN_ROWS   = 50   # mínimo de velas no TF alvo para treinar

LGBM_PARAMS = dict(
    max_depth=4, learning_rate=0.03, n_estimators=200,
    subsample=0.8, colsample_bytree=0.7,
    min_child_samples=15, random_state=42, verbose=-1,
)

fe = FeatureEngine()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _carregar_m5(par: str) -> pd.DataFrame | None:
    path = BOT_DIR / "data" / f"candles_{par}_M5_90d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for old, new in [("min", "low"), ("max", "high"), ("Min", "low"), ("Max", "high")]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(df.columns):
        return None
    if "volume" not in df.columns:
        df["volume"] = 1.0
    return df.reset_index(drop=True)


def _resample(df: pd.DataFrame, tf_secs: int) -> pd.DataFrame:
    """Reamostra DataFrame M5 para o TF alvo."""
    d = df.copy()
    if "from" in d.columns:
        d["datetime"] = pd.to_datetime(d["from"], unit="s")
    elif "datetime" not in d.columns:
        d["datetime"] = pd.date_range("2024-01-01", periods=len(d), freq="5min")
    else:
        d["datetime"] = pd.to_datetime(d["datetime"])

    d = d.sort_values("datetime").set_index("datetime")
    rule = f"{tf_secs}s"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in d.columns}
    agg["_cnt"] = ("open", "count") if "open" in d.columns else None

    # Agregação com contagem de integridade
    g = d.resample(rule, closed="left", label="left")
    out = g.agg({k: v for k, v in agg.items() if k != "_cnt"})
    out["_cnt"] = g["open"].count()

    n_per_bar = tf_secs // 300
    if n_per_bar > 1:
        out = out[out["_cnt"] == n_per_bar]
    return out.drop(columns=["_cnt"]).dropna().reset_index()


def _build_dataset(df: pd.DataFrame, tf_secs: int):
    """Reamostra, extrai features e target binário."""
    if tf_secs > 300:
        df_tf = _resample(df, tf_secs)
    else:
        df_tf = df.copy()

    if len(df_tf) < MIN_ROWS:
        return None, None

    feats = fe.calculate_features(df_tf)
    if feats.empty or len(feats) < MIN_ROWS:
        return None, None

    c = df_tf["close"].reindex(feats.index)
    target = (c.shift(-1) > c).astype(int)
    mask = target.notna()
    X = feats.loc[mask]
    y = target.loc[mask].astype(int)
    return X, y


def _cv_metrics(X: pd.DataFrame, y: pd.Series):
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
        if mask_thr.sum() > 0:
            precs.append(float(y_val.values[mask_thr].mean()))
            ns.append(int(mask_thr.sum()))
    return (
        float(np.mean(aucs)) if aucs else 0.0,
        float(np.mean(precs)) if precs else 0.0,
        int(np.mean(ns)) if ns else 0,
    )


def _walk_forward(X: pd.DataFrame, y: pd.Series):
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
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0
    wr = float(y_te.values[mask].mean())
    lucro = (wr * STAKE * PAYOUT - (1 - wr) * STAKE) * n
    return round(wr * 100, 1), round(lucro, 1), n


# ── Execução ──────────────────────────────────────────────────────────────────

def main():
    BREAKEVEN = 1 / (1 + PAYOUT) * 100
    print("=" * 80)
    print(f"BACKTEST COMPARATIVO DE TIMEFRAMES  |  threshold={THRESHOLD}  payout={PAYOUT*100:.0f}%  stake=R${STAKE}")
    print(f"Breakeven WR: {BREAKEVEN:.1f}%   |   Pares: {len(PARES)}   |   TFs: {list(TIMEFRAMES)}")
    print("=" * 80)

    resultados = []

    for par in PARES:
        df_m5 = _carregar_m5(par)
        if df_m5 is None:
            print(f"  [SKIP] {par} — arquivo não encontrado")
            continue
        print(f"\n▶ {par} ({len(df_m5)} velas M5)")

        row = {"par": par}
        for tf_name, tf_secs in TIMEFRAMES.items():
            X, y = _build_dataset(df_m5, tf_secs)
            if X is None:
                print(f"  {tf_name:5s} | dados insuficientes após resample")
                row.update({f"{tf_name}_auc": None, f"{tf_name}_prec": None,
                             f"{tf_name}_wr": None, f"{tf_name}_n": 0, f"{tf_name}_lucro": None})
                continue
            try:
                auc, prec, n_cv = _cv_metrics(X, y)
                wr_wf, lucro_wf, n_wf = _walk_forward(X, y)
                print(
                    f"  {tf_name:5s} | velas={len(X):4d} | AUC={auc:.4f} | "
                    f"Prec@{THRESHOLD}={prec*100:.1f}% (n≈{n_cv}) | "
                    f"WF WR={wr_wf:.1f}% n={n_wf:3d} lucro={lucro_wf:+8.1f}"
                )
                row.update({
                    f"{tf_name}_auc":   round(auc, 4),
                    f"{tf_name}_prec":  round(prec * 100, 1),
                    f"{tf_name}_wr":    wr_wf,
                    f"{tf_name}_n":     n_wf,
                    f"{tf_name}_lucro": lucro_wf,
                })
            except Exception as e:
                print(f"  {tf_name:5s} | ERRO: {e}")
                row.update({f"{tf_name}_auc": None, f"{tf_name}_prec": None,
                             f"{tf_name}_wr": None, f"{tf_name}_n": 0, f"{tf_name}_lucro": None})
        resultados.append(row)

    if not resultados:
        print("\nNenhum par processado.")
        return

    df_res = pd.DataFrame(resultados)

    # ── Resumo por TF ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RANKING POR TIMEFRAME (médias sobre todos os pares)")
    print("=" * 80)
    print(f"{'TF':6s} | {'AUC médio':>10} | {'Prec médio':>10} | {'WR WF médio':>12} | {'Lucro total':>12} | {'N trades':>9}")
    print("-" * 75)

    tf_stats = []
    for tf_name in TIMEFRAMES:
        aucs  = pd.to_numeric(df_res.get(f"{tf_name}_auc"),   errors="coerce").dropna()
        precs = pd.to_numeric(df_res.get(f"{tf_name}_prec"),  errors="coerce").dropna()
        wrs   = pd.to_numeric(df_res.get(f"{tf_name}_wr"),    errors="coerce").dropna()
        lucros = pd.to_numeric(df_res.get(f"{tf_name}_lucro"), errors="coerce").dropna()
        ns    = pd.to_numeric(df_res.get(f"{tf_name}_n"),     errors="coerce").dropna()
        if aucs.empty:
            continue
        auc_m  = aucs.mean()
        prec_m = precs.mean() if not precs.empty else 0
        wr_m   = wrs.mean()   if not wrs.empty   else 0
        luc_t  = lucros.sum() if not lucros.empty else 0
        n_t    = int(ns.sum())if not ns.empty     else 0
        print(f"{tf_name:6s} | {auc_m:>10.4f} | {prec_m:>9.1f}% | {wr_m:>11.1f}% | {luc_t:>+12.1f} | {n_t:>9d}")
        tf_stats.append({"tf": tf_name, "auc": auc_m, "wr": wr_m, "lucro": luc_t})

    # ── Tabela detalhada por par ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABELA DETALHADA: AUC por par × TF")
    print("=" * 80)
    cols_auc = [c for c in [f"{tf}_auc" for tf in TIMEFRAMES] if c in df_res.columns]
    show = df_res[["par"] + cols_auc].copy()
    show.columns = ["Par"] + list(TIMEFRAMES.keys())
    print(show.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABELA DETALHADA: WR walk-forward por par × TF")
    print("=" * 80)
    cols_wr = [c for c in [f"{tf}_wr" for tf in TIMEFRAMES] if c in df_res.columns]
    show_wr = df_res[["par"] + cols_wr].copy()
    show_wr.columns = ["Par"] + list(TIMEFRAMES.keys())
    print(show_wr.to_string(index=False))

    # ── Veredito ──────────────────────────────────────────────────────────────
    if tf_stats:
        melhor = max(tf_stats, key=lambda x: x["auc"])
        print(f"\n{'='*80}")
        print(f"MELHOR TF POR AUC: {melhor['tf']} (AUC médio {melhor['auc']:.4f})")
        acima_be = [s for s in tf_stats if s["wr"] >= BREAKEVEN]
        if acima_be:
            print(f"TFs acima do breakeven ({BREAKEVEN:.1f}%): {[s['tf'] for s in acima_be]}")
        else:
            print(f"Nenhum TF atingiu o breakeven ({BREAKEVEN:.1f}%) no walk-forward.")
        print(f"{'='*80}")

    print(f"\nBreakeven WR ≥ {BREAKEVEN:.1f}% para lucro com payout {PAYOUT*100:.0f}%")


if __name__ == "__main__":
    main()
