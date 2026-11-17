"""Script de migração: live_trades.csv schema v1 → v2.

Remove colunas legadas (feat_rvol, feat_tmz_*) e adiciona colunas novas
(feat_rsi_14, feat_macd_hist, feat_roc_5, feat_cci_20) com 0.0 para linhas
existentes. Reordena para a ordem canônica de FEATURE_COLUMNS.

Uso:
    python scripts/migrate_live_trades.py
    python scripts/migrate_live_trades.py --dry-run
"""

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sys

# Ordem canônica — deve espelhar features.FEATURE_COLUMNS
FEATURE_COLUMNS_V2 = [
    "feat_body_ratio",
    "feat_upper_shadow",
    "feat_lower_shadow",
    "feat_engulfing",
    "feat_inside_bar",
    "feat_ema_alignment",
    "feat_ema21_slope",
    "feat_adx",
    "feat_afastamento_pct",
    "feat_afastamento_cross",
    "feat_rsi_14",
    "feat_macd_hist",
    "feat_roc_5",
    "feat_cci_20",
    "feat_atr_20",
    "feat_bollinger_pct_b",
]

LEGACY_COLUMNS = {"feat_rvol", "feat_tmz_v3_vol", "feat_tmz_v2_vol", "feat_tmz_v1_vol"}

META_COLUMNS = ["target", "direction", "is_win", "par", "timestamp"]
FINAL_COLUMNS = FEATURE_COLUMNS_V2 + META_COLUMNS


def migrate(csv_path: Path, dry_run: bool = False) -> None:
    if not csv_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    linhas_originais = len(df)
    cols_originais = list(df.columns)

    print(f"[INFO] Linhas: {linhas_originais}")
    print(f"[INFO] Colunas originais ({len(cols_originais)}): {cols_originais}")

    # Verificar se já está migrado
    colunas_legadas_presentes = LEGACY_COLUMNS & set(df.columns)
    novas_faltando = [c for c in FEATURE_COLUMNS_V2 if c not in df.columns]

    if not colunas_legadas_presentes and not novas_faltando:
        print("[OK] CSV já está no schema v2. Nenhuma migração necessária.")
        return

    print(f"\n[MIGRAÇÃO] Colunas legadas a remover: {sorted(colunas_legadas_presentes)}")
    print(f"[MIGRAÇÃO] Novas colunas a adicionar (valor=0.0): {novas_faltando}")

    if dry_run:
        print("\n[DRY-RUN] Nenhum arquivo foi modificado.")
        return

    # Backup automático
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_name(f"live_trades_backup_{ts}.csv")
    shutil.copy2(csv_path, backup_path)
    print(f"\n[BACKUP] Salvo em: {backup_path}")

    # Remover colunas legadas
    df = df.drop(columns=list(colunas_legadas_presentes), errors="ignore")

    # Adicionar colunas novas com 0.0
    for col in novas_faltando:
        df[col] = 0.0

    # Garantir que todas as colunas meta existam
    for col in META_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Reordenar para schema canônico
    df = df[FINAL_COLUMNS]

    df.to_csv(csv_path, index=False)
    print(f"[OK] Migração concluída: {linhas_originais} linhas escritas.")
    print(f"[OK] Colunas finais ({len(FINAL_COLUMNS)}): {FINAL_COLUMNS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migra live_trades.csv para schema v2.")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o diff, sem modificar.")
    parser.add_argument("--path", default="data/live_trades.csv", help="Caminho do CSV (padrão: data/live_trades.csv).")
    args = parser.parse_args()

    migrate(Path(args.path), dry_run=args.dry_run)
