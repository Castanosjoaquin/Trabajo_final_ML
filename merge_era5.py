"""Merge de las features ERA5-Land (salida de extract_era5.py) al panel — limpio.

El CSV de GEE ya viene una fila por (depto, campaña) con las 4 columnas derivadas
(no hace falta pivotear; `year` = campania_inicio). Mergea por clave geográfica
(provincia, departamento, campania_inicio) con nombres NORMALIZADOS (sin acentos),
reporta cobertura y escribe el panel aumentado (no destructivo).

Uso:
    python merge_era5.py era5_departamentos.csv
    python merge_era5.py <csv> --panel data/processed/panel_union.parquet \\
                                --out data/processed/panel_union_era5.parquet
"""
from __future__ import annotations

import argparse
import unicodedata

import pandas as pd

from componente_a.config import ERA5_COLS


def _norm(s: pd.Series) -> pd.Series:
    def one(x: str) -> str:
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return " ".join(x.upper().split())
    return s.map(one)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge features ERA5-Land al panel")
    ap.add_argument("csv", help="CSV de GEE (extract_era5.py)")
    ap.add_argument("--panel", default="data/processed/panel_union.parquet")
    ap.add_argument("--out", default="data/processed/panel_union_era5.parquet")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    era = pd.read_csv(args.csv).rename(
        columns={"ADM1_NAME": "provincia", "ADM2_NAME": "departamento", "year": "campania_inicio"})
    era["campania_inicio"] = era["campania_inicio"].astype(int)
    era_cols = [c for c in ERA5_COLS if c in era.columns]
    era = era.dropna(subset=era_cols, how="all")

    for df in (panel, era):
        df["_prov"] = _norm(df["provincia"])
        df["_dep"] = _norm(df["departamento"])
    key = ["_prov", "_dep", "campania_inicio"]

    merged = panel.merge(era[key + era_cols], on=key, how="left").drop(columns=["_prov", "_dep"])

    has = merged[era_cols[0]].notna()
    panel_pairs = set(zip(_norm(panel["provincia"]), _norm(panel["departamento"])))
    era_pairs = set(zip(era["_prov"], era["_dep"]))
    sin_match = sorted(panel_pairs - era_pairs)

    print(f"panel: {len(panel)} filas | era5 (depto×campaña): {len(era)}")
    print(f"cobertura: {has.sum()}/{len(merged)} filas con ERA5 ({has.mean():.1%})")
    print(f"pares (prov,depto) del panel sin match: {len(sin_match)}/{len(panel_pairs)}")
    for prov, dep in sin_match[:15]:
        print(f"    {prov} / {dep}")

    merged.to_parquet(args.out, index=False)
    print(f"\nescrito: {args.out}  (+{len(era_cols)} columnas: {', '.join(era_cols)})")
    print("Para usarlo: en el YAML poné `panel_path: data/processed/panel_union_era5.parquet` "
          "y `use_era5_features: true`.")


if __name__ == "__main__":
    main()
