"""Merge del NDVI-AVHRR (salida de extract_avhrr_ndvi.py) al panel — limpio.

Toma el CSV largo de GEE (ADM1_NAME, ADM2_NAME, year, month, mean), lo mapea a
campaña, lo pivotea a columnas mensuales `ndvi_avhrr_<mes>` y lo mergea al panel
por clave geográfica (provincia, departamento, campania_inicio) con nombres
NORMALIZADOS (mayúsculas, sin acentos), porque GAUL y el panel pueden escribir
distinto. Reporta la cobertura y los departamentos que NO matchean — para que el
dato quede controlado, no a ciegas.

Salida: data/processed/panel_union_ndvi.parquet (no toca el panel original).

Uso:
    python merge_avhrr_ndvi.py ndvi_avhrr_departamentos.csv
    python merge_avhrr_ndvi.py <csv> --panel data/processed/panel_union.parquet \\
                                      --out data/processed/panel_union_ndvi.parquet
"""
from __future__ import annotations

import argparse
import unicodedata

import pandas as pd

# mes calendario -> nombre de mes del panel
_MONTH_NAME = {9: "sep", 10: "oct", 11: "nov", 12: "dic", 1: "ene", 2: "feb", 3: "mar"}


def _norm(s: pd.Series) -> pd.Series:
    """Normaliza nombres para el join: sin acentos, mayúsculas, espacios colapsados."""
    def one(x: str) -> str:
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return " ".join(x.upper().split())
    return s.map(one)


def load_ndvi_wide(csv_path: str) -> pd.DataFrame:
    """CSV largo de GEE → tabla ancha por (provincia, departamento, campania_inicio)
    con columnas ndvi_avhrr_<mes>."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"ADM1_NAME": "provincia", "ADM2_NAME": "departamento",
                            "mean": "ndvi"})
    df = df.dropna(subset=["ndvi"])
    # Mes calendario → campaña: Sep–Dic pertenecen al año Y; Ene–Mar al año Y-1.
    df["campania_inicio"] = df.apply(
        lambda r: int(r["year"]) if int(r["month"]) >= 9 else int(r["year"]) - 1, axis=1)
    df["mes"] = df["month"].map(_MONTH_NAME)
    df = df.dropna(subset=["mes"])

    wide = df.pivot_table(
        index=["provincia", "departamento", "campania_inicio"],
        columns="mes", values="ndvi", aggfunc="mean").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={m: f"ndvi_avhrr_{m}" for m in _MONTH_NAME.values()})
    return wide


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge NDVI-AVHRR al panel")
    ap.add_argument("csv", help="CSV de GEE (extract_avhrr_ndvi.py)")
    ap.add_argument("--panel", default="data/processed/panel_union.parquet")
    ap.add_argument("--out", default="data/processed/panel_union_ndvi.parquet")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    ndvi = load_ndvi_wide(args.csv)
    ndvi_cols = [c for c in ndvi.columns if c.startswith("ndvi_avhrr_")]

    # --- Claves normalizadas en ambos lados ---
    for df in (panel, ndvi):
        df["_prov"] = _norm(df["provincia"])
        df["_dep"] = _norm(df["departamento"])
    key = ["_prov", "_dep", "campania_inicio"]

    merged = panel.merge(ndvi[key + ndvi_cols], on=key, how="left")
    merged = merged.drop(columns=["_prov", "_dep"])

    # --- Reporte de cobertura ---
    has = merged[ndvi_cols[0]].notna() if ndvi_cols else pd.Series(False, index=merged.index)
    cov = has.mean()
    panel_pairs = set(zip(_norm(panel["provincia"]), _norm(panel["departamento"])))
    ndvi_pairs = set(zip(ndvi["_prov"], ndvi["_dep"]))
    sin_match = sorted(panel_pairs - ndvi_pairs)

    print(f"panel: {len(panel)} filas | ndvi (depto×campaña): {len(ndvi)}")
    print(f"cobertura: {has.sum()}/{len(merged)} filas con NDVI ({cov:.1%})")
    print(f"pares (prov,depto) del panel sin match en GAUL/NDVI: "
          f"{len(sin_match)}/{len(panel_pairs)}")
    if sin_match:
        print("  ejemplos sin match (revisar nombres):")
        for prov, dep in sin_match[:15]:
            print(f"    {prov} / {dep}")

    merged.to_parquet(args.out, index=False)
    print(f"\nescrito: {args.out}  (+{len(ndvi_cols)} columnas ndvi_avhrr_<mes>)")
    print("Para usarlo: en el YAML poné `panel_path: data/processed/panel_union_ndvi.parquet` "
          "y `use_ndvi: true` (sin train_start → historia completa 1981+).")


if __name__ == "__main__":
    main()
