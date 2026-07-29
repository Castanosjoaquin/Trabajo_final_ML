"""
validate_capas_estaticas.py — Chequeos de sanidad de las capas de suelo y geografía.

Valida el producto de build_capas_estaticas.py y, si está mergeado, su integración
al panel. Está pensado para correrse después de cada extracción: los modos de falla
de GEE son silenciosos (un reduceRegions mal proyectado devuelve None sin lanzar
excepción, no un error) y solo se detectan mirando cobertura y rangos.

Chequeos:
  1. Estructura: una sola fila por (provincia, departamento).
  2. Nulos: qué deptos no tienen dato (los urbanos no tienen píxeles de suelo).
  3. Rangos plausibles por variable (límites FÍSICOS, no estadísticos: buscan
     cazar un error de unidades o una extracción vacía, no outliers legítimos).
  4. Coherencia interna: las texturas suman ~100%, wv0033 > wv1500.
  5. Cobertura: todos los departamentos del panel tienen valores, sin nulls.
  6. Constancia: cada capa es un único valor por departamento (son estáticas).

Errores vs avisos: un ERROR es algo que invalida la extracción (capa vacía,
unidades mal, deptos del panel sin dato). Un AVISO es algo esperado y ya
entendido, como los deptos urbanos sin suelo o el agua útil negativa de SoilGrids.

Uso:
    python eda/validate_capas_estaticas.py
    python eda/validate_capas_estaticas.py --estricto   # los avisos también fallan
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EST_PATH = PROC / "capas_estaticas_wide.parquet"
PANEL_PATH = PROC / "panel_union.parquet"

PROFUNDIDADES = ["0_5", "5_15", "15_30", "30_60"]

# columna -> (mínimo, máximo) físicamente admisible. No son rangos estadísticos:
# la idea es cazar un fallo de extracción (todo nulo, unidades sin convertir,
# valores en la escala cruda de SoilGrids), no señalar outliers legítimos.
RANGOS = {
    "suelo_bdod":     (0.5, 2.2),     # g/cm3 — densidad aparente
    "suelo_cec":      (0.0, 100.0),   # cmol(c)/kg
    "suelo_clay":     (0.0, 100.0),   # %
    "suelo_sand":     (0.0, 100.0),
    "suelo_silt":     (0.0, 100.0),
    # g/kg. Tope alto a propósito: en suelos hidromórficos y turbosos SoilGrids da
    # valores muy por encima de lo típico agrícola (Samborombón ~80, Ushuaia ~113),
    # y además sobreestima las propiedades orgánicas en Argentina (mismo sesgo
    # documentado para el SOC). El chequeo busca cazar un error de UNIDADES —que
    # estaría 100x afuera—, no marcar suelos orgánicos legítimos.
    "suelo_nitrogen": (0.0, 150.0),
    "suelo_phh2o":    (3.0, 10.0),    # pH
    "suelo_soc":      (0.0, 300.0),   # g/kg
    "suelo_wv0010":   (0.0, 80.0),    # % vol
    "suelo_wv0033":   (0.0, 80.0),
    "suelo_wv1500":   (0.0, 80.0),
}
RANGOS_GEO = {
    "geo_elev_mean":   (-100.0, 7000.0),   # m
    "geo_elev_std":    (0.0, 2000.0),
    "geo_slope_mean":  (0.0, 60.0),        # grados
    "geo_dist_rio_km": (0.0, 1000.0),      # km
}


def _norm(s: pd.Series) -> pd.Series:
    def one(x):
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return " ".join(x.upper().split())
    return s.map(one)


class Reporte:
    def __init__(self):
        self.errores, self.avisos, self.oks = [], [], []

    def error(self, msg): self.errores.append(msg); print(f"  [ERROR] {msg}")
    def aviso(self, msg): self.avisos.append(msg); print(f"  [AVISO] {msg}")
    def ok(self, msg):    self.oks.append(msg);    print(f"  [OK]    {msg}")


def validar_estaticas(est: pd.DataFrame, rep: Reporte) -> None:
    print("\n1. Estructura y constancia")
    dup = est.duplicated(subset=["provincia", "departamento"]).sum()
    if dup:
        rep.error(f"{dup} filas duplicadas por (provincia, departamento): "
                  f"el merge al panel las multiplicaría")
    else:
        rep.ok(f"{len(est)} filas, una por departamento")

    print("\n2. Nulos")
    cols_todas = [c for c in est.columns if c.startswith(("suelo_", "geo_"))]
    sin_dato = est[est[cols_todas].isna().any(axis=1)]
    if len(sin_dato):
        # Esperable: GAUL trae polígonos urbanos (las comunas de CABA, Vicente
        # López) donde SoilGrids no tiene píxeles de suelo. Solo es un problema
        # si alguno de esos deptos está en el panel — lo chequea el punto 4.
        rep.aviso(f"{len(sin_dato)} deptos con alguna capa nula "
                  f"(p.ej. {', '.join(sin_dato['departamento'].head(3))}) — "
                  f"típicamente urbanos, sin píxeles de suelo")
    else:
        rep.ok("ninguna capa tiene nulos")

    print("\n3. Rangos físicos")
    for base, (lo, hi) in RANGOS.items():
        cols = [f"{base}_{p}" for p in PROFUNDIDADES if f"{base}_{p}" in est.columns]
        if not cols:
            rep.error(f"{base}: no hay ninguna columna en el parquet")
            continue
        for c in cols:
            s = est[c].dropna()
            if s.empty:
                rep.error(f"{c}: 100% nulo (¿reduceRegions sin crs?)")
            elif not s.between(lo, hi).all():
                fuera = (~s.between(lo, hi)).sum()
                rep.error(f"{c}: {fuera} valores fuera de [{lo}, {hi}] "
                          f"— observado [{s.min():.2f}, {s.max():.2f}]")
    for c, (lo, hi) in RANGOS_GEO.items():
        if c not in est.columns:
            rep.error(f"{c}: falta la columna"); continue
        s = est[c].dropna()
        if s.empty:
            rep.error(f"{c}: 100% nulo")
        elif not s.between(lo, hi).all():
            rep.error(f"{c}: fuera de [{lo}, {hi}] — observado [{s.min():.2f}, {s.max():.2f}]")
    if not rep.errores:
        rep.ok("todas las variables dentro de rango físico")

    print("\n4. Coherencia interna")
    for p in PROFUNDIDADES:
        cols = [f"suelo_{t}_{p}" for t in ("clay", "sand", "silt")]
        if not all(c in est.columns for c in cols):
            continue
        # Solo filas con las tres texturas presentes: sumar con NaN da 0 y
        # marcaría como rotos a los deptos sin suelo (CABA, Vicente López).
        val = est[cols].dropna()
        suma = val.sum(axis=1)
        malas = (~suma.between(98, 102)).sum()
        if malas:
            rep.error(f"textura {p}: {malas} deptos con clay+sand+silt fuera de "
                      f"[98,102]% — observado [{suma.min():.1f}, {suma.max():.1f}]")
        else:
            rep.ok(f"textura {p}: suma ~100% en los {len(val)} deptos con dato")

    for p in PROFUNDIDADES:
        fc, pm = f"suelo_wv0033_{p}", f"suelo_wv1500_{p}"
        if fc not in est.columns or pm not in est.columns:
            continue
        d = est[fc] - est[pm]
        neg = (d <= 0).sum()
        if neg:
            # Artefacto conocido de SoilGrids: predice cada profundidad de forma
            # independiente y en suelos muy arcillosos capacidad de campo y punto
            # de marchitez se cruzan. Se corrige con clip(0) en la feature derivada.
            rep.aviso(f"agua útil {p}: {neg} deptos con wv0033 <= wv1500 "
                      f"(mín {d.min():.2f}) — artefacto de SoilGrids, se clipea a 0")
        else:
            rep.ok(f"agua útil {p}: wv0033 > wv1500 en todos los deptos")


def validar_cobertura(est: pd.DataFrame, rep: Reporte) -> None:
    print("\n5. Cobertura del panel")
    if not PANEL_PATH.exists():
        rep.aviso(f"no está {PANEL_PATH}, se saltea el cruce con el panel")
        return
    pan = pd.read_parquet(PANEL_PATH)
    est_k = set(zip(_norm(est["provincia"]), _norm(est["departamento"])))
    pan_k = set(zip(_norm(pan["provincia"]), _norm(pan["departamento"])))
    faltan = pan_k - est_k
    if faltan:
        rep.error(f"{len(faltan)} deptos del panel sin capas estáticas: "
                  f"{sorted(faltan)[:8]}")
    else:
        rep.ok(f"los {len(pan_k)} deptos del panel tienen capas estáticas")

    cols = [c for c in est.columns if c.startswith(("suelo_", "geo_"))]
    sub = est[[(p, d) in pan_k for p, d in
               zip(_norm(est["provincia"]), _norm(est["departamento"]))]]
    nulos = sub[cols].isna().sum()
    if (nulos > 0).any():
        rep.error(f"columnas con nulls en deptos del panel:\n{nulos[nulos > 0].to_string()}")
    else:
        rep.ok(f"{len(cols)} columnas sin ningún null en los deptos del panel")

    # Si el panel ya tiene las columnas mergeadas, verificar que sean constantes.
    en_panel = [c for c in cols if c in pan.columns]
    if not en_panel:
        rep.aviso("el panel todavía no tiene las capas mergeadas "
                  "(correr build_panel_union.py, paso 6c)")
        return
    print("\n6. Constancia dentro del panel")
    nun = pan.groupby(["provincia", "departamento"])[en_panel].nunique(dropna=False).max()
    variables = nun[nun > 1]
    if len(variables):
        rep.error(f"capas que NO son constantes por departamento:\n{variables.to_string()}")
    else:
        rep.ok(f"las {len(en_panel)} capas son constantes dentro de cada departamento")


def main(estricto: bool = False) -> int:
    if not EST_PATH.exists():
        print(f"[ERROR] falta {EST_PATH}. Correr: python eda/build_capas_estaticas.py")
        return 1
    est = pd.read_parquet(EST_PATH)
    print(f"Validando {EST_PATH.name}: {est.shape[0]} deptos × "
          f"{est.shape[1] - 2} capas")

    rep = Reporte()
    validar_estaticas(est, rep)
    validar_cobertura(est, rep)

    print("\n" + "=" * 62)
    print(f"RESULTADO: {len(rep.oks)} OK | {len(rep.avisos)} avisos | "
          f"{len(rep.errores)} errores")
    if rep.errores:
        return 1
    if rep.avisos and estricto:
        print("(--estricto: los avisos cuentan como fallo)")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estricto", action="store_true",
                    help="devuelve código de salida 1 también ante avisos")
    sys.exit(main(estricto=ap.parse_args().estricto))
