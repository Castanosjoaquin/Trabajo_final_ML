"""
build_suelo_idecor.py — Suelo de Córdoba con datos país-específicos (IDECOR).

Alternativa de mayor calidad a las capas de SoilGrids para los departamentos de
Córdoba: cartas de suelo a escala **1:50.000** publicadas por IDECOR / Mapas
Córdoba, contra los 250 m globales de SoilGrids.

Lo que aporta y SoilGrids no puede dar:
  - `au_1m`, `au_1_5m`, `au_2m`: agua útil en **mm de lámina**, MEDIDA a partir de
    las constantes hídricas de cada unidad cartográfica, no estimada por un modelo
    global. Es la variable que el ablation (nb 12) mostró como la más útil en
    pre-siembra, que es justo donde el modelo hoy no le gana a la climatología.
  - `ip`: Índice de Productividad (paramétrico FAO adaptado a la Pampa).
  - `cu`/`clase`: capacidad de uso.

Cobertura: SOLO Córdoba (23 de los 307 deptos del panel). Es deliberado — se usa
como banco de prueba para responder si mejores datos de suelo mueven la aguja,
antes de invertir en conseguir el equivalente para Buenos Aires y Santa Fe.

Insumos (ambos en data/raw/idecor/, que está gitignored por su tamaño):
  - carta_suelo_50mil_2025.zip : WFS de IDECOR, capa `idecor:carta_suelo_50mil_2025`
        https://idecor-ws.mapascordoba.gob.ar/geoserver/idecor/wfs
  - gaul_cordoba.geojson       : polígonos departamentales FAO GAUL nivel 2, los
        mismos que usan NDVI/ERA5 y las capas estáticas (ver build_capas_estaticas).

Uso:
    python eda/build_suelo_idecor.py
    python eda/build_suelo_idecor.py --force
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.ops

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "idecor"
PROC = ROOT / "data" / "processed"

SUELO_ZIP = RAW / "carta_suelo_50mil_2025.zip"
DEPTOS_GJ = RAW / "gaul_cordoba.geojson"
OUT = PROC / "suelo_idecor_wide.parquet"

# CRS de igual área para Sudamérica. Es obligatorio: ponderar por área en
# EPSG:4326 (grados) daría pesos mal calculados, porque un grado cuadrado no mide
# lo mismo en el norte que en el sur de la provincia.
CRS_AREA = "ESRI:102033"

# Variables a promediar por departamento, ponderadas por área de intersección.
VARS = ["au_1m", "au_1_5m", "au_2m", "ip"]

# Variables donde el CERO significa "sin dato", no "cero milímetros de agua".
#
# 467 de las 1555 unidades traen au_1m = 0, pero solo ~20 son cuerpos de agua
# reales (lagunas, ríos, bañados, mallines); el resto son áreas fuera del
# relevamiento, codificadas con 0 en vez de nulo. Promediarlas como ceros hunde el
# valor del departamento: Río Seco daba 1.9 mm de agua útil, físicamente absurdo
# para suelo agrícola.
#
# Se excluyen los dos casos, y está bien que así sea: ni el área no relevada ni una
# laguna aportan a "cuánta agua retiene el suelo de este departamento".
CERO_ES_NULO = ["au_1m", "au_1_5m", "au_2m"]


# Envolvente aproximada de Argentina continental, para detectar ejes invertidos.
_LON_AR = (-74.0, -53.0)
_LAT_AR = (-56.0, -21.0)


def _arreglar_ejes(gdf: gpd.GeoDataFrame, nombre: str) -> gpd.GeoDataFrame:
    """Corrige el orden de ejes si el archivo vino en (lat, lon) en vez de (lon, lat).

    WFS 2.0.0 sigue la definición OGC de EPSG:4326, que es **(lat, lon)**, mientras
    que GeoJSON, los shapefiles habituales y GAUL usan (lon, lat). El export
    SHAPE-ZIP de GeoServer conserva ese orden, así que las cartas de IDECOR salen
    con x=latitud e y=longitud: los polígonos caen en otro punto del planeta y el
    overlay devuelve CERO intersecciones **sin lanzar ningún error**.

    Se detecta por bounds contra la envolvente de Argentina en vez de asumirlo, para
    que el script siga andando si algún día el servicio corrige el orden.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    ok = (_LON_AR[0] <= minx <= _LON_AR[1]) and (_LAT_AR[0] <= miny <= _LAT_AR[1])
    invertido = (_LAT_AR[0] <= minx <= _LAT_AR[1]) and (_LON_AR[0] <= miny <= _LON_AR[1])
    if ok:
        return gdf
    if not invertido:
        raise ValueError(
            f"{nombre}: bounds {gdf.total_bounds} no caen ni en (lon,lat) ni en "
            f"(lat,lon) para Argentina. Revisar el CRS del archivo.")
    log.warning("%s: ejes invertidos (lat,lon) -> se dan vuelta a (lon,lat)", nombre)
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.map(
        lambda g: shapely.ops.transform(lambda x, y: (y, x), g))
    return gdf


def _promedio_areal(inter: gpd.GeoDataFrame, var: str) -> pd.Series:
    """Media de `var` por departamento, ponderada por área de intersección.

    Los polígonos sin dato se excluyen del numerador Y del denominador: si el 40 %
    de un departamento son lagunas sin IP, el índice del resto no se diluye contra
    cero. Por eso el peso se recalcula por variable en vez de usar uno común.
    """
    valido = inter[var].notna()
    if var in CERO_ES_NULO:
        valido &= inter[var] > 0          # ver CERO_ES_NULO
    d = inter[valido].copy()
    if d.empty:
        return pd.Series(dtype=float)
    d["_num"] = d[var] * d["area_m2"]
    g = d.groupby("departamento")
    return g["_num"].sum() / g["area_m2"].sum()


def build_suelo_idecor(force: bool = False,
                       min_cobertura: float = 0.7) -> pd.DataFrame:
    if OUT.exists() and not force:
        log.info("IDECOR: usando caché %s", OUT)
        return pd.read_parquet(OUT)
    for p in (SUELO_ZIP, DEPTOS_GJ):
        if not p.exists():
            raise FileNotFoundError(
                f"Falta {p}. Ver el docstring de este módulo para cómo obtenerlo.")

    log.info("leyendo cartas de suelo IDECOR...")
    suelo = gpd.read_file(f"zip://{SUELO_ZIP}")
    deptos = gpd.read_file(DEPTOS_GJ)
    deptos = deptos.rename(columns={"ADM2_NAME": "departamento",
                                    "ADM1_NAME": "provincia"})
    log.info("  %d unidades cartográficas | %d departamentos", len(suelo), len(deptos))

    suelo = _arreglar_ejes(suelo, "cartas IDECOR")
    deptos = _arreglar_ejes(deptos, "deptos GAUL")

    # Saneamiento: geometrías inválidas rompen el overlay en silencio o lo abortan.
    for nombre, gdf in (("suelo", suelo), ("deptos", deptos)):
        malas = int((~gdf.geometry.is_valid).sum())
        if malas:
            log.warning("  %s: %d geometrías inválidas -> buffer(0)", nombre, malas)
            gdf["geometry"] = gdf.geometry.buffer(0)

    suelo = suelo.to_crs(CRS_AREA)
    deptos = deptos.to_crs(CRS_AREA)

    log.info("intersectando (esto tarda)...")
    inter = gpd.overlay(suelo[VARS + ["geometry"]],
                        deptos[["departamento", "provincia", "geometry"]],
                        how="intersection", keep_geom_type=True)
    inter["area_m2"] = inter.geometry.area
    log.info("  %d piezas de intersección", len(inter))

    out = pd.DataFrame({v: _promedio_areal(inter, v) for v in VARS})
    out.index.name = "departamento"
    out = out.reset_index()
    out.columns = ["departamento"] + [f"idecor_{v}" for v in VARS]
    out.insert(0, "provincia", "CORDOBA")

    # Fracción del departamento efectivamente cubierta por cada variable: sirve
    # para descartar deptos donde el promedio se apoya en una porción mínima.
    area_dep = inter.groupby("departamento")["area_m2"].sum()
    for v in VARS:
        valido = inter[v].notna()
        if v in CERO_ES_NULO:
            valido &= inter[v] > 0
        con = inter[valido].groupby("departamento")["area_m2"].sum() / area_dep
        out[f"cob_{v}"] = out["departamento"].map(con).fillna(0.0)

    # Un promedio calculado sobre una fracción mínima del departamento no es un dato,
    # es un artefacto: Río Seco daba 205 mm a partir de <5 % de su superficie, y
    # Capital 149 mm a partir del 20 %. Se anulan por debajo del umbral en vez de
    # dejarlos pasar como si fueran comparables al resto.
    for v in VARS:
        flojo = out[f"cob_{v}"] < min_cobertura
        if flojo.any():
            log.warning("%s: %d deptos anulados por cobertura < %.0f%% (%s)",
                        v, int(flojo.sum()), 100 * min_cobertura,
                        ", ".join(out.loc[flojo, "departamento"]))
            out.loc[flojo, f"idecor_{v}"] = np.nan
    out = out[out[[f"idecor_{v}" for v in VARS]].notna().any(axis=1)].reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    log.info("IDECOR: %d deptos × %d variables → %s", len(out), len(VARS), OUT)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="ignora la caché")
    ap.add_argument("--min-cobertura", type=float, default=0.7,
                    help="fracción mínima del depto con dato para aceptar el promedio")
    a = ap.parse_args()
    df = build_suelo_idecor(force=a.force, min_cobertura=a.min_cobertura)
    pd.set_option("display.width", 160)
    print("\n" + df.round(1).to_string(index=False))
