"""
build_capas_estaticas.py — Capas ESTÁTICAS por departamento: suelo y geografía.

Complemento de build_panel_union.py. La diferencia de fondo: estas capas NO varían
por campaña ni por cultivo, así que no hay bucle temporal — una sola extracción por
departamento y el valor se repite en todas sus filas del panel.

Fuentes (todas vía Google Earth Engine, mismo auth/proyecto que CHIRPS/NDVI/ERA5):
  - SoilGrids v2.0 (ISRIC, 250 m): textura (clay/sand/silt), densidad aparente,
    CIC, nitrógeno, pH, carbono orgánico, y contenido volumétrico de agua a
    10/33/1500 kPa. Profundidades 0-5, 5-15, 15-30 y 30-60 cm (zona radicular
    de soja/maíz; no hace falta bajar a 60-200 cm).
  - SRTM GL1 (30 m): elevación media, desvío de elevación (rugosidad) y pendiente.
  - HydroSHEDS Free-Flowing Rivers: distancia media a la red fluvial relevante.

Geometría: polígonos departamentales FAO/GAUL/2015/level2 — los mismos que ya usan
NDVI y ERA5 en build_panel_union.py (CHIRPS y NASA POWER usan centroide, no polígono).

Convención source-only: se guardan los valores de la fuente, sin derivar features.
Se aplican SOLO los factores de conversión de unidades documentados por ISRIC
(SoilGrids publica enteros escalados), igual que build_panel_union.py ya aplica
NDVI_SCALE al NDVI crudo. El agua útil (wv0033 - wv1500) y cualquier agregación
por profundidad se calculan aguas abajo, en componente_a/src/data.py.

Cachés independientes por grupo: si falla un grupo, los otros no se pierden.

Uso:
    python eda/build_capas_estaticas.py                  # todo, usando cachés
    python eda/build_capas_estaticas.py --force          # reextrae todo
    python eda/build_capas_estaticas.py --solo suelo     # suelo | geo | rios
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── Config GEE (mismos valores que build_panel_union.py) ───────────────────
EE_PROJECT = "tp-final-ml-499915"
GAUL = "FAO/GAUL/2015/level2"
GEE_MAX_RETRIES = 5
GEE_BACKOFF_BASE = 4

# ── SoilGrids ──────────────────────────────────────────────────────────────
# Profundidades: (token del asset, sufijo de columna). El token difiere entre las
# Images por propiedad ('0-5cm') y la ImageCollection de agua ('0_5'), ver abajo.
PROFUNDIDADES = [("0-5cm", "0_5"), ("5-15cm", "5_15"),
                 ("15-30cm", "15_30"), ("30-60cm", "30_60")]

# propiedad -> (factor de conversión, unidad final). Factores de la doc de ISRIC:
# SoilGrids publica enteros escalados para ahorrar espacio.
SUELO_PROPS = {
    "bdod":     (0.01, "g/cm3"),      # cg/cm3  -> g/cm3
    "cec":      (0.1,  "cmol(c)/kg"), # mmol/kg -> cmol/kg
    "clay":     (0.1,  "%"),          # g/kg    -> %
    "sand":     (0.1,  "%"),
    "silt":     (0.1,  "%"),
    "nitrogen": (0.01, "g/kg"),       # cg/kg   -> g/kg
    "phh2o":    (0.1,  "pH"),         # pH*10   -> pH
    "soc":      (0.1,  "g/kg"),       # dg/kg   -> g/kg
}

# Contenido volumétrico de agua. Dos trampas acá, ambas verificadas contra GEE:
#
#  1) NO existen como Image suelta ('projects/soilgrids-isric/wv0033_mean' da
#     asset-not-found): viven en esta ImageCollection, y sus bandas se llaman
#     'val_<prof>cm_mean' — NO 'wv0033_<prof>_mean' como las demás propiedades.
#
#  2) El factor de conversión es DISTINTO al de las otras propiedades. Los assets
#     'projects/soilgrids-isric/*' devuelven enteros escalados (arcilla en
#     Pergamino = 275 g/kg → ×0.1 = 27.5 %), pero esta colección curada por GEE
#     ya devuelve floats en unidades nativas (wv0033 en Pergamino = 0.328
#     cm3/cm3). Por eso el factor es 100 (fracción → %), no 0.1: llevarlas a %
#     las deja en la misma unidad que clay/sand/silt.
SUELO_IC = "ISRIC/SoilGrids250m/v2_0"
SUELO_WV = {
    "wv0010": (100.0, "% vol"),  # 10 kPa   — cerca de saturación
    "wv0033": (100.0, "% vol"),  # 33 kPa   — capacidad de campo
    "wv1500": (100.0, "% vol"),  # 1500 kPa — punto de marchitez permanente
}
SUELO_SCALE_M = 250

# CRS explícito en TODOS los reduceRegions. No es opcional: los assets
# 'projects/soilgrids-isric/*' están en World_Mollweide, y si no se fuerza el CRS
# GEE reduce sobre la grilla nativa y devuelve None en casi toda Argentina
# (en una corrida sin esto, 520 de 525 departamentos salieron nulos, sin error:
# reduceRegions "tiene éxito" y entrega la propiedad vacía). La ImageCollection
# ISRIC/SoilGrids250m/v2_0 sí está en 4326, pero lo pasamos igual: es no-op ahí.
CRS = "EPSG:4326"

# ── Geografía ──────────────────────────────────────────────────────────────
SRTM = "USGS/SRTMGL1_003"
SRTM_SCALE_M = 250          # SRTM es de 30 m; a 250 m la media departamental no
                            # cambia y el reduce es mucho más barato.
RIOS = "WWF/HydroSHEDS/v1/FreeFlowingRivers"
RIOS_ORD_MAX = 5            # orden de río: <=5 deja los cursos relevantes
                            # (Paraná, Carcarañá) y descarta arroyos menores.
RIOS_RADIO_M = 500_000
RIOS_SCALE_M = 1_000


# ── Helpers GEE (calcados de build_panel_union.py) ─────────────────────────

def _ee_init():
    import ee
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as e:
        log.warning("EE.Initialize falló (%s). Autenticando...", e)
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    return ee


def _gee_getinfo_retry(obj, descripcion=""):
    last_err = None
    STRUCTURAL = ("over 5000 elements", "collection query aborted")
    for intento in range(1, GEE_MAX_RETRIES + 1):
        try:
            return obj.getInfo()
        except Exception as e:
            if any(m in str(e).lower() for m in STRUCTURAL):
                raise RuntimeError(f"GEE error estructural ({e})") from e
            last_err = e
            espera = GEE_BACKOFF_BASE * (2 ** (intento - 1))
            log.warning("GEE retry %d/%d (%s) → backoff %ds",
                        intento, GEE_MAX_RETRIES, e, espera)
            time.sleep(espera)
    raise RuntimeError(f"GEE agotó reintentos para {descripcion}: {last_err}")


def _deptos_argentina(ee):
    return ee.FeatureCollection(GAUL).filter(ee.Filter.eq("ADM0_NAME", "Argentina"))


def _reduce_a_filas(ee, img, deptos, scale, cols_map, descripcion):
    """reduceRegions(mean) sobre los polígonos → lista de dicts {provincia, departamento, ...}.

    `cols_map` mapea nombre de banda GEE → (columna de salida, factor de conversión).
    `tileScale` sube para que el reduce no se quede sin memoria: los departamentos
    argentinos son grandes y SoilGrids es de 250 m."""
    fc = img.reduceRegions(collection=deptos, reducer=ee.Reducer.mean(),
                           scale=scale, crs=CRS, tileScale=4)
    info = _gee_getinfo_retry(fc, descripcion=descripcion)
    filas, nulos = [], 0
    for ft in info["features"]:
        p = ft["properties"]
        fila = {"provincia": p.get("ADM1_NAME"), "departamento": p.get("ADM2_NAME")}
        for banda, (col, factor) in cols_map.items():
            v = p.get(banda)
            if v is None:
                nulos += 1
            fila[col] = None if v is None else v * factor
        filas.append(fila)
    # Un reduce mal proyectado NO tira excepción: devuelve todo None. Avisamos.
    total = len(filas) * len(cols_map)
    if total and nulos / total > 0.5:
        log.warning("%s: %.0f%% de valores nulos (%d/%d) — ¿problema de CRS/cobertura?",
                    descripcion, 100 * nulos / total, nulos, total)
    return filas


# ── Grupo 1: SoilGrids ─────────────────────────────────────────────────────

def load_suelo(force: bool = False) -> pd.DataFrame:
    """Propiedades de suelo por departamento (SoilGrids v2.0, 4 profundidades).

    Un getInfo por propiedad (11 en total): cada uno devuelve ~525 features × 4
    bandas, muy por debajo del límite de 5000 elementos de GEE. Chunkear por
    propiedad en vez de por región mantiene el script simple y hace que un fallo
    puntual cueste una sola propiedad."""
    out = PROC / "suelo_soilgrids_wide.parquet"
    if out.exists() and not force:
        log.info("SUELO: usando caché %s", out)
        return pd.read_parquet(out)

    ee = _ee_init()
    deptos = _deptos_argentina(ee)
    acumulado: dict[tuple, dict] = {}

    def _acumular(filas):
        for f in filas:
            key = (f["provincia"], f["departamento"])
            acumulado.setdefault(key, {"provincia": f["provincia"],
                                       "departamento": f["departamento"]}).update(
                {k: v for k, v in f.items() if k not in ("provincia", "departamento")})

    # 1a) Propiedades publicadas como Image por propiedad.
    for prop, (factor, unidad) in SUELO_PROPS.items():
        log.info("SUELO: extrayendo %s (%s)", prop, unidad)
        img = ee.Image(f"projects/soilgrids-isric/{prop}_mean")
        cols_map = {f"{prop}_{tok}_mean": (f"suelo_{prop}_{sfx}", factor)
                    for tok, sfx in PROFUNDIDADES}
        img = img.select(list(cols_map))
        try:
            _acumular(_reduce_a_filas(ee, img, deptos, SUELO_SCALE_M, cols_map,
                                      f"SoilGrids {prop}"))
        except Exception as e:
            log.warning("SUELO: saltando %s (%s)", prop, e)

    # 1b) Contenido de agua: ImageCollection con bandas 'val_<prof>cm_mean'.
    ic = ee.ImageCollection(SUELO_IC)
    for wv, (factor, unidad) in SUELO_WV.items():
        log.info("SUELO: extrayendo %s (%s)", wv, unidad)
        img = ee.Image(ic.filter(ee.Filter.eq("system:index", wv)).first())
        cols_map = {f"val_{sfx}cm_mean": (f"suelo_{wv}_{sfx}", factor)
                    for _, sfx in PROFUNDIDADES}
        img = img.select(list(cols_map))
        try:
            _acumular(_reduce_a_filas(ee, img, deptos, SUELO_SCALE_M, cols_map,
                                      f"SoilGrids {wv}"))
        except Exception as e:
            log.warning("SUELO: saltando %s (%s)", wv, e)

    if not acumulado:
        log.error("SUELO: no se extrajo nada")
        return pd.DataFrame()
    wide = pd.DataFrame(list(acumulado.values()))
    wide.to_parquet(out, index=False)
    log.info("SUELO: %d deptos × %d columnas → %s", len(wide), wide.shape[1] - 2, out)
    return wide


# ── Grupo 2: elevación y pendiente ─────────────────────────────────────────

def load_geo(force: bool = False) -> pd.DataFrame:
    """Elevación (media y desvío) y pendiente media por departamento (SRTM).

    El desvío de elevación es un proxy de rugosidad del terreno. La pendiente se
    calcula a 250 m, no a los 30 m nativos: en la Pampa la diferencia es
    despreciable para una media departamental y el reduce es mucho más barato."""
    out = PROC / "geo_srtm_wide.parquet"
    if out.exists() and not force:
        log.info("GEO: usando caché %s", out)
        return pd.read_parquet(out)

    ee = _ee_init()
    deptos = _deptos_argentina(ee)
    dem = ee.Image(SRTM).select("elevation")
    img = dem.addBands(ee.Terrain.slope(dem).rename("slope"))

    reductor = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    fc = img.reduceRegions(collection=deptos, reducer=reductor,
                           scale=SRTM_SCALE_M, crs=CRS, tileScale=4)
    info = _gee_getinfo_retry(fc, descripcion="SRTM")

    filas = []
    for ft in info["features"]:
        p = ft["properties"]
        filas.append({
            "provincia": p.get("ADM1_NAME"), "departamento": p.get("ADM2_NAME"),
            "geo_elev_mean":  p.get("elevation_mean"),
            "geo_elev_std":   p.get("elevation_stdDev"),
            "geo_slope_mean": p.get("slope_mean"),
        })
    wide = pd.DataFrame(filas).dropna(
        subset=["geo_elev_mean", "geo_slope_mean"], how="all")
    wide.to_parquet(out, index=False)
    log.info("GEO: %d deptos → %s", len(wide), out)
    return wide


# ── Grupo 3: distancia a cursos de agua ────────────────────────────────────

def load_rios(force: bool = False) -> pd.DataFrame:
    """Distancia media del departamento a la red fluvial relevante (RIV_ORD<=5).

    Desvío deliberado del spec, que pedía la distancia desde el CENTROIDE: acá se
    promedia la distancia sobre el polígono, para quedar consistente con el resto
    de las capas (todas son medias sobre el polígono GAUL). En departamentos
    grandes o alargados la media es además más representativa que el centroide."""
    out = PROC / "geo_rios_wide.parquet"
    if out.exists() and not force:
        log.info("RIOS: usando caché %s", out)
        return pd.read_parquet(out)

    ee = _ee_init()
    deptos = _deptos_argentina(ee)
    rios = (ee.FeatureCollection(RIOS)
            .filter(ee.Filter.lte("RIV_ORD", RIOS_ORD_MAX))
            .filterBounds(deptos.geometry().bounds()))
    dist_km = rios.distance(RIOS_RADIO_M).divide(1000.0).rename("dist_rio_km")

    fc = dist_km.reduceRegions(collection=deptos, reducer=ee.Reducer.mean(),
                               scale=RIOS_SCALE_M, crs=CRS, tileScale=4)
    info = _gee_getinfo_retry(fc, descripcion="HydroSHEDS")

    filas = [{"provincia": ft["properties"].get("ADM1_NAME"),
              "departamento": ft["properties"].get("ADM2_NAME"),
              "geo_dist_rio_km": ft["properties"].get("mean")}
             for ft in info["features"]]
    wide = pd.DataFrame(filas)
    wide.to_parquet(out, index=False)
    log.info("RIOS: %d deptos → %s", len(wide), out)
    return wide


# ── Ensamble ───────────────────────────────────────────────────────────────

def build_capas_estaticas(force: bool = False, solo: str | None = None) -> pd.DataFrame:
    """Une los tres grupos en un único wide por (provincia, departamento)."""
    partes = []
    if solo in (None, "suelo"):
        partes.append(load_suelo(force))
    if solo in (None, "geo"):
        partes.append(load_geo(force))
    if solo in (None, "rios"):
        partes.append(load_rios(force))
    partes = [p for p in partes if len(p)]
    if not partes:
        log.error("no se extrajo ninguna capa")
        return pd.DataFrame()

    wide = partes[0]
    for p in partes[1:]:
        wide = wide.merge(p, on=["provincia", "departamento"], how="outer")

    if solo is not None:
        # Con --solo el resultado es PARCIAL: no piso el combinado, que es el que
        # consume build_panel_union.py. Cada grupo ya quedó en su propia caché.
        log.info("CAPAS ESTÁTICAS (--solo %s): %d deptos × %d columnas. "
                 "No se reescribe el combinado; correr sin --solo para eso.",
                 solo, len(wide), wide.shape[1] - 2)
        return wide

    out = PROC / "capas_estaticas_wide.parquet"
    wide.to_parquet(out, index=False)
    log.info("CAPAS ESTÁTICAS: %d deptos × %d columnas → %s",
             len(wide), wide.shape[1] - 2, out)
    return wide


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="ignora las cachés y reextrae de GEE")
    ap.add_argument("--solo", choices=["suelo", "geo", "rios"], default=None,
                    help="extrae un solo grupo de capas")
    args = ap.parse_args()
    build_capas_estaticas(force=args.force, solo=args.solo)
