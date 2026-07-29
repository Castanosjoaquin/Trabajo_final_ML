"""
build_panel_union.py — Panel expandido: unión de todos los departamentos con
cobertura mínima, sin restricción a la región núcleo BCR. Es el builder canónico
del panel del Componente A.

Características:
  - MIN_CAMPANAS por cultivo (default 20), sin REGION_NUCLEO hardcodeado.
  - Centroides via Nominatim (OSM) con caché local; los 26 del núcleo se preservan tal cual.
  - Columna 'region' para ablations (núcleo vs resto).
  - Fuentes: MAGyP (rinde) + ONI + NASA POWER + CHIRPS + NDVI-AVHRR + ERA5-Land.
    NDVI y ERA5 se extraen de GEE acá mismo (paso 6b, mismo auth que CHIRPS) y se
    mergean al panel; las filas sin cobertura satelital se descartan. El NDVI de
    MODIS se retiró (2026-07). Antes NDVI/ERA5 vivían en data_sources/ (borrado).
  - Sin imputación de rinde: se reportan faltantes al final.
  - Output: data/processed/panel_union.parquet (panel ÚNICO del proyecto).

Uso:
    python eda/build_panel_union.py                      # MIN_CAMPANAS=20
    python eda/build_panel_union.py --min-campanas 30
    python eda/build_panel_union.py --min-campanas 20 --skip-chirps
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
RAW   = ROOT / "data" / "raw"
PROC  = ROOT / "data" / "processed"
for d in [RAW, PROC]:
    d.mkdir(parents=True, exist_ok=True)

# ── Centroides existentes del núcleo (preservados exactamente) ─────────────
CENTROIDES_NUCLEO = {
    "Caseros":               (-33.03, -61.47),
    "Villa Constitucion":    (-33.67, -60.33),
    "General Lopez":         (-34.17, -61.88),
    "Rosario":               (-33.02, -60.63),
    "San Lorenzo":           (-32.75, -60.73),
    "Iriondo":               (-32.93, -61.23),
    "Belgrano":              (-32.57, -61.57),
    "Marcos Juarez":         (-33.03, -62.10),
    "Union":                 (-33.13, -62.87),
    "Juarez Celman":         (-33.40, -63.42),
    "General San Martin":    (-32.72, -62.98),
    "Pergamino":             (-33.89, -60.57),
    "Colon":                 (-33.89, -61.10),
    "Rojas":                 (-34.20, -60.73),
    "Salto":                 (-34.29, -60.25),
    "San Nicolas":           (-33.34, -60.22),
    "Ramallo":               (-33.49, -60.01),
    "San Pedro":             (-33.68, -59.67),
    "Baradero":              (-33.81, -59.51),
    "Arrecifes":             (-34.07, -60.10),
    "Capitan Sarmiento":     (-34.18, -59.79),
    "Carmen de Areco":       (-34.38, -59.82),
    "Chacabuco":             (-34.64, -60.47),
    "Junin":                 (-34.59, -60.95),
    "General Arenales":      (-34.32, -61.28),
    "Leandro N. Alem":       (-34.23, -61.47),
}
NUCLEO_DEPTOS = set(CENTROIDES_NUCLEO.keys())

# Provincia de cada departamento del núcleo. Vive a nivel de módulo porque la usan
# DOS pasos: la asignación de `region` (paso 1) y la geocodificación (paso 2, que
# inyecta los centroides canónicos). Sin la provincia, marcar el núcleo por nombre
# solo arrastra homónimos de otras provincias: Chacabuco y Junín de San Luis,
# San Pedro de Misiones y de Jujuy, Belgrano de Santiago del Estero, etc.
PROV_NUCLEO = {
    "Caseros": "SANTA FE", "Villa Constitucion": "SANTA FE",
    "General Lopez": "SANTA FE", "Rosario": "SANTA FE",
    "San Lorenzo": "SANTA FE", "Iriondo": "SANTA FE", "Belgrano": "SANTA FE",
    "Marcos Juarez": "CORDOBA", "Union": "CORDOBA",
    "Juarez Celman": "CORDOBA", "General San Martin": "CORDOBA",
    "Pergamino": "BUENOS AIRES", "Colon": "BUENOS AIRES",
    "Rojas": "BUENOS AIRES", "Salto": "BUENOS AIRES",
    "San Nicolas": "BUENOS AIRES", "Ramallo": "BUENOS AIRES",
    "San Pedro": "BUENOS AIRES", "Baradero": "BUENOS AIRES",
    "Arrecifes": "BUENOS AIRES", "Capitan Sarmiento": "BUENOS AIRES",
    "Carmen de Areco": "BUENOS AIRES", "Chacabuco": "BUENOS AIRES",
    "Junin": "BUENOS AIRES", "General Arenales": "BUENOS AIRES",
    "Leandro N. Alem": "BUENOS AIRES",
}

# ── Parámetros NASA POWER ──────────────────────────────────────────────────
NASA_END_YEAR = 2025
NASA_PARAMS   = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M"
MESES_AE      = [9, 10, 11, 12, 1, 2, 3]
MES_NAMES     = {9:"sep", 10:"oct", 11:"nov", 12:"dic", 1:"ene", 2:"feb", 3:"mar"}

# ── Parámetros CHIRPS (Earth Engine) ──────────────────────────────────────
EE_PROJECT         = "tp-final-ml-499915"
CHIRPS_DATASET     = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_BAND        = "precipitation"
CHIRPS_START       = "1981-01-01"
CHIRPS_END         = "2025-07-01"
CHIRPS_SCALE_FB    = 5566
GEE_MAX_RETRIES    = 5
GEE_BACKOFF_BASE   = 4


# ── Helpers ────────────────────────────────────────────────────────────────

def normalizar(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split()).title()


def _norm_one(x) -> str:
    """Un nombre normalizado para comparar entre fuentes: sin acentos, en
    mayúsculas y con espacios colapsados. Es la forma canónica del proyecto para
    matchear departamentos/provincias, porque MAGyP, GAUL y las constantes de este
    archivo los escriben distinto ('Villa Constitución' vs 'Villa Constitucion',
    'Carmen de Areco' vs 'Carmen De Areco')."""
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return " ".join(x.upper().split())


def campaign_year(camp_str: str) -> int:
    return int(str(camp_str).split("/")[0])


def mes_a_campania_inicio(mes: int, anio: int):
    if mes >= 9:
        return anio
    elif mes <= 3:
        return anio - 1
    return None


# ── Paso 1: MAGyP sin filtro geográfico ───────────────────────────────────

def _asignar_region(full: pd.DataFrame) -> pd.DataFrame:
    """Marca cada fila como 'nucleo' o 'resto' (columna para ablations).

    Compara el par (departamento, provincia) con los nombres NORMALIZADOS. Las dos
    cosas importan: sin la provincia entran homónimos de otras regiones (Chacabuco
    y Junín de San Luis, San Pedro de Misiones y Jujuy, Belgrano de Santiago del
    Estero); sin normalizar quedan afuera núcleos legítimos que las fuentes
    escriben distinto ('Carmen De Areco' en el panel vs 'Carmen de Areco' acá).
    Con el match por nombre suelto salían 34 deptos núcleo en vez de 25."""
    nucleo_norm = {(_norm_one(d), _norm_one(p)) for d, p in PROV_NUCLEO.items()}
    full = full.copy()
    full["region"] = [
        "nucleo" if (_norm_one(d), _norm_one(p)) in nucleo_norm else "resto"
        for d, p in zip(full["departamento"], full["provincia"])
    ]
    return full


def load_magyp_full(min_campanas: int) -> pd.DataFrame:
    """Lee ambos CSVs de MAGyP sin restricción geográfica.
    Aplica MIN_CAMPANAS por (departamento, provincia, cultivo).
    No imputa rinde: reporta NaN count al final."""
    out = RAW / f"magyp_full_min{min_campanas}.parquet"
    if out.exists():
        log.info("MAGyP full: usando caché %s", out)
        # `region` se recalcula SIEMPRE, también al leer de caché: es derivada y
        # barata, y las cachés viejas la traen mal (se asignaba por nombre de
        # depto sin provincia). Si se dejara pasar, el fix quedaría neutralizado
        # por un parquet de hace meses, en silencio.
        return _asignar_region(pd.read_parquet(out))

    frames = []
    for cultivo, fname in [("soja", "magyp_soja.csv"), ("maiz", "magyp_maiz.csv")]:
        fpath = RAW / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Falta {fpath}")
        log.info("MAGyP: leyendo %s...", fname)

        df = None
        for sep in [",", ";", "\t"]:
            for enc in ["utf-8-sig", "latin1", "utf-8"]:
                try:
                    tmp = pd.read_csv(fpath, sep=sep, encoding=enc, nrows=3, dtype=str)
                    if len(tmp.columns) >= 4:
                        df = pd.read_csv(fpath, sep=sep, encoding=enc, dtype=str)
                        df.columns = [c.strip() for c in df.columns]
                        break
                except Exception:
                    pass
            if df is not None:
                break
        if df is None:
            raise ValueError(f"No se pudo parsear {fpath}")

        col_map = {}
        for c in df.columns:
            cl = c.lower().strip().replace(".", "").replace(" ", "_")
            if cl == "cultivo":                        col_map[c] = "cultivo"
            elif "campa" in cl:                        col_map[c] = "campania"
            elif cl == "provincia":                    col_map[c] = "provincia"
            elif cl in ("departamento", "partido"):    col_map[c] = "departamento"
            elif "sembrada" in cl:                     col_map[c] = "sup_sembrada_ha"
            elif "cosechada" in cl:                    col_map[c] = "sup_cosechada_ha"
            elif "producci" in cl:                     col_map[c] = "produccion_tn"
            elif "rendimiento" in cl:                  col_map[c] = "rinde_kgha"
        df = df.rename(columns=col_map)

        df["cultivo"] = cultivo
        df["provincia"] = df["provincia"].apply(normalizar).str.upper()
        df["departamento"] = df["departamento"].apply(normalizar)
        # Fix nombre canónico
        df["departamento"] = df["departamento"].replace({
            "Constitucion": "Villa Constitucion",
            "Major Luis J. Fontana": "Mayor Luis Jorge Fontana",
            "Mayor Luis J. Fontana": "Mayor Luis Jorge Fontana",
        })

        df["campania_inicio"] = df["campania"].apply(campaign_year)
        for col in ["sup_sembrada_ha", "sup_cosechada_ha", "produccion_tn", "rinde_kgha"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["campania_inicio"] >= 1981].copy()
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    # Filtro MIN_CAMPANAS: contar campañas con rinde no nulo por (depto, prov, cultivo)
    camp_counts = (
        full[full["rinde_kgha"].notna() & (full["rinde_kgha"] > 0)]
        .groupby(["departamento", "provincia", "cultivo"])["campania_inicio"]
        .nunique()
        .rename("n_camp")
        .reset_index()
    )
    valid = camp_counts[camp_counts["n_camp"] >= min_campanas][
        ["departamento", "provincia", "cultivo"]
    ]
    full = full.merge(valid, on=["departamento", "provincia", "cultivo"], how="inner")

    full = _asignar_region(full)

    n_deptos = full.groupby("cultivo")["departamento"].nunique()
    log.info("MAGyP full (min_camp=%d): %d filas | soja=%d deptos | maiz=%d deptos",
             min_campanas, len(full),
             n_deptos.get("soja", 0), n_deptos.get("maiz", 0))
    full.to_parquet(out, index=False)
    return full


# ── Paso 2: Geocodificación con Nominatim ─────────────────────────────────

def geocode_deptos(deptos_prov: pd.DataFrame) -> pd.DataFrame:
    """Para cada (departamento, provincia) devuelve (lat, lon).
    Usa caché local. Los del núcleo se inyectan directamente sin llamar a Nominatim.
    Respeta rate limit OSM: 1 req/s."""
    cache_path = RAW / "centroides_all_deptos.parquet"
    cache = {}
    if cache_path.exists():
        df_c = pd.read_parquet(cache_path)
        for _, row in df_c.iterrows():
            cache[(row["departamento"], row["provincia"])] = (row["lat"], row["lon"])
        log.info("Centroides: caché con %d entradas", len(cache))

    # Inyectar núcleo (son canónicos, no geocodificar). PROV_NUCLEO vive a nivel de
    # módulo: lo comparte la asignación de `region` en load_magyp_full.
    for depto, (lat, lon) in CENTROIDES_NUCLEO.items():
        prov = PROV_NUCLEO[depto]
        cache[(depto, prov)] = (lat, lon)

    # Geocodificar los que faltan
    pendientes = [
        (d, p) for d, p in zip(deptos_prov["departamento"], deptos_prov["provincia"])
        if (d, p) not in cache
    ]
    if pendientes:
        log.info("Geocodificando %d departamentos nuevos via Nominatim...", len(pendientes))

    headers = {"User-Agent": "tp-final-ml-anomalias/1.0 (castanosjoaquin@gmail.com)"}
    failed = []
    for i, (depto, prov) in enumerate(pendientes):
        # Nominatim query: "Departamento X, Provincia Y, Argentina"
        prov_title = prov.title()
        query = f"{depto}, {prov_title}, Argentina"
        url = (
            "https://nominatim.openstreetmap.org/search"
            f"?q={requests.utils.quote(query)}&format=json&limit=1&countrycodes=ar"
        )
        try:
            r = requests.get(url, headers=headers, timeout=10)
            results = r.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                cache[(depto, prov)] = (lat, lon)
                log.info("  [%d/%d] %s, %s → (%.4f, %.4f)",
                         i+1, len(pendientes), depto, prov, lat, lon)
            else:
                log.warning("  [%d/%d] %s, %s → SIN RESULTADO", i+1, len(pendientes), depto, prov)
                failed.append((depto, prov))
        except Exception as e:
            log.warning("  [%d/%d] %s, %s → ERROR: %s", i+1, len(pendientes), depto, prov, e)
            failed.append((depto, prov))
        time.sleep(1.1)  # OSM rate limit: 1 req/s

    if failed:
        log.warning("No se pudo geocodificar %d departamentos: %s", len(failed), failed[:10])

    # Guardar caché actualizado
    rows = [{"departamento": d, "provincia": p, "lat": lat, "lon": lon}
            for (d, p), (lat, lon) in cache.items()]
    pd.DataFrame(rows).to_parquet(cache_path, index=False)
    log.info("Centroides: caché guardado con %d entradas", len(rows))

    return pd.DataFrame(rows)


# ── Paso 3: NASA POWER ────────────────────────────────────────────────────

def fetch_power_departamento(depto: str, lat: float, lon: float,
                              start_year=1981, end_year=NASA_END_YEAR):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={NASA_PARAMS}&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_year}0101&end={end_year}1231&format=JSON"
    )
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        log.warning("NASA POWER error %d para %s", r.status_code, depto)
        return None
    data = r.json()
    params = data["properties"]["parameter"]
    dates = list(next(iter(params.values())).keys())
    records = []
    for d in dates:
        row = {"fecha": pd.to_datetime(d, format="%Y%m%d"), "departamento": depto}
        for param, vals in params.items():
            v = vals.get(d, np.nan)
            row[param.lower()] = np.nan if v == -999.0 else v
        records.append(row)
    return pd.DataFrame(records)


def load_nasa_power_for(centroides: pd.DataFrame) -> pd.DataFrame:
    """Descarga NASA POWER para todos los departamentos de `centroides`.
    Reutiliza caché existente por departamento (mismo naming que build_panel_chirps)."""
    out_panel = RAW / "nasa_power_union_wide.parquet"
    all_daily = []

    for _, row in centroides.iterrows():
        depto = row["departamento"]
        lat, lon = row["lat"], row["lon"]
        cache_key = depto.lower().replace(" ", "_").replace(".", "_")
        cache_file = RAW / f"nasa_power_{cache_key}.parquet"

        if cache_file.exists():
            df_d = pd.read_parquet(cache_file)
            max_year = df_d["fecha"].dt.year.max() if len(df_d) else 0
            if max_year >= NASA_END_YEAR:
                log.info("  [caché] %s (hasta %d)", depto, max_year)
                all_daily.append(df_d)
                continue
            else:
                log.info("  [re-descargando] %s (caché hasta %d)", depto, max_year)
                cache_file.unlink()

        log.info("  [descargando] %s (%.4f, %.4f)", depto, lat, lon)
        df_d = fetch_power_departamento(depto, lat, lon, end_year=NASA_END_YEAR)
        if df_d is None:
            log.warning("  Saltando %s por error de descarga", depto)
            continue
        df_d.to_parquet(cache_file, index=False)
        all_daily.append(df_d)
        time.sleep(1)

    if not all_daily:
        log.error("NASA POWER: ningún departamento descargado.")
        return pd.DataFrame()

    df_daily = pd.concat(all_daily, ignore_index=True)
    df_daily["anio"] = df_daily["fecha"].dt.year
    df_daily["mes"]  = df_daily["fecha"].dt.month
    df_daily["campania_inicio"] = df_daily.apply(
        lambda r: mes_a_campania_inicio(r["mes"], r["anio"]), axis=1
    )

    clim_cols = [c for c in df_daily.columns
                 if c not in ["fecha", "departamento", "anio", "mes", "campania_inicio"]]

    df_ae = df_daily[df_daily["mes"].isin(MESES_AE)].copy()
    monthly = (
        df_ae.groupby(["departamento", "campania_inicio", "mes"])[clim_cols]
        .mean().reset_index()
    )
    monthly["mes_str"] = monthly["mes"].map(MES_NAMES)

    wide = monthly.pivot_table(
        index=["departamento", "campania_inicio"],
        columns="mes_str",
        values=clim_cols,
        aggfunc="mean",
    )
    wide.columns = [f"{v}_{m}" for v, m in wide.columns]
    wide = wide.reset_index()

    log.info("NASA POWER union: %d filas | %d deptos | campañas %d–%d",
             len(wide), wide["departamento"].nunique(),
             wide["campania_inicio"].min(), wide["campania_inicio"].max())
    wide.to_parquet(out_panel, index=False)
    return wide


# ── Paso 4: ONI ──────────────────────────────────────────────────────────

def load_oni() -> pd.DataFrame:
    """Igual que en build_panel_chirps. Lee caché si existe."""
    out = RAW / "oni_campana_wide.parquet"
    if out.exists():
        log.info("ONI: usando caché %s", out)
        return pd.read_parquet(out)

    log.info("ONI: descargando...")
    url = "https://raw.githubusercontent.com/ahuang11/oni/master/oni.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    SEASON_TO_MONTH = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    df["mes"] = df["season"].map(SEASON_TO_MONTH)
    df = df[df["year"] >= 1980].copy()
    df["anom"] = df["anom_c"]

    def assign_campaign(row):
        if row["mes"] >= 10:   return row["year"]
        elif row["mes"] <= 2:  return row["year"] - 1
        return None

    df["campania_inicio"] = df.apply(assign_campaign, axis=1)
    df = df.dropna(subset=["campania_inicio"])
    df["campania_inicio"] = df["campania_inicio"].astype(int)

    oct_feb = df[df["mes"].isin([10, 11, 12, 1, 2])].copy()
    MES_NAMES_ONI = {10:"oct", 11:"nov", 12:"dic", 1:"ene", 2:"feb"}
    oct_feb["mes_str"] = oct_feb["mes"].map(MES_NAMES_ONI)
    camp_oni = oct_feb.pivot_table(
        index="campania_inicio", columns="mes_str", values="anom", aggfunc="first"
    ).reset_index()
    camp_oni.columns.name = None
    camp_oni = camp_oni.rename(columns={
        "oct":"oni_oct","nov":"oni_nov","dic":"oni_dic","ene":"oni_ene","feb":"oni_feb"
    })
    camp_oni.to_parquet(out, index=False)
    return camp_oni


# NOTA (2026-07): el NDVI de MODIS se RETIRÓ del panel. Existía solo desde 2002 y
# 3 de sus 4 columnas eran estáticas por departamento (cero señal temporal). El
# único NDVI que se usa es el AVHRR/VIIRS mensual 1981+ (`ndvi_avhrr_<mes>`), que
# se extrae e integra en el paso 6b de este mismo builder.


# ── Paso 5: CHIRPS ────────────────────────────────────────────────────────

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
            log.warning("GEE retry %d/%d (%s) → backoff %ds", intento, GEE_MAX_RETRIES, e, espera)
            time.sleep(espera)
    raise RuntimeError(f"GEE agotó reintentos para {descripcion}: {last_err}")


def load_chirps_for(centroides: pd.DataFrame) -> pd.DataFrame:
    """CHIRPS mensual para todos los departamentos de `centroides`.
    Cachea en data/processed/chirps_union_wide.parquet.
    Si ya existe y está actualizado, lo reutiliza."""
    out = PROC / "chirps_union_wide.parquet"
    if out.exists():
        log.info("CHIRPS union: usando caché %s", out)
        return pd.read_parquet(out)

    try:
        ee = _ee_init()
    except Exception as e:
        log.error("Earth Engine no disponible: %s. Saltando CHIRPS.", e)
        return pd.DataFrame()

    feats = [
        ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]]),
                   {"departamento": row["departamento"]})
        for _, row in centroides.iterrows()
    ]
    points = ee.FeatureCollection(feats)

    try:
        img = (ee.ImageCollection(CHIRPS_DATASET).select(CHIRPS_BAND)
               .filterDate("2020-01-01", "2020-01-02").first())
        scale = float(img.projection().nominalScale().getInfo())
    except Exception:
        scale = float(CHIRPS_SCALE_FB)
    log.info("CHIRPS: escala = %.1f m | descargando mes a mes...", scale)

    fin = pd.Timestamp(CHIRPS_END)
    periodo = pd.Timestamp(CHIRPS_START).replace(day=1)
    frames = []

    while periodo < fin:
        siguiente = periodo + pd.offsets.MonthBegin(1)
        start = max(pd.Timestamp(CHIRPS_START), periodo)
        end   = min(fin, siguiente)
        if start >= end:
            periodo = siguiente
            continue

        log.info("CHIRPS: %s .. %s", start.date(), end.date())

        # Partimos el mes en chunks para no superar el límite de GEE.
        # Cada elemento ≈ 1 día x 1 departamento.
        # Usamos 4000 como margen de seguridad frente al límite de 5000.
        n_points = max(1, len(centroides))
        target_elements = 4000
        chunk_days = max(1, int(target_elements // n_points))

        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end)

            log.info(
                "CHIRPS chunk: %s .. %s (%d días)",
                chunk_start.date(),
                chunk_end.date(),
                (chunk_end - chunk_start).days,
            )

            col_chunk = (
                ee.ImageCollection(CHIRPS_DATASET)
                .select(CHIRPS_BAND)
                .filterDate(
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
            )

            def per_image(img):
                fecha = img.date().format("YYYY-MM-dd")
                fc = img.reduceRegions(
                    collection=points,
                    reducer=ee.Reducer.first(),
                    scale=scale,
                )
                return fc.map(lambda f: f.set("fecha", fecha))

            flat = ee.FeatureCollection(col_chunk.map(per_image)).flatten()

            try:
                info = _gee_getinfo_retry(
                    flat,
                    descripcion=f"CHIRPS {chunk_start.date()}..{chunk_end.date()}",
                )
            except Exception as e:
                log.warning(
                    "CHIRPS: saltando chunk %s..%s (%s)",
                    chunk_start.date(),
                    chunk_end.date(),
                    e,
                )
                chunk_start = chunk_end
                continue

            rows = []
            for feat in info["features"]:
                p = feat["properties"]
                rows.append({
                    "departamento": p.get("departamento"),
                    "fecha": p.get("fecha"),
                    "precip_mm": p.get(CHIRPS_BAND, p.get("first", np.nan)),
                })

            if rows:
                frames.append(pd.DataFrame(rows))

            time.sleep(1)
            chunk_start = chunk_end

        periodo = siguiente

    if not frames:
        log.warning("CHIRPS: no se obtuvo ningún dato.")
        return pd.DataFrame()

    daily = (pd.concat(frames, ignore_index=True)
             .assign(fecha=lambda d: pd.to_datetime(d["fecha"]))
             .assign(precip_mm=lambda d: pd.to_numeric(d["precip_mm"], errors="coerce"))
             .dropna(subset=["fecha", "departamento"])
             .drop_duplicates(["departamento", "fecha"]))

    daily["anio"] = daily["fecha"].dt.year
    daily["mes"]  = daily["fecha"].dt.month
    daily["campania_inicio"] = daily.apply(
        lambda r: mes_a_campania_inicio(r["mes"], r["anio"]), axis=1
    )
    df_ae = daily[daily["mes"].isin(MESES_AE)].copy()

    monthly = (
        df_ae.groupby(["departamento", "campania_inicio", "mes"])["precip_mm"]
        .mean().reset_index()
    )
    monthly["mes_str"] = monthly["mes"].map(MES_NAMES)
    wide = monthly.pivot_table(
        index=["departamento", "campania_inicio"],
        columns="mes_str",
        values="precip_mm",
        aggfunc="mean",
    )
    wide.columns = [f"chirps_precip_{m}" for m in wide.columns]
    wide = wide.reset_index()
    orden = ["departamento", "campania_inicio"] + [
        f"chirps_precip_{m}" for m in ["sep","oct","nov","dic","ene","feb","mar"]
        if f"chirps_precip_{m}" in wide.columns
    ]
    wide = wide[[c for c in orden if c in wide.columns]]
    wide["campania_inicio"] = wide["campania_inicio"].astype(int)

    log.info("CHIRPS union: %d filas | %d deptos", len(wide), wide["departamento"].nunique())
    wide.to_parquet(out, index=False)
    return wide


# ── Paso 6: NDVI-AVHRR + ERA5-Land (satelital, GEE) ───────────────────────
# Antes vivían en data_sources/ (extract_*.py exportaban a Drive + merge_*.py).
# Acá se folden al builder usando el MISMO auth de GEE que CHIRPS y el mismo
# patrón de getInfo chunkeado (por año), para no exceder el límite de 5000
# elementos que obligaba al export a Drive. Reduce sobre los polígonos GAUL
# nivel-2 del país (media por departamento), como los scripts originales.
GAUL = "FAO/GAUL/2015/level2"
_SAT_MONTHS = {9: "sep", 10: "oct", 11: "nov", 12: "dic", 1: "ene", 2: "feb", 3: "mar"}
NDVI_SCALE, NDVI_SCALE_M = 0.0001, 5000
ERA5_SW = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
           "volumetric_soil_water_layer_3"]          # zona radicular 0-100 cm (3 capas)
ERA5_W = [0.07, 0.21, 0.72]                           # espesores 0-7, 7-28, 28-100 cm
FREEZE_K, ERA5_SCALE_M = 273.15, 9000                 # escala nativa ERA5-Land


def _norm_ascii(s: pd.Series) -> pd.Series:
    """Nombres para el join: sin acentos, mayúsculas, espacios colapsados."""
    return s.map(_norm_one)


def load_ndvi_avhrr() -> pd.DataFrame:
    """NDVI-AVHRR/VIIRS mensual (Sep–Mar, 1981+) medio por departamento → tabla
    ancha `ndvi_avhrr_<mes>`. Cachea en data/processed/ndvi_avhrr_wide.parquet."""
    out = PROC / "ndvi_avhrr_wide.parquet"
    if out.exists():
        log.info("NDVI-AVHRR: usando caché %s", out)
        return pd.read_parquet(out)
    try:
        ee = _ee_init()
    except Exception as e:
        log.error("Earth Engine no disponible: %s. Saltando NDVI.", e)
        return pd.DataFrame()

    deptos = ee.FeatureCollection(GAUL).filter(ee.Filter.eq("ADM0_NAME", "Argentina"))
    col = (ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5").select("NDVI")
           .merge(ee.ImageCollection("NOAA/CDR/VIIRS/NDVI/V1").select("NDVI")))
    empty = ee.Image.constant(0).rename("ndvi").updateMask(ee.Image.constant(0))
    rows = []
    for year in range(1981, 2026):                 # chunk por año (evita >5000 elems)
        log.info("NDVI-AVHRR: procesando año %d", year)
        def per_month(m):
            m = ee.Number(m)
            start = ee.Date.fromYMD(year, m, 1); end = start.advance(1, "month")
            c = col.filterDate(start, end)
            img = ee.Image(ee.Algorithms.If(c.size().gt(0),
                    c.max().multiply(NDVI_SCALE).rename("ndvi"), empty))
            fc = img.reduceRegions(deptos, ee.Reducer.mean(), NDVI_SCALE_M)
            return fc.map(lambda f: f.set("month", m))
        flat = ee.FeatureCollection(ee.List(list(_SAT_MONTHS)).map(per_month)).flatten()
        try:
            info = _gee_getinfo_retry(flat, descripcion=f"NDVI {year}")
        except Exception as e:
            log.warning("NDVI: saltando %d (%s)", year, e); continue
        for ft in info["features"]:
            p = ft["properties"]
            if p.get("mean") is None:
                continue
            rows.append({"provincia": p.get("ADM1_NAME"), "departamento": p.get("ADM2_NAME"),
                         "year": year, "month": int(p["month"]), "ndvi": p["mean"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["campania_inicio"] = np.where(df["month"] >= 9, df["year"], df["year"] - 1)
    df["mes"] = df["month"].map(_SAT_MONTHS)
    wide = (df.pivot_table(index=["provincia", "departamento", "campania_inicio"],
                           columns="mes", values="ndvi", aggfunc="mean").reset_index())
    wide.columns.name = None
    wide = wide.rename(columns={m: f"ndvi_avhrr_{m}" for m in _SAT_MONTHS.values()})
    wide.to_parquet(out, index=False)
    return wide


def load_era5() -> pd.DataFrame:
    """Features ERA5-Land por campaña (humedad de suelo siembra/invierno, días de
    helada) medias por departamento. Cachea en data/processed/era5_wide.parquet."""
    out = PROC / "era5_wide.parquet"
    if out.exists():
        log.info("ERA5: usando caché %s", out)
        return pd.read_parquet(out)
    try:
        ee = _ee_init()
    except Exception as e:
        log.error("Earth Engine no disponible: %s. Saltando ERA5.", e)
        return pd.DataFrame()

    deptos = ee.FeatureCollection(GAUL).filter(ee.Filter.eq("ADM0_NAME", "Argentina"))
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

    def rootzone(c):
        img = c.select(ERA5_SW).mean()
        out_i = img.select(ERA5_SW[0]).multiply(ERA5_W[0])
        for b, w in zip(ERA5_SW[1:], ERA5_W[1:]):
            out_i = out_i.add(img.select(b).multiply(w))
        return out_i.rename("sm")

    def frost(c):
        return c.select("temperature_2m_min").map(lambda im: im.lt(FREEZE_K)).sum().rename("frost")

    rows = []
    for year in range(1981, 2025):                 # chunk por año (campañas 1981–2024)
        log.info("ERA5: procesando campaña %d", year)
        y = ee.Number(year); d = lambda yy, m: ee.Date.fromYMD(yy, m, 1)
        img = (rootzone(era5.filterDate(d(y, 9), d(y, 12))).rename("sm_planting")
               .addBands(rootzone(era5.filterDate(d(y, 6), d(y, 9))).rename("sm_winter"))
               .addBands(frost(era5.filterDate(d(y, 9), d(y.add(1), 4))).rename("frost_days"))
               .addBands(frost(era5.filterDate(d(y, 9), d(y, 12))).rename("frost_days_early")))
        fc = img.reduceRegions(deptos, ee.Reducer.mean(), ERA5_SCALE_M)
        try:
            info = _gee_getinfo_retry(fc, descripcion=f"ERA5 {year}")
        except Exception as e:
            log.warning("ERA5: saltando %d (%s)", year, e); continue
        for ft in info["features"]:
            p = ft["properties"]
            rows.append({"provincia": p.get("ADM1_NAME"), "departamento": p.get("ADM2_NAME"),
                         "campania_inicio": year,
                         **{k: p.get(k) for k in ERA5_COLS}})
    if not rows:
        return pd.DataFrame()
    wide = pd.DataFrame(rows).dropna(subset=ERA5_COLS, how="all")
    wide.to_parquet(out, index=False)
    return wide


ERA5_COLS = ["sm_planting", "sm_winter", "frost_days", "frost_days_early"]


def _merge_satelital(panel: pd.DataFrame, sat: pd.DataFrame, cols) -> pd.DataFrame:
    """Merge de features satelitales al panel por (provincia, departamento,
    campania_inicio) con nombres normalizados (sin acentos/mayúsculas)."""
    if not len(sat):
        return panel
    panel = panel.copy()
    panel["_p"], panel["_d"] = _norm_ascii(panel["provincia"]), _norm_ascii(panel["departamento"])
    sat = sat.copy()
    sat["_p"], sat["_d"] = _norm_ascii(sat["provincia"]), _norm_ascii(sat["departamento"])
    key = ["_p", "_d", "campania_inicio"]
    merged = panel.merge(sat[key + list(cols)], on=key, how="left").drop(columns=["_p", "_d"])
    log.info("satelital mergeado (+%d cols): cobertura %.1f%%",
             len(cols), 100 * merged[cols[0]].notna().mean())
    return merged


# ── Ensamble ──────────────────────────────────────────────────────────────

def build_panel_union(
    min_campanas: int = 20,
    skip_chirps: bool = False,
    skip_satelital: bool = False,
) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("BUILD PANEL UNION (min_campanas=%d) — iniciando", min_campanas)
    log.info("=" * 60)

    # 1. MAGyP sin filtro geográfico
    magyp = load_magyp_full(min_campanas)
    magyp["campania"] = magyp["campania_inicio"].apply(lambda y: f"{y}/{str(y+1)[-2:]}")

    # 2. Centroides para todos los deptos que quedaron
    deptos_prov = (
        magyp[["departamento", "provincia"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    log.info("Departamentos únicos a geocodificar: %d", len(deptos_prov))
    centroides = geocode_deptos(deptos_prov)

    # Deptos sin centroide (geocoding falló)
    sin_centroide = set(deptos_prov["departamento"]) - set(centroides["departamento"])
    if sin_centroide:
        log.warning("%d deptos SIN centroide (se excluirán de NASA/CHIRPS): %s",
                    len(sin_centroide), sorted(sin_centroide)[:10])

    # 3. ONI
    oni = load_oni()

    # 4. NASA POWER
    log.info("NASA POWER: descargando/cacheando %d departamentos...", len(centroides))
    nasa = load_nasa_power_for(centroides)

    # 5. CHIRPS
    if skip_chirps:
        log.info("CHIRPS: saltado (--skip-chirps)")
        chirps = pd.DataFrame()
    else:
        log.info("CHIRPS: descargando/cacheando %d departamentos...", len(centroides))
        chirps = load_chirps_for(centroides)

    # 6. Merge
    panel = magyp[[
        "cultivo", "campania_inicio", "campania", "provincia", "departamento",
        "sup_sembrada_ha", "sup_cosechada_ha", "produccion_tn", "rinde_kgha", "region"
    ]].copy()

    panel = panel.merge(centroides[["departamento","lat","lon"]], on="departamento", how="left")
    panel = panel.merge(oni, on="campania_inicio", how="left")

    if len(nasa):
        nasa["departamento"] = nasa["departamento"].apply(normalizar)
        panel = panel.merge(nasa, on=["departamento","campania_inicio"], how="left")
        log.info("NASA mergeado: %d columnas", len(panel.columns))

    if len(chirps):
        chirps["departamento"] = chirps["departamento"].apply(normalizar)
        panel = panel.merge(chirps, on=["departamento","campania_inicio"], how="left")
        log.info("CHIRPS mergeado: %d columnas", len(panel.columns))

    # 6b. Satelital: NDVI-AVHRR + ERA5-Land (GEE). Se descartan las filas sin
    # cobertura satelital (panel unificado sin NaN en NDVI/ERA5).
    if not skip_satelital:
        ndvi = load_ndvi_avhrr()
        ndvi_cols = [c for c in ndvi.columns if c.startswith("ndvi_avhrr_")]
        panel = _merge_satelital(panel, ndvi, ndvi_cols)
        era5 = load_era5()
        panel = _merge_satelital(panel, era5, ERA5_COLS)
        sat_cols = ndvi_cols + [c for c in ERA5_COLS if c in panel.columns]
        if sat_cols:
            n0 = len(panel)
            panel = panel.dropna(subset=sat_cols).reset_index(drop=True)
            log.info("descarte filas sin cobertura satelital: %d -> %d", n0, len(panel))

    # 7. Reporte de faltantes (sin imputar)
    out_path = PROC / "panel_union.parquet"
    panel.to_parquet(out_path, index=False)

    log.info("\n%s", "=" * 60)
    log.info("PANEL UNION guardado: %s", out_path)
    log.info("  Shape: %s", panel.shape)
    log.info("  Departamentos: %d", panel["departamento"].nunique())
    log.info("  Núcleo: %d | Resto: %d",
             (panel["region"] == "nucleo").any() and panel[panel["region"]=="nucleo"]["departamento"].nunique(),
             panel[panel["region"]=="resto"]["departamento"].nunique())
    log.info("  Cultivos: %s", panel["cultivo"].unique().tolist())
    log.info("  Campañas: %d–%d", panel["campania_inicio"].min(), panel["campania_inicio"].max())

    # NaN report (columnas climáticas)
    clim_cols = [c for c in panel.columns if any(
        c.startswith(p) for p in ["t2m","prect","rh2m","allsky","ws2m","oni","chirps"]
    )]
    nan_pct = panel[clim_cols].isna().mean().sort_values(ascending=False)
    nan_alto = nan_pct[nan_pct > 0.05]
    if len(nan_alto):
        log.warning("Columnas con >5%% NaN:\n%s", nan_alto.to_string())
    else:
        log.info("NaN climáticos: todos <= 5%%")

    return panel


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build panel union (sin restricción núcleo)")
    ap.add_argument("--min-campanas", type=int, default=20,
                    help="Mínimo de campañas con rinde válido por (depto, prov, cultivo) [default: 20]")
    ap.add_argument("--skip-chirps", action="store_true",
                    help="Saltar descarga de CHIRPS (útil si EE no está configurado)")
    ap.add_argument("--skip-satelital", action="store_true",
                    help="Saltar NDVI-AVHRR + ERA5-Land (GEE). El panel queda sin esas features")
    args = ap.parse_args()

    panel = build_panel_union(
        min_campanas=args.min_campanas,
        skip_chirps=args.skip_chirps,
        skip_satelital=args.skip_satelital,
    )
    print(f"\nPanel guardado: data/processed/panel_union.parquet  ({panel.shape})")
