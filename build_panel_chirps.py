"""
ETL — Sistema Integrado de Pronóstico y Detección de Anomalías en Campañas Agrícolas
build_panel_chirps.py: variante de build_panel.py que AGREGA CHIRPS como nueva fuente
de precipitación (vía Google Earth Engine), sin reemplazar ni modificar ninguna fuente
existente. Produce:
    - data/processed/chirps_diario.parquet   (fuente CRUDA: departamento × fecha, precip mm/día)
    - data/processed/panel_chirps.parquet     (agregado depto × campaña: chirps_precip_sep…mar)
    - data/processed/panel_nucleo.parquet     (REGENERADO: todas las fuentes + columnas chirps_*)

Diseño SOURCE-ONLY: este script solo mergea fuentes crudas y agrega la precip mensual de
CHIRPS al mismo nivel que NASA POWER. Nada de SPI, z-scores, días secos, lags → eso vive en
los notebooks. La precip de NASA POWER (prectotcorr_*) y la de CHIRPS (chirps_precip_*)
CONVIVEN en el panel; la redundancia se resuelve en el notebook de modelado, no acá.

Uso:
    python build_panel_chirps.py

Prerequisitos:
    - Los mismos que build_panel.py (CSV de MAGyP en data/raw/).
    - earthengine-api instalado y autenticado (`earthengine authenticate`), proyecto cloud
      con la Earth Engine API habilitada. Configurar EE_PROJECT abajo.

NOTA: build_panel.py y validate_panel.py originales NO se modifican. Este archivo es una
copia extendida que preserva el approach previo como registro.
"""

import os
import io
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

for d in [RAW, PROC]:
    d.mkdir(parents=True, exist_ok=True)

# Región núcleo según definición BCR
REGION_NUCLEO = {
    "SANTA FE": [
        # FIX: "Constitucion" y "Villa Constitucion" son el mismo departamento.
        # MAGyP lo reporta como "Villa Constitucion" → usamos ese nombre canónico.
        # "Constitucion" se elimina para evitar duplicados en el merge.
        "Villa Constitucion", "General Lopez", "Rosario",
        "San Lorenzo", "Iriondo", "Caseros", "Belgrano",
    ],
    "CORDOBA": ["Marcos Juarez", "Union", "Juarez Celman", "General San Martin"],
    "BUENOS AIRES": [
        "Pergamino", "Colon", "Rojas", "Salto", "San Nicolas", "Ramallo",
        "San Pedro", "Baradero", "Arrecifes", "Capitan Sarmiento",
        "Carmen de Areco", "Chacabuco", "Junin", "General Arenales",
        "Leandro N. Alem",
    ],
}

# Centroides departamentales (lat, lon) para NASA POWER  ── reutilizados TAL CUAL por CHIRPS
CENTROIDES = {
    # Santa Fe
    "Caseros":               (-33.03, -61.47),
    # FIX: unificado como "Villa Constitucion" (nombre MAGyP); "Constitucion" eliminado
    "Villa Constitucion":    (-33.67, -60.33),
    "General Lopez":         (-34.17, -61.88),
    "Rosario":               (-33.02, -60.63),
    "San Lorenzo":           (-32.75, -60.73),
    "Iriondo":               (-32.93, -61.23),
    "Belgrano":              (-32.57, -61.57),
    # Córdoba
    "Marcos Juarez":         (-33.03, -62.10),
    "Union":                 (-33.13, -62.87),
    "Juarez Celman":         (-33.40, -63.42),
    "General San Martin":    (-32.72, -62.98),
    # Buenos Aires
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

# Split temporal: definido en los notebooks de modelado, no en el ETL.

# FIX: extendido de 2024 a 2025 para cubrir la campaña 2024/25 completa
NASA_END_YEAR = 2025

NASA_PARAMS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M"

# ── CHIRPS (Google Earth Engine) ───────────────────────────────
EE_PROJECT     = "tp-final-ml-499915"          # ID del proyecto cloud (configurable)
CHIRPS_DATASET = "UCSB-CHG/CHIRPS/DAILY"        # CHIRPS v2.0, precip diaria, 0.05°
CHIRPS_BAND    = "precipitation"                # mm/día
CHIRPS_START   = "1981-01-01"                   # CHIRPS empieza en 1981
CHIRPS_END     = "2025-07-01"                   # fin (exclusivo) del rango a descargar
CHIRPS_SCALE   = None                           # None → escala nativa (≈5566 m); fallback abajo
CHIRPS_SCALE_FALLBACK = 5566                     # m, ~0.05° en el ecuador
CHIRPS_CACHE_TOL_DAYS = 40                       # latencia típica del producto final CHIRPS
GEE_MAX_RETRIES = 5
GEE_BACKOFF_BASE = 4                             # segundos (backoff exponencial)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def campaign_year(campania_str):
    """'2017/18' → 2017 (año de inicio)."""
    return int(str(campania_str).split("/")[0])


def normalizar_nombre(s):
    """Normalización básica de nombres de departamentos para joins."""
    import unicodedata
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = " ".join(s.split())
    return s.title()


def mes_a_campania_inicio(mes, anio):
    """Asignación de mes calendario a campaña agrícola (idéntica a NASA POWER):
    Sep–Dic del año Y → campaña Y ; Ene–Mar del año Y → campaña Y-1 ; resto → None."""
    if mes >= 9:
        return anio
    elif mes <= 3:
        return anio - 1
    return None


# ──────────────────────────────────────────────
# FUENTE 1: ONI
# ──────────────────────────────────────────────

def load_oni():
    """
    Descarga ONI mensual desde GitHub (ahuang11/oni).
    Devuelve el valor de anomalía crudo por mes de campaña: oni_oct … oni_feb.
    """
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
        if row["mes"] >= 10:
            return row["year"]
        elif row["mes"] <= 2:
            return row["year"] - 1
        else:
            return None

    df["campania_inicio"] = df.apply(assign_campaign, axis=1)
    df = df.dropna(subset=["campania_inicio"])
    df["campania_inicio"] = df["campania_inicio"].astype(int)

    oct_feb = df[df["mes"].isin([10, 11, 12, 1, 2])].copy()
    MES_NAMES_ONI = {10: "oct", 11: "nov", 12: "dic", 1: "ene", 2: "feb"}
    oct_feb["mes_str"] = oct_feb["mes"].map(MES_NAMES_ONI)

    camp_oni = oct_feb.pivot_table(
        index="campania_inicio",
        columns="mes_str",
        values="anom",
        aggfunc="first",
    ).reset_index()
    camp_oni.columns.name = None
    camp_oni = camp_oni.rename(columns={
        "oct": "oni_oct", "nov": "oni_nov", "dic": "oni_dic",
        "ene": "oni_ene", "feb": "oni_feb",
    })

    log.info("ONI: %d campañas procesadas (%d–%d)",
             len(camp_oni), camp_oni["campania_inicio"].min(),
             camp_oni["campania_inicio"].max())
    camp_oni.to_parquet(out, index=False)
    return camp_oni


# ──────────────────────────────────────────────
# FUENTE 2: MAGyP
# ──────────────────────────────────────────────

def _detect_and_read_csv(fpath):
    """Auto-detecta sep y encoding del CSV de MAGyP."""
    for sep in [",", ";", "\t"]:
        for enc in ["utf-8-sig", "latin1", "utf-8"]:
            try:
                df = pd.read_csv(fpath, sep=sep, encoding=enc, nrows=3, dtype=str)
                if len(df.columns) >= 4:
                    df = pd.read_csv(fpath, sep=sep, encoding=enc, dtype=str)
                    df.columns = [c.strip() for c in df.columns]
                    return df, sep, enc
            except Exception:
                pass
    raise ValueError(f"No se pudo parsear {fpath}")


def load_magyp():
    """
    Lee los CSV de MAGyP descargados desde datos.magyp.gob.ar
    """
    out = RAW / "magyp_panel.parquet"
    if out.exists():
        log.info("MAGyP: usando caché %s", out)
        return pd.read_parquet(out)

    frames = []
    for cultivo, fname in [("soja", "magyp_soja.csv"), ("maiz", "magyp_maiz.csv")]:
        fpath = RAW / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"Falta el archivo: {fpath}\n"
                f"Descargalo desde:\n"
                f"  Soja: https://datos.magyp.gob.ar/dataset/soja-siembra-cosecha-produccion-rendimiento\n"
                f"  Maíz: https://datos.magyp.gob.ar/dataset/maiz-siembra-cosecha-rendimiento-produccion\n"
                f"  Descargá el CSV de la serie histórica departamental.\n"
                f"{'='*60}"
            )
        log.info("MAGyP: leyendo %s...", fpath.name)
        df, sep, enc = _detect_and_read_csv(fpath)
        log.info("  Detectado: sep='%s' enc=%s | columnas: %s", sep, enc, df.columns.tolist())

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

        if "provincia" not in df.columns or "departamento" not in df.columns:
            raise ValueError(
                f"El CSV {fname} no tiene columnas provincia/departamento.\n"
                f"Columnas: {df.columns.tolist()}"
            )

        df["cultivo"] = cultivo
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    for col in ["sup_sembrada_ha", "sup_cosechada_ha", "produccion_tn", "rinde_kgha"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["provincia"]     = df["provincia"].apply(normalizar_nombre).str.upper()
    df["departamento"]  = df["departamento"].apply(normalizar_nombre)
    df["campania_inicio"] = df["campania"].apply(campaign_year)

    # FIX: MAGyP usa "Villa Constitucion" como nombre del depto.
    df["departamento"] = df["departamento"].replace({
        "Constitucion": "Villa Constitucion",
    })

    # Filtrar región núcleo
    mask = pd.Series(False, index=df.index)
    for prov, deptos in REGION_NUCLEO.items():
        deptos_norm = [normalizar_nombre(d) for d in deptos]
        m = (df["provincia"] == prov) & (df["departamento"].isin(deptos_norm))
        mask |= m
    df = df[mask].copy()

    df = df[df["campania_inicio"] >= 1981].copy()
    df = df[df["rinde_kgha"].notna() & (df["rinde_kgha"] > 0)].copy()

    log.info(
        "MAGyP: %d filas región núcleo | %s cultivos | campañas %d–%d",
        len(df),
        df["cultivo"].unique().tolist(),
        df["campania_inicio"].min(),
        df["campania_inicio"].max(),
    )

    df.to_parquet(out, index=False)
    return df


# ──────────────────────────────────────────────
# FUENTE 3: NASA POWER
# ──────────────────────────────────────────────

def fetch_power_departamento(depto, lat, lon, start_year=1981, end_year=NASA_END_YEAR):
    """
    Descarga datos diarios de NASA POWER para un punto (lat, lon).
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={NASA_PARAMS}"
        f"&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_year}0101&end={end_year}1231"
        f"&format=JSON"
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


def load_nasa_power():
    """
    Descarga NASA POWER para todos los departamentos de la región núcleo.
    Cachea por departamento. Agrega features mensuales por campaña.
    """
    out_panel = RAW / "nasa_power_campana_wide.parquet"
    if out_panel.exists():
        log.info("NASA POWER: usando caché %s", out_panel)
        return pd.read_parquet(out_panel)

    log.info("NASA POWER: descargando %d departamentos (hasta %d)...",
             len(CENTROIDES), NASA_END_YEAR)
    all_daily = []

    for depto, (lat, lon) in CENTROIDES.items():
        cache_file = RAW / f"nasa_power_{depto.lower().replace(' ', '_')}.parquet"
        if cache_file.exists():
            df_d = pd.read_parquet(cache_file)
            max_year = df_d["fecha"].dt.year.max() if len(df_d) else 0
            if max_year < NASA_END_YEAR:
                log.info("  [re-descargando] %s (caché hasta %d, necesitamos %d)",
                         depto, max_year, NASA_END_YEAR)
                cache_file.unlink()
            else:
                log.info("  [caché] %s (hasta %d)", depto, max_year)
                all_daily.append(df_d)
                continue

        log.info("  [descargando] %s (%.2f, %.2f)", depto, lat, lon)
        df_d = fetch_power_departamento(depto, lat, lon, end_year=NASA_END_YEAR)
        if df_d is None:
            log.warning("  Saltando %s por error", depto)
            continue
        df_d.to_parquet(cache_file, index=False)
        all_daily.append(df_d)
        time.sleep(1)

    if not all_daily:
        log.warning("NASA POWER: ningún departamento descargado.")
        pd.DataFrame().to_parquet(out_panel, index=False)
        return None

    df_daily = pd.concat(all_daily, ignore_index=True)
    df_daily["anio"] = df_daily["fecha"].dt.year
    df_daily["mes"]  = df_daily["fecha"].dt.month

    df_daily["campania_inicio"] = df_daily.apply(
        lambda row: mes_a_campania_inicio(row["mes"], row["anio"]), axis=1
    )

    clim_cols = [c for c in df_daily.columns
                 if c not in ["fecha", "departamento", "anio", "mes", "campania_inicio"]]

    meses_ae = [9, 10, 11, 12, 1, 2, 3]
    df_ae = df_daily[df_daily["mes"].isin(meses_ae)].copy()

    monthly = (
        df_ae
        .groupby(["departamento", "campania_inicio", "mes"])[clim_cols]
        .mean()
        .reset_index()
    )

    MES_NAMES = {9:"sep", 10:"oct", 11:"nov", 12:"dic", 1:"ene", 2:"feb", 3:"mar"}
    monthly["mes_str"] = monthly["mes"].map(MES_NAMES)

    monthly_wide = monthly.pivot_table(
        index=["departamento", "campania_inicio"],
        columns="mes_str",
        values=clim_cols,
        aggfunc="mean"
    )
    monthly_wide.columns = [f"{v}_{m}" for v, m in monthly_wide.columns]
    monthly_wide = monthly_wide.reset_index()

    nasa_panel = monthly_wide

    log.info(
        "NASA POWER: %d filas | %d columnas | campañas %d–%d",
        len(nasa_panel),
        len(nasa_panel.columns),
        nasa_panel["campania_inicio"].min(),
        nasa_panel["campania_inicio"].max(),
    )
    nasa_panel.to_parquet(out_panel, index=False)
    return nasa_panel


# ──────────────────────────────────────────────
# FUENTE 4: NDVI (HDX/WFP)
# ──────────────────────────────────────────────

NUCLEO_PCODE_MAP = {
    # Santa Fe (AR082)
    "AR082028": "Belgrano",
    "AR082035": "Caseros",
    # FIX: AR082042 es "Constitución" en INDEC → nombre canónico "Villa Constitucion".
    "AR082042": "Villa Constitucion",
    "AR082049": "General Lopez",
    "AR082056": "Iriondo",
    "AR082070": "Rosario",
    "AR082077": "San Lorenzo",
    # Córdoba (AR014)
    "AR014021": "General San Martin",
    "AR014042": "Juarez Celman",
    "AR014049": "Marcos Juarez",
    "AR014098": "Union",
    # Buenos Aires (AR006)
    "AR006014": "Arrecifes",
    "AR006021": "Baradero",
    "AR006056": "Capitan Sarmiento",
    "AR006063": "Carmen De Areco",
    "AR006070": "Chacabuco",
    "AR006098": "Colon",
    "AR006147": "General Arenales",
    "AR006336": "Junin",
    # FIX: código INDEC de Leandro N. Alem es 371, no 372.
    "AR006371": "Leandro N. Alem",
    "AR006595": "Pergamino",
    "AR006630": "Ramallo",
    "AR006638": "Rojas",
    "AR006648": "Salto",
    "AR006658": "San Nicolas",
    "AR006672": "San Pedro",
}


def load_ndvi():
    """
    Descarga NDVI dekadal admin2 desde HDX. Filtra región núcleo. Agrega por campaña.
    """
    out = RAW / "ndvi_subnacional.parquet"
    if out.exists():
        log.info("NDVI: usando caché %s", out)
        return pd.read_parquet(out)

    log.info("NDVI: intentando cargar desde archivo local o HDX...")
    ndvi_local = RAW / "ndvi_arg_adm2.csv"
    if ndvi_local.exists():
        log.info("NDVI: leyendo desde archivo local %s", ndvi_local)
        r_content = ndvi_local.read_bytes()
        class FakeResponse:
            status_code = 200
            text = r_content.decode("utf-8", errors="replace")
        r = FakeResponse()
    else:
        url = "https://data.humdata.org/dataset/arg-ndvi-subnational/resource/download/arg-ndvi-adm2-full.csv"
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})

    if r.status_code != 200:
        log.warning("NDVI: no se pudo descargar (status %d). Continuando sin NDVI.", r.status_code)
        return None

    df = pd.read_csv(io.StringIO(r.text))
    log.info("NDVI raw: %d filas, columnas: %s", len(df), df.columns.tolist()[:10])

    df.columns = [c.lower().strip() for c in df.columns]

    date_col  = next((c for c in df.columns if "date" in c), None)
    pcode_col = next((c for c in df.columns if "pcode" in c), None)
    vim_col   = next((c for c in df.columns if c in ("vim", "vim_avg")), None)
    viq_col   = next((c for c in df.columns if c == "viq"), None)

    log.info("NDVI cols detectadas: date=%s pcode=%s vim=%s viq=%s",
             date_col, pcode_col, vim_col, viq_col)
    if not all([date_col, pcode_col, vim_col]):
        log.error("NDVI: columnas inesperadas %s", df.columns.tolist())
        return None

    df["fecha"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["fecha"])

    if "adm_level" in df.columns:
        df = df[df["adm_level"] == 2].copy()

    df[pcode_col] = df[pcode_col].str.strip().str.upper()
    df["departamento"] = df[pcode_col].map(NUCLEO_PCODE_MAP)

    pcodes_in_csv = set(df[pcode_col].unique())
    pcodes_nucleo = set(NUCLEO_PCODE_MAP.keys())
    encontrados   = pcodes_nucleo & pcodes_in_csv
    faltantes     = pcodes_nucleo - pcodes_in_csv
    log.info("NDVI: PCODEs núcleo encontrados en CSV: %d/%d", len(encontrados), len(pcodes_nucleo))
    if faltantes:
        log.warning("NDVI: PCODEs no encontrados → %s",
                    {k: NUCLEO_PCODE_MAP[k] for k in sorted(faltantes)})

    df = df[df["departamento"].notna()].copy()
    log.info("NDVI: %d filas después de filtrar núcleo", len(df))

    df["mes"]  = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year

    df["campania_inicio"] = df.apply(
        lambda r: r["anio"] if r["mes"] >= 10 else (r["anio"] - 1 if r["mes"] <= 3 else None),
        axis=1
    )
    df = df.dropna(subset=["campania_inicio"])
    df["campania_inicio"] = df["campania_inicio"].astype(int)

    df = df[df["mes"].isin([10, 11, 12, 1, 2, 3])].copy()

    agg_dict = {vim_col: ["mean", "min", "max"]}
    if viq_col:
        agg_dict[viq_col] = "mean"

    ndvi_camp = (
        df.groupby(["departamento", "campania_inicio"])
        .agg(agg_dict)
        .reset_index()
    )
    ndvi_camp.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in ndvi_camp.columns
    ]
    ndvi_camp = ndvi_camp.rename(columns={
        f"{vim_col}_mean": "ndvi_mean",
        f"{vim_col}_min":  "ndvi_min",
        f"{vim_col}_max":  "ndvi_max",
        **(
            {f"{viq_col}_mean": "ndvi_anomalia_pct"} if viq_col else {}
        ),
    })

    log.info(
        "NDVI: %d filas | campañas %d–%d",
        len(ndvi_camp),
        ndvi_camp["campania_inicio"].min(),
        ndvi_camp["campania_inicio"].max(),
    )
    ndvi_camp.to_parquet(out, index=False)
    return ndvi_camp


# ──────────────────────────────────────────────
# FUENTE 5: CHIRPS (Google Earth Engine)  ── NUEVA
# ──────────────────────────────────────────────

def _ee_init():
    """Inicializa Earth Engine con el proyecto configurado. Import perezoso."""
    import ee
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as e:
        log.warning("EE.Initialize falló (%s). Intentando autenticar...", e)
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    return ee


def _chirps_points(ee):
    """FeatureCollection con los 26 centroides (mismos puntos que NASA POWER)."""
    feats = [
        ee.Feature(ee.Geometry.Point([lon, lat]), {"departamento": depto})
        for depto, (lat, lon) in CENTROIDES.items()
    ]
    return ee.FeatureCollection(feats)


def _chirps_scale(ee):
    """Escala NATIVA de CHIRPS (m). Punto: muestrear al pixel exacto del centroide."""
    if CHIRPS_SCALE is not None:
        return CHIRPS_SCALE
    try:
        img = (ee.ImageCollection(CHIRPS_DATASET).select(CHIRPS_BAND)
               .filterDate("2020-01-01", "2020-01-02").first())
        scale = img.projection().nominalScale().getInfo()
        log.info("CHIRPS: escala nativa detectada = %.1f m", scale)
        return float(scale)
    except Exception as e:
        log.warning("CHIRPS: no se pudo obtener escala nativa (%s); usando fallback %s m",
                    e, CHIRPS_SCALE_FALLBACK)
        return float(CHIRPS_SCALE_FALLBACK)


# Marcadores de errores ESTRUCTURALES de GEE: el backoff no sirve porque el límite
# es duro, no transitorio. Si aparecen, hay que reducir el tamaño del chunk, no reintentar.
_GEE_STRUCTURAL_MARKERS = (
    "over 5000 elements",
    "collection query aborted",
)


def _es_error_estructural(err):
    msg = str(err).lower()
    return any(marker in msg for marker in _GEE_STRUCTURAL_MARKERS)


def _gee_getinfo_retry(obj, descripcion=""):
    """getInfo() con reintentos y backoff exponencial SOLO para errores TRANSITORIOS
    (cuota, timeout). Los errores ESTRUCTURALES (p.ej. "over 5000 elements") abortan
    de inmediato: reintentar es inútil porque el límite por llamada es duro."""
    last_err = None
    for intento in range(1, GEE_MAX_RETRIES + 1):
        try:
            return obj.getInfo()
        except Exception as e:
            if _es_error_estructural(e):
                raise RuntimeError(
                    f"GEE getInfo {descripcion}: error ESTRUCTURAL no recuperable "
                    f"({e}). El chunk excede el límite duro de elementos por llamada; "
                    f"reducí el rango (procesá por mes en vez de por año). No reintento."
                ) from e
            last_err = e
            espera = GEE_BACKOFF_BASE * (2 ** (intento - 1))
            log.warning("GEE getInfo %s falló (intento %d/%d): %s → backoff %ds",
                        descripcion, intento, GEE_MAX_RETRIES, e, espera)
            time.sleep(espera)
    raise RuntimeError(f"GEE getInfo agotó reintentos para {descripcion}: {last_err}")


def fetch_chirps_range(ee, points, scale, start, end):
    """
    Descarga la precip diaria de CHIRPS para los 26 puntos en el rango [start, end).
    Pensada para chunks de UN MES: con ~30 días × 26 puntos ≈ 780 features por getInfo,
    muy por debajo del límite duro de 5000 elementos por llamada (un año entero —
    26 × 365 ≈ 9490 — lo excede y aborta). Devuelve DataFrame largo: departamento,
    fecha, precip_mm. Muestreo a escala NATIVA → valor exacto del pixel que contiene
    cada centroide (point query).
    """
    col = (ee.ImageCollection(CHIRPS_DATASET)
           .select(CHIRPS_BAND)
           .filterDate(start, end))

    def per_image(img):
        fecha = img.date().format("YYYY-MM-dd")
        # Reducer.first() sobre puntos a escala nativa = valor del pixel del centroide
        fc = img.reduceRegions(collection=points, reducer=ee.Reducer.first(), scale=scale)
        return fc.map(lambda f: f.set("fecha", fecha))

    flat = ee.FeatureCollection(col.map(per_image)).flatten()
    info = _gee_getinfo_retry(flat, descripcion=f"CHIRPS {start}..{end}")

    rows = []
    for feat in info["features"]:
        p = feat["properties"]
        rows.append({
            "departamento": p.get("departamento"),
            "fecha": p.get("fecha"),
            # Reducer.first() nombra la propiedad como la banda ('precipitation');
            # algunas versiones la nombran 'first'. Contemplamos ambas.
            "precip_mm": p.get(CHIRPS_BAND, p.get("first", np.nan)),
        })
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    return df


def load_chirps_daily():
    """
    OUTPUT 1 — chirps_diario.parquet (fuente CRUDA: una fila por departamento × fecha).
    Sin agregar ni derivar. Cache incremental: si ya cubre hasta la fecha final, no
    re-descarga; si está stale, completa desde el último año cacheado.
    """
    out = PROC / "chirps_diario.parquet"
    desired_last = pd.Timestamp(CHIRPS_END) - pd.Timedelta(days=1)
    tol = pd.Timedelta(days=CHIRPS_CACHE_TOL_DAYS)
    fin = pd.Timestamp(CHIRPS_END)

    existing = None
    # Primer MES a descargar (no año): por defecto, el mes de inicio de CHIRPS.
    resume_from = pd.Timestamp(CHIRPS_START).replace(day=1)
    if out.exists():
        existing = pd.read_parquet(out)
        existing["fecha"] = pd.to_datetime(existing["fecha"])
        last = existing["fecha"].max()
        if last >= desired_last - tol:
            log.info("CHIRPS diario: caché cubre hasta %s (≥ %s) → no re-descargo",
                     last.date(), (desired_last - tol).date())
            return existing
        # Re-baja desde el inicio del MES del último dato (ese mes puede estar incompleto).
        resume_from = last.replace(day=1)
        log.info("CHIRPS diario: caché hasta %s; completando desde %s",
                 last.date(), resume_from.date())

    ee = _ee_init()
    points = _chirps_points(ee)
    scale = _chirps_scale(ee)

    frames = []
    if existing is not None:
        # Conservamos lo anterior al mes que vamos a re-descargar.
        frames.append(existing[existing["fecha"] < resume_from])

    # Iterar por MES: cada llamada cubre ~26 puntos × ~30 días ≈ 780 elementos, muy
    # por debajo del límite duro de 5000 de getInfo. Concatenamos todos los meses.
    periodo = max(pd.Timestamp(CHIRPS_START), resume_from).replace(day=1)
    while periodo < fin:
        siguiente = periodo + pd.offsets.MonthBegin(1)
        start = max(pd.Timestamp(CHIRPS_START), periodo)
        end   = min(fin, siguiente)
        if start >= end:
            periodo = siguiente
            continue
        log.info("CHIRPS: descargando %s .. %s", start.date(), end.date())
        df_m = fetch_chirps_range(ee, points, scale,
                                  start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        log.info("  %d filas (deptos×días)", len(df_m))
        frames.append(df_m)
        time.sleep(1)  # cuota amigable
        periodo = siguiente

    daily = (pd.concat(frames, ignore_index=True)
             .dropna(subset=["fecha", "departamento"])
             .drop_duplicates(["departamento", "fecha"])
             .sort_values(["departamento", "fecha"])
             .reset_index(drop=True))

    log.info("CHIRPS diario: %d filas | %d deptos | %s .. %s",
             len(daily), daily["departamento"].nunique(),
             daily["fecha"].min().date(), daily["fecha"].max().date())
    daily.to_parquet(out, index=False)
    return daily


def load_chirps():
    """
    OUTPUT 2 — panel_chirps.parquet (agregado depto × campaña).
    Precip mensual media Sep–Mar con el mismo naming pattern que NASA POWER:
    chirps_precip_sep … chirps_precip_mar. Solo agregado mensual crudo (SPI/días secos
    NO van acá → notebooks).
    """
    out = PROC / "panel_chirps.parquet"
    daily = load_chirps_daily()

    daily = daily.copy()
    daily["anio"] = daily["fecha"].dt.year
    daily["mes"]  = daily["fecha"].dt.month
    daily["campania_inicio"] = daily.apply(
        lambda r: mes_a_campania_inicio(r["mes"], r["anio"]), axis=1
    )

    meses_ae = [9, 10, 11, 12, 1, 2, 3]
    df_ae = daily[daily["mes"].isin(meses_ae)].copy()

    monthly = (
        df_ae
        .groupby(["departamento", "campania_inicio", "mes"])["precip_mm"]
        .mean()
        .reset_index()
    )

    MES_NAMES = {9: "sep", 10: "oct", 11: "nov", 12: "dic", 1: "ene", 2: "feb", 3: "mar"}
    monthly["mes_str"] = monthly["mes"].map(MES_NAMES)

    wide = monthly.pivot_table(
        index=["departamento", "campania_inicio"],
        columns="mes_str",
        values="precip_mm",
        aggfunc="mean",
    )
    wide.columns = [f"chirps_precip_{m}" for m in wide.columns]
    wide = wide.reset_index()
    # Ordenar columnas mensuales sep→mar para consistencia
    orden = ["departamento", "campania_inicio"] + \
            [f"chirps_precip_{m}" for m in ["sep", "oct", "nov", "dic", "ene", "feb", "mar"]
             if f"chirps_precip_{m}" in wide.columns]
    wide = wide[orden]

    wide["campania_inicio"] = wide["campania_inicio"].astype(int)
    log.info("CHIRPS panel: %d filas | %d columnas | campañas %d–%d",
             len(wide), len(wide.columns),
             wide["campania_inicio"].min(), wide["campania_inicio"].max())
    wide.to_parquet(out, index=False)
    return wide


# ──────────────────────────────────────────────
# ENSAMBLE DEL PANEL
# ──────────────────────────────────────────────

def build_panel():
    log.info("=" * 60)
    log.info("BUILD PANEL NUCLEO (+ CHIRPS) — iniciando")
    log.info("=" * 60)

    oni    = load_oni()
    magyp  = load_magyp()
    nasa   = load_nasa_power()
    ndvi   = load_ndvi()
    chirps = load_chirps()

    panel = magyp[[
        "cultivo", "campania_inicio", "provincia", "departamento",
        "sup_sembrada_ha", "sup_cosechada_ha", "produccion_tn", "rinde_kgha"
    ]].copy()

    centroides_df = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in CENTROIDES.items()],
        columns=["departamento", "lat_centroide", "lon_centroide"]
    )
    centroides_df["departamento"] = centroides_df["departamento"].apply(normalizar_nombre)
    panel = panel.merge(centroides_df, on="departamento", how="left")

    panel["campania"] = panel["campania_inicio"].apply(
        lambda y: f"{y}/{str(y+1)[-2:]}"
    )

    panel = panel.merge(oni, on="campania_inicio", how="left")

    if nasa is not None:
        nasa["departamento"] = nasa["departamento"].apply(normalizar_nombre)
        panel = panel.merge(nasa, on=["departamento", "campania_inicio"], how="left")
        log.info("NASA POWER mergeado: %d columnas totales", len(panel.columns))

    if ndvi is not None:
        ndvi["departamento"] = ndvi["departamento"].apply(normalizar_nombre)
        panel = panel.merge(ndvi, on=["departamento", "campania_inicio"], how="left")
        log.info("NDVI mergeado")

    # ── CHIRPS: merge al mismo nivel (depto × campaña). Convive con prectotcorr_* de NASA.
    if chirps is not None:
        n_before = len(panel)
        chirps = chirps.copy()
        chirps["departamento"] = chirps["departamento"].apply(normalizar_nombre)
        panel = panel.merge(chirps, on=["departamento", "campania_inicio"], how="left")
        assert len(panel) == n_before, "CHIRPS alteró el número de filas del panel"
        log.info("CHIRPS mergeado: %d columnas totales (sin cambiar filas)", len(panel.columns))

    out_panel = PROC / "panel_nucleo.parquet"
    panel.to_parquet(out_panel, index=False)
    log.info("\n✓ Panel guardado: %s", out_panel)
    log.info("  Shape: %s", panel.shape)
    log.info("  Columnas: %d", len(panel.columns))
    log.info("  Campañas: %d–%d", panel["campania_inicio"].min(), panel["campania_inicio"].max())
    log.info("  Departamentos: %d", panel["departamento"].nunique())
    log.info("  Cultivos: %s", panel["cultivo"].unique().tolist())

    # Parquets por fuente (para EDA y reproducibilidad)
    magyp[[
        "cultivo", "campania_inicio", "provincia", "departamento",
        "sup_sembrada_ha", "sup_cosechada_ha", "produccion_tn", "rinde_kgha"
    ]].to_parquet(PROC / "fuente_magyp.parquet", index=False)
    oni.to_parquet(PROC / "fuente_oni.parquet", index=False)
    if nasa is not None:
        nasa.to_parquet(PROC / "fuente_nasa_power.parquet", index=False)
    if ndvi is not None:
        ndvi.to_parquet(PROC / "fuente_ndvi.parquet", index=False)
    if chirps is not None:
        chirps.to_parquet(PROC / "fuente_chirps.parquet", index=False)

    log.info("Parquets por fuente guardados en %s", PROC)
    return panel


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    panel = build_panel()
    print("\n=== RESUMEN DEL PANEL (+ CHIRPS) ===")
    chirps_cols = [c for c in panel.columns if c.startswith("chirps_")]
    print("Columnas CHIRPS:", chirps_cols)
    print(panel[["cultivo", "campania_inicio", "departamento", "rinde_kgha"] + chirps_cols]
          .describe(include="all"))
