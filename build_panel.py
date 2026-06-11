"""
ETL — Sistema Integrado de Pronóstico y Detección de Anomalías en Campañas Agrícolas
build_panel.py: descarga y procesa las 4 fuentes, produce data/processed/panel_nucleo.parquet

Uso:
    python src/etl/build_panel.py

Prerequisitos manuales (dos descargas que no se pueden automatizar):
    1. MAGyP Soja: ir a https://datosestimaciones.magyp.gob.ar/reportes.php?reporte=Estimaciones
       Seleccionar: Cultivo=Soja, Nivel=Departamento, todas las campañas → Descargar CSV
       Guardar como: data/raw/magyp_soja.csv
    2. MAGyP Maíz: ídem con Cultivo=Maíz → data/raw/magyp_maiz.csv

Fuentes automáticas:
    - ONI: GitHub (ahuang11/oni)
    - NDVI: HDX/WFP (arg-ndvi-adm2-full.csv)
    - NASA POWER: API REST por centroide departamental
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
SPLITS = ROOT / "data" / "splits"

for d in [RAW, PROC, SPLITS]:
    d.mkdir(parents=True, exist_ok=True)

# Región núcleo según definición BCR
REGION_NUCLEO = {
    "SANTA FE": ["Caseros", "Constitucion", "Villa Constitucion", "General Lopez", "Rosario",
                 "San Lorenzo", "Iriondo", "Belgrano"],
    "CORDOBA":  ["Marcos Juarez", "Union", "Juarez Celman", "General San Martin"],
    "BUENOS AIRES": [
        "Pergamino", "Colon", "Rojas", "Salto", "San Nicolas", "Ramallo",
        "San Pedro", "Baradero", "Arrecifes", "Capitan Sarmiento",
        "Carmen de Areco", "Chacabuco", "Junin", "General Arenales",
        "Leandro N. Alem"
    ],
}

# Centroides departamentales (lat, lon) para NASA POWER
# Fuente: IGN Argentina / cálculo propio sobre shapefiles
CENTROIDES = {
    # Santa Fe
    "Caseros":               (-33.03, -61.47),
    "Constitucion":          (-33.67, -60.33),
    "Villa Constitucion":          (-33.67, -60.33),
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

# Split temporal (campañas = año de inicio)
TRAIN_END   = 2017  # 1981/82 – 2017/18
VAL_END     = 2020  # 2018/19 – 2020/21
# Test: 2021/22 – 2023/24

NASA_PARAMS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M"


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
    # colapsa espacios dobles
    s = " ".join(s.split())
    return s.title()


# ──────────────────────────────────────────────
# FUENTE 1: ONI
# ──────────────────────────────────────────────

def load_oni():
    """
    Descarga ONI mensual desde GitHub (ahuang11/oni).
    Calcula oni_oct_feb_mean por campaña y categoría Niña/Niño/Neutro.
    """
    out = RAW / "oni_mensual.parquet"
    if out.exists():
        log.info("ONI: usando caché %s", out)
        return pd.read_parquet(out)

    log.info("ONI: descargando...")
    url = "https://raw.githubusercontent.com/ahuang11/oni/master/oni.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    # Columnas: season, year, sst_c, anom_c, threshold, cumulative, ntotal, oni
    # 'season' es el trimestre centrado en formato DJF, JFM, etc.
    # Necesitamos mes central de cada ventana de 3 meses para mapear a campaña agrícola

    # Mapeamos season → mes central
    SEASON_TO_MONTH = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    df["mes"] = df["season"].map(SEASON_TO_MONTH)

    # Para NDJ: el año del registro es el año de N/D, el mes real es diciembre del año-1
    # Ej: NDJ 2018 → el diciembre es 2017, enero-febrero son 2018
    # Simplificamos: usamos year como está (season DJF 2018 = ene-feb 2018 = campaña 2017/18)

    df = df[df["year"] >= 1980].copy()
    df["anom"] = df["anom_c"]
    df["oni_cat"] = df["oni"].map(
        {"el_nino": "nino", "la_nina": "nina", "neutral": "neutral"}
    ).fillna("neutral")

    # Campaña agrícola Oct–Abr: oct año_inicio a feb año_inicio+1
    # Asignamos cada registro a una campaña:
    #   mes 10,11,12 del año Y → campaña Y
    #   mes 1,2 del año Y → campaña Y-1
    def assign_campaign(row):
        if row["mes"] >= 10:
            return row["year"]
        elif row["mes"] <= 2:
            return row["year"] - 1
        else:
            return None  # meses 3-9: no relevantes para Oct-Feb

    df["campania_inicio"] = df.apply(assign_campaign, axis=1)
    df = df.dropna(subset=["campania_inicio"])
    df["campania_inicio"] = df["campania_inicio"].astype(int)

    # Solo Oct–Feb (meses 10,11,12,1,2)
    oct_feb = df[df["mes"].isin([10, 11, 12, 1, 2])].copy()

    camp_oni = (
        oct_feb
        .groupby("campania_inicio")
        .agg(
            oni_oct_feb_mean=("anom", "mean"),
            oni_oct_feb_min=("anom", "min"),
        )
        .reset_index()
    )

    # Categoría de campaña: modo de oni_cat en Oct-Feb
    cat_mode = (
        oct_feb.groupby("campania_inicio")["oni_cat"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"oni_cat": "oni_categoria"})
    )
    camp_oni = camp_oni.merge(cat_mode, on="campania_inicio")

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
    Formato esperado (serie departamental):
        cultivo, anio, campania, provincia, provincia_id,
        departamento, departamento_id,
        superficie_sembrada_ha, superficie_cosechada_ha,
        produccion_tm, rendimiento_kgxha
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

        # Renombrar columnas al esquema estándar
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

    # Convertir numéricos
    for col in ["sup_sembrada_ha", "sup_cosechada_ha", "produccion_tn", "rinde_kgha"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalizar nombres
    df["provincia"]     = df["provincia"].apply(normalizar_nombre).str.upper()
    df["departamento"]  = df["departamento"].apply(normalizar_nombre)
    df["campania_inicio"] = df["campania"].apply(campaign_year)

    # Filtrar región núcleo
    mask = pd.Series(False, index=df.index)
    for prov, deptos in REGION_NUCLEO.items():
        deptos_norm = [normalizar_nombre(d) for d in deptos]
        m = (df["provincia"] == prov) & (df["departamento"].isin(deptos_norm))
        mask |= m
    df = df[mask].copy()

    # Filtrar rango temporal (1981/82 en adelante)
    df = df[df["campania_inicio"] >= 1981].copy()

    # Excluir filas sin rinde
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

def fetch_power_departamento(depto, lat, lon, start_year=1981, end_year=2024):
    """
    Descarga datos diarios de NASA POWER para un punto (lat, lon).
    Devuelve DataFrame diario.
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
    # params es dict: {PARAM: {YYYYMMDD: value}}
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
    out_panel = RAW / "nasa_power_monthly.parquet"
    if out_panel.exists():
        log.info("NASA POWER: usando caché %s", out_panel)
        return pd.read_parquet(out_panel)

    log.info("NASA POWER: descargando %d departamentos...", len(CENTROIDES))
    all_daily = []

    for depto, (lat, lon) in CENTROIDES.items():
        cache_file = RAW / f"nasa_power_{depto.lower().replace(' ', '_')}.parquet"
        if cache_file.exists():
            log.info("  [caché] %s", depto)
            df_d = pd.read_parquet(cache_file)
        else:
            log.info("  [descargando] %s (%.2f, %.2f)", depto, lat, lon)
            df_d = fetch_power_departamento(depto, lat, lon)
            if df_d is None:
                log.warning("  Saltando %s por error", depto)
                continue
            df_d.to_parquet(cache_file, index=False)
            time.sleep(1)  # rate limit amigable

        all_daily.append(df_d)

    if not all_daily:
        log.warning("NASA POWER: ningún departamento descargado. Continuando sin datos climáticos.")
        # Guardamos un parquet vacío para no re-intentar en ejecuciones futuras del mismo entorno
        pd.DataFrame().to_parquet(out_panel, index=False)
        return None

    df_daily = pd.concat(all_daily, ignore_index=True)
    df_daily["anio"] = df_daily["fecha"].dt.year
    df_daily["mes"]  = df_daily["fecha"].dt.month

    # Asignar campaña agrícola (Oct año Y → campaña Y, Nov-Dic año Y → campaña Y,
    # Ene-Feb-Mar año Y+1 → campaña Y)
    def mes_a_campania_inicio(row):
        m, y = row["mes"], row["anio"]
        if m >= 10:
            return y
        elif m <= 3:
            return y - 1
        else:
            return None  # abr-sep no entra en features principales

    df_daily["campania_inicio"] = df_daily.apply(mes_a_campania_inicio, axis=1)

    # Columnas climáticas que vienen de NASA POWER (en minúsculas)
    clim_cols = [c for c in df_daily.columns
                 if c not in ["fecha", "departamento", "anio", "mes", "campania_inicio"]]

    # Agregados mensuales Sep–Mar (para representación tabular del AE)
    meses_ae = [9, 10, 11, 12, 1, 2, 3]
    df_ae = df_daily[df_daily["mes"].isin(meses_ae)].copy()

    monthly = (
        df_ae
        .groupby(["departamento", "campania_inicio", "mes"])[clim_cols]
        .mean()
        .reset_index()
    )

    # Pivot: una columna por (variable, mes)
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

    # Features expertas adicionales sobre el año completo
    # Temperaturas extremas, GDD, precipitación por fases fenológicas
    camp_clim = df_daily[df_daily["campania_inicio"].notna()].copy()
    camp_clim["campania_inicio"] = camp_clim["campania_inicio"].astype(int)

    # GDD: base 10°C
    if "t2m" in clim_cols:
        camp_clim["gdd_daily"] = ((camp_clim["t2m"] - 10).clip(lower=0))
    if "t2m_max" in clim_cols:
        camp_clim["dias_t_mayor_35"] = (camp_clim["t2m_max"] > 35).astype(int)

    expert = (
        camp_clim
        .groupby(["departamento", "campania_inicio"])
        .agg(
            tmean=("t2m", "mean") if "t2m" in clim_cols else ("t2m", "first"),
            tmax_p95=("t2m_max", lambda x: x.quantile(0.95)) if "t2m_max" in clim_cols else ("t2m_max", "first"),
            precip_total=("prectotcorr", "sum") if "prectotcorr" in clim_cols else ("prectotcorr", "first"),
            gdd=("gdd_daily", "sum") if "gdd_daily" in camp_clim.columns else ("t2m", "first"),
            dias_t_mayor_35=("dias_t_mayor_35", "sum") if "dias_t_mayor_35" in camp_clim.columns else ("t2m", "first"),
            rad_solar_mean=("allsky_sfc_sw_dwn", "mean") if "allsky_sfc_sw_dwn" in clim_cols else ("t2m", "first"),
        )
        .reset_index()
    )

    # Precipitación por fases fenológicas (aproximadas para región núcleo)
    # Soja: siembra Oct-Nov, vegetativo Dic, floración/llenado Ene-Feb-Mar
    # Maíz: siembra Sep-Oct, vegetativo Nov-Dic, VT-R1 Ene, llenado Feb-Mar
    for fase, meses in [
        ("siembra", [9, 10, 11]),
        ("vegetativo", [12]),
        ("r1_r5", [1, 2]),
        ("llenado", [3]),
    ]:
        fase_df = (
            df_daily[df_daily["mes"].isin(meses) & df_daily["campania_inicio"].notna()]
            .groupby(["departamento", "campania_inicio"])["prectotcorr"]
            .sum()
            .reset_index()
            .rename(columns={"prectotcorr": f"precip_{fase}"})
        ) if "prectotcorr" in clim_cols else None
        if fase_df is not None:
            expert = expert.merge(fase_df, on=["departamento", "campania_inicio"], how="left")

    # Merge expert + monthly_wide
    nasa_panel = expert.merge(monthly_wide, on=["departamento", "campania_inicio"], how="left")

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

# PCODEs WFP → nombre departamento (núcleo)
# Fuente: INDEC codificación divisiones político-territoriales
NUCLEO_PCODE_MAP = {
    # Santa Fe (AR082)
    "AR082028": "Belgrano",        "AR082035": "Caseros",
    "AR082042": "Constitucion",    "AR082049": "General Lopez",
    "AR082056": "Iriondo",         "AR082070": "Rosario",
    "AR082077": "San Lorenzo",
    # Córdoba (AR014)
    "AR014021": "General San Martin", "AR014042": "Juarez Celman",
    "AR014049": "Marcos Juarez",      "AR014098": "Union",
    # Buenos Aires (AR006)
    "AR006014": "Arrecifes",       "AR006021": "Baradero",
    "AR006056": "Capitan Sarmiento","AR006063": "Carmen De Areco",
    "AR006070": "Chacabuco",       "AR006098": "Colon",
    "AR006147": "General Arenales","AR006336": "Junin",
    "AR006372": "Leandro N. Alem", "AR006595": "Pergamino",
    "AR006630": "Ramallo",         "AR006638": "Rojas",
    "AR006648": "Salto",           "AR006658": "San Nicolas",
    "AR006672": "San Pedro",
}


def load_ndvi():
    """
    Descarga NDVI dekadal admin2 desde HDX.
    Filtra departamentos de región núcleo.
    Agrega por campaña agrícola.
    """
    out = RAW / "ndvi_subnacional.parquet"
    if out.exists():
        log.info("NDVI: usando caché %s", out)
        return pd.read_parquet(out)

    log.info("NDVI: descargando desde HDX...")
    # Descarga manual: entrá a https://data.humdata.org/dataset/arg-ndvi-subnational
    # y descargá el archivo "arg-ndvi-adm2-full.csv" (serie completa admin2)
    # Guardalo como data/raw/ndvi_arg_adm2.csv y el script lo levanta automáticamente
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

    # Formato HDX WFP: date, adm_level, adm_id, PCODE, n_pixels, vim, vim_avg, viq
    df.columns = [c.lower().strip() for c in df.columns]

    # columnas ya están en lowercase por el paso anterior
    date_col  = next((c for c in df.columns if "date" in c), None)
    pcode_col = next((c for c in df.columns if "pcode" in c), None)
    vim_col   = next((c for c in df.columns if c in ("vim", "vim_avg")), None)
    viq_col   = next((c for c in df.columns if c == "viq"), None)

    log.info("NDVI cols detectadas: date=%s pcode=%s vim=%s viq=%s", date_col, pcode_col, vim_col, viq_col)
    if not all([date_col, pcode_col, vim_col]):
        log.error("NDVI: columnas inesperadas %s", df.columns.tolist())
        return None

    df["fecha"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["fecha"])

    # Filtrar admin2 y mapear PCODE → nombre departamento
    if "adm_level" in df.columns:
        df = df[df["adm_level"] == 2].copy()

    df[pcode_col] = df[pcode_col].str.strip().str.upper()
    df["departamento"] = df[pcode_col].map(NUCLEO_PCODE_MAP)
    df = df[df["departamento"].notna()].copy()
    log.info("NDVI: %d filas después de filtrar núcleo", len(df))

    df["mes"]  = df["fecha"].dt.month
    df["anio"] = df["fecha"].dt.year

    # Asignar campaña
    df["campania_inicio"] = df.apply(
        lambda r: r["anio"] if r["mes"] >= 10 else (r["anio"] - 1 if r["mes"] <= 3 else None),
        axis=1
    )
    df = df.dropna(subset=["campania_inicio"])
    df["campania_inicio"] = df["campania_inicio"].astype(int)

    # Meses relevantes: Oct-Mar (campaña activa)
    df = df[df["mes"].isin([10, 11, 12, 1, 2, 3])].copy()

    # Agregados por campaña
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
        f"{viq_col}_mean": "ndvi_anomalia_pct" if viq_col else None,
    })
    ndvi_camp = ndvi_camp[[c for c in ndvi_camp.columns if c is not None]]

    log.info(
        "NDVI: %d filas | campañas %d–%d",
        len(ndvi_camp),
        ndvi_camp["campania_inicio"].min(),
        ndvi_camp["campania_inicio"].max(),
    )
    ndvi_camp.to_parquet(out, index=False)
    return ndvi_camp


# ──────────────────────────────────────────────
# ENSAMBLE DEL PANEL
# ──────────────────────────────────────────────

def compute_lags_and_zrinde(df):
    """
    Agrega lags del rinde (1-5 años), media móvil, z-rinde sin look-ahead.
    CRÍTICO: rolling con shift(1) para evitar incluir campaña actual.
    """
    df = df.sort_values(["cultivo", "departamento", "campania_inicio"]).copy()

    # Lags y rolling por grupo (cultivo × departamento)
    result = []
    for (cult, depto), grp in df.groupby(["cultivo", "departamento"]):
        grp = grp.sort_values("campania_inicio").copy()
        rinde = grp["rinde_kgha"]

        for lag in range(1, 6):
            grp[f"rinde_lag{lag}"] = rinde.shift(lag)

        # Media móvil de 5 años sobre los 5 años ANTERIORES (shift(1) primero)
        # rolling(5, min_periods=3) sobre la serie shifteada
        rinde_shifted = rinde.shift(1)
        grp["rinde_ma5"]   = rinde_shifted.rolling(5, min_periods=3).mean()
        grp["rinde_std5"]  = rinde_shifted.rolling(5, min_periods=3).std()
        grp["z_rinde"]     = (rinde - grp["rinde_ma5"]) / grp["rinde_std5"].replace(0, np.nan)

        result.append(grp)

    df = pd.concat(result, ignore_index=True)
    return df


def flag_anomalas_train(df, oni):
    """
    Marca campañas anómalas dentro del período de train, usando criterio automático
    sin look-ahead (usamos z_rinde ya calculado con rolling correcto).

    Criterio 1: ≥30% de departamentos con z_rinde < -1.5 (por cultivo × campaña)
    Criterio 2: ONI Oct-Feb ≤ -0.5 Y rinde promedio regional < media_ma5 - 1*std
    """
    df = df.copy()
    df["es_anomala_train"] = False

    train_mask = df["campania_inicio"] <= TRAIN_END

    # Criterio 1
    c1 = (
        df[train_mask]
        .groupby(["cultivo", "campania_inicio"])
        .apply(lambda g: (g["z_rinde"] < -1.5).sum() / max(len(g), 1) >= 0.30)
        .reset_index()
        .rename(columns={0: "c1"})
    )

    # Criterio 2: ONI ≤ -0.5 sostenido Oct-Feb
    oni_nina = oni[oni["oni_oct_feb_mean"] <= -0.5][["campania_inicio"]].copy()
    oni_nina["oni_nina"] = True

    rinde_regional = (
        df[train_mask]
        .groupby(["cultivo", "campania_inicio"])
        .agg(rinde_reg_mean=("rinde_kgha", "mean"),
             rinde_reg_ma5=("rinde_ma5", "mean"),
             rinde_reg_std5=("rinde_std5", "mean"))
        .reset_index()
    )
    rinde_regional = rinde_regional.merge(oni_nina, on="campania_inicio", how="left")
    rinde_regional["oni_nina"] = rinde_regional["oni_nina"].fillna(False)
    rinde_regional["c2"] = (
        rinde_regional["oni_nina"] &
        (rinde_regional["rinde_reg_mean"] <
         rinde_regional["rinde_reg_ma5"] - rinde_regional["rinde_reg_std5"])
    )

    anomalas = c1.merge(rinde_regional[["cultivo","campania_inicio","c2"]],
                        on=["cultivo","campania_inicio"], how="left")
    anomalas["c2"] = anomalas["c2"].fillna(False)
    anomalas["es_anomala"] = anomalas["c1"] | anomalas["c2"]

    # Excluir campañas sin z_rinde suficiente (primeras 5): son anomalas=False por default
    camp_anomalas = anomalas[anomalas["es_anomala"]][["cultivo","campania_inicio"]].copy()
    df["es_anomala_train"] = False

    for _, row in camp_anomalas.iterrows():
        mask = (
            (df["cultivo"] == row["cultivo"]) &
            (df["campania_inicio"] == row["campania_inicio"]) &
            train_mask
        )
        df.loc[mask, "es_anomala_train"] = True

    # Primeras 5 campañas (1981-1985): excluimos del train AE directamente
    df.loc[(df["campania_inicio"] <= 1985) & train_mask, "es_anomala_train"] = True

    anomalas_log = (
        df[df["es_anomala_train"]]
        .groupby(["cultivo","campania_inicio"])
        .size()
        .reset_index(name="n_deptos")
    )
    log.info("\nCampañas marcadas como anómalas en train:")
    for _, r in anomalas_log.iterrows():
        log.info("  %s %d/%d", r["cultivo"], r["campania_inicio"], r["campania_inicio"]+1)

    return df


def normalize_climate_features(df, clim_cols):
    """
    Normalización z-score POR DEPARTAMENTO, parámetros calculados SOLO sobre train.
    Devuelve df con columnas _norm y guarda los parámetros de normalización.
    """
    train = df[df["campania_inicio"] <= TRAIN_END]
    norm_params = []

    for col in clim_cols:
        if col not in df.columns:
            continue
        # Estadísticos por departamento usando solo train
        stats = (
            train.groupby("departamento")[col]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{col}_mu", "std": f"{col}_sigma"})
            .reset_index()
        )
        df = df.merge(stats, on="departamento", how="left")
        sigma_col = f"{col}_sigma"
        mu_col    = f"{col}_mu"
        df[f"{col}_norm"] = (df[col] - df[mu_col]) / df[sigma_col].replace(0, np.nan)
        norm_params.append(stats)
        df = df.drop(columns=[mu_col, sigma_col])

    # Guardar parámetros
    if norm_params:
        all_params = norm_params[0]
        for p in norm_params[1:]:
            all_params = all_params.merge(p, on="departamento", how="outer")
        all_params.to_parquet(PROC / "norm_params.parquet", index=False)

    return df


def build_panel():
    log.info("=" * 60)
    log.info("BUILD PANEL NUCLEO — iniciando")
    log.info("=" * 60)

    # 1. Cargar fuentes
    oni   = load_oni()
    magyp = load_magyp()
    nasa  = load_nasa_power()
    ndvi  = load_ndvi()

    # 2. Base: MAGyP
    panel = magyp[[
        "cultivo", "campania_inicio", "provincia", "departamento",
        "sup_sembrada_ha", "sup_cosechada_ha", "produccion_tn", "rinde_kgha"
    ]].copy()

    # Añadir coordenadas de centroide
    centroides_df = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in CENTROIDES.items()],
        columns=["departamento", "lat_centroide", "lon_centroide"]
    )
    centroides_df["departamento"] = centroides_df["departamento"].apply(normalizar_nombre)
    panel = panel.merge(centroides_df, on="departamento", how="left")

    # Formato campaña string
    panel["campania"] = panel["campania_inicio"].apply(
        lambda y: f"{y}/{str(y+1)[-2:]}"
    )

    # 3. Merge ONI
    panel = panel.merge(oni, on="campania_inicio", how="left")

    # 4. Merge NASA POWER
    if nasa is not None:
        nasa["departamento"] = nasa["departamento"].apply(normalizar_nombre)
        panel = panel.merge(nasa, on=["departamento", "campania_inicio"], how="left")
        log.info("NASA POWER mergeado: %d columnas totales", len(panel.columns))

    # 5. Merge NDVI
    if ndvi is not None:
        ndvi["departamento"] = ndvi["departamento"].apply(normalizar_nombre)
        panel = panel.merge(ndvi, on=["departamento", "campania_inicio"], how="left")
        log.info("NDVI mergeado")

    # 6. Lags y z-rinde (sin look-ahead)
    panel = compute_lags_and_zrinde(panel)

    # 7. Flag campañas anómalas (train)
    panel = flag_anomalas_train(panel, oni)

    # 8. Split column
    def assign_split(y):
        if y <= TRAIN_END:    return "train"
        elif y <= VAL_END:    return "val"
        else:                 return "test"
    panel["split"] = panel["campania_inicio"].apply(assign_split)

    # 9. Normalización de features climáticas (solo sobre train, por departamento)
    clim_base = ["tmean", "tmax_p95", "precip_total", "gdd", "dias_t_mayor_35",
                 "rad_solar_mean", "precip_siembra", "precip_vegetativo",
                 "precip_r1_r5", "precip_llenado"]
    clim_monthly = [c for c in panel.columns
                    if any(c.startswith(f"{v}_") for v in
                           ["t2m","t2m_max","t2m_min","prectotcorr","rh2m","allsky_sfc_sw_dwn","ws2m"])]
    clim_all = [c for c in clim_base + clim_monthly if c in panel.columns]
    panel = normalize_climate_features(panel, clim_all)

    # 10. Guardar
    out_panel = PROC / "panel_nucleo.parquet"
    panel.to_parquet(out_panel, index=False)
    log.info("\n✓ Panel guardado: %s", out_panel)
    log.info("  Shape: %s", panel.shape)
    log.info("  Columnas: %d", len(panel.columns))
    log.info("  Campañas: %d–%d", panel["campania_inicio"].min(), panel["campania_inicio"].max())
    log.info("  Departamentos: %d", panel["departamento"].nunique())
    log.info("  Splits: %s", panel.groupby("split")["campania_inicio"].agg(["min","max"]).to_dict())

    # Guardar subconjunto de campañas normales para train del AE
    normales_train = panel[
        (panel["split"] == "train") &
        (~panel["es_anomala_train"])
    ][["cultivo", "campania_inicio", "departamento"]].drop_duplicates()
    normales_train.to_parquet(SPLITS / "campanias_normales_train.parquet", index=False)
    log.info("  Campañas normales train (AE): %d únicas",
             normales_train["campania_inicio"].nunique())

    return panel


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    panel = build_panel()
    print("\n=== RESUMEN DEL PANEL ===")
    print(panel[["cultivo","campania_inicio","departamento","split","rinde_kgha",
                 "es_anomala_train"]].describe(include="all"))
