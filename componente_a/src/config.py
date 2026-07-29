"""
Constantes compartidas del Componente A: rutas, features, splits y etiqueta.

Un solo lugar para los valores que usan los notebooks y el pipeline de datos.
Los hiperparámetros de cada modelo NO viven acá: van visibles en el notebook
que entrena ese modelo.
"""
from __future__ import annotations

import os

# --- Rutas ---
# data/ vive en la RAÍZ del repo (compartida entre componentes). Se resuelve en
# ABSOLUTO desde este archivo (componente_a/src/config.py → tres niveles arriba),
# para no depender del cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
# Panel ÚNICO del proyecto: base (rinde + clima) + NDVI-AVHRR + ERA5-Land, ya
# unificado por build_panel_union.py. No hay más variantes de dataset.
PANEL_PATH = os.path.join(DATA_DIR, "processed", "panel_union.parquet")

# --- Cultivos soportados ---
CULTIVOS = ["soja", "maiz"]

# --- Meses de la campaña (Sep–Mar) ---
MESES = ["sep", "oct", "nov", "dic", "ene", "feb", "mar"]

# --- Prefijos de variables climáticas (NASA POWER + ONI + CHIRPS) ---
# NO incluye rinde ni superficie. 7 vars NASA POWER × 7 meses + ONI × 5 meses
# + CHIRPS × 7 meses = 61 (con NDVI-AVHRR y ERA5 la X final llega a 72 features).
CLIM_PREFIXES = [
    "allsky_sfc_sw_dwn",  # radiación solar
    "prectotcorr",        # precipitación NASA POWER
    "rh2m",               # humedad relativa
    "t2m",                # temperatura media
    "t2m_max",            # temperatura máxima
    "t2m_min",            # temperatura mínima
    "ws2m",               # viento
    "oni",                # ONI (ENSO) — solo meses disponibles en el panel
    "chirps_precip",      # precipitación CHIRPS (7 meses en el panel unificado)
]
# NDVI: el único usado es el AVHRR mensual (1981+, `ndvi_avhrr_<mes>`), integrado en
# el panel unificado por build_panel_union.py. El NDVI viejo de MODIS se RETIRÓ
# (2026-07): existía solo desde 2002 y 3 de sus 4 columnas eran estáticas por depto.

# --- Features ERA5-Land (estado del suelo + heladas), del merge externo ---
#   sm_planting      : humedad de suelo 0-100cm en la siembra (Sep–Nov).
#   sm_winter        : humedad de suelo 0-100cm en el invierno previo (Jun–Ago).
#   frost_days       : nº de días con helada (Tmin<0) en la campaña (Sep–Mar).
#   frost_days_early : nº de heladas tardías de primavera (Sep–Nov).
ERA5_COLS = ["sm_planting", "sm_winter", "frost_days", "frost_days_early"]

# --- Ventana crítica del cultivo (floración + llenado de grano) ---
# Meses donde el estrés hídrico/térmico pega más fuerte en el rinde.
CRITICAL_MONTHS = {
    "soja": ["dic", "ene", "feb"],
    "maiz": ["nov", "dic", "ene"],
}

# --- Suelo (SoilGrids v2.0) y geografía (SRTM + HydroSHEDS) ---
# Capas ESTÁTICAS por departamento: no varían por campaña ni por cultivo. Las
# extrae eda/build_capas_estaticas.py y las mergea el paso 6c del ETL. Son 44
# columnas de suelo (11 propiedades × 4 profundidades) + 4 de geografía.
SUELO_PROFUNDIDADES = ["0_5", "5_15", "15_30", "30_60"]
SUELO_PROPS = ["bdod", "cec", "clay", "sand", "silt", "nitrogen", "phh2o", "soc",
               "wv0010", "wv0033", "wv1500"]
GEO_COLS = ["geo_elev_mean", "geo_elev_std", "geo_slope_mean", "geo_dist_rio_km"]

# Espesor de cada horizonte en mm. Se usa para integrar el agua útil a lámina
# (mm de agua disponible) y para ponderar los promedios por profundidad: un
# promedio simple le daría el mismo peso a un horizonte de 5 cm que a uno de 30.
SUELO_ESPESOR_MM = {"0_5": 50, "5_15": 100, "15_30": 150, "30_60": 300}

# Horizontes del promedio "superficial" (0-30 cm) de las variables químicas,
# donde se concentra la actividad radicular temprana.
SUELO_PROF_SUPERFICIAL = ["0_5", "5_15", "15_30"]

# --- Etiqueta proxy ---
ROLLING_WINDOW = 5     # ventana de la media móvil del rinde
Z_THRESH = -1.5        # anomalía = z_rinde < Z_THRESH

# --- Split temporal (sobre campania_inicio, entero) ---
# Solo train / test. NO hay validación: el bloque 2018–2020 se pliega al train
# porque sus anomalías no tienen firma climática (val "ciego" → PR-AUC ≈ azar,
# no discrimina modelos). La selección de HP se ilustra en test (data snooping
# declarado). El early stopping de los autoencoders usa un 15% interno del train,
# no este split.
TRAIN_END = 2020       # train: 1981/82–2020/21 (incluye el ex-val)
TEST_START = 2021      # test:  2021/22–2024/25

# --- Años excluidos del train (sequías generalizadas; ver experiments/00) ---
EXCLUDED_TRAIN_YEARS = [1988, 1996, 2008, 2017]
