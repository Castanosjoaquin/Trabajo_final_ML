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
PANEL_PATH = os.path.join(DATA_DIR, "processed", "panel_union.parquet")
PANEL_NDVI_PATH = os.path.join(DATA_DIR, "processed", "panel_union_ndvi.parquet")
PANEL_ERA5_PATH = os.path.join(DATA_DIR, "processed", "panel_union_era5.parquet")

# --- Cultivos soportados ---
CULTIVOS = ["soja", "maiz"]

# --- Meses de la campaña (Sep–Mar) ---
MESES = ["sep", "oct", "nov", "dic", "ene", "feb", "mar"]

# --- Prefijos de variables climáticas (NASA POWER + ONI) ---
# NO incluye rinde ni superficie. 7 variables × 7 meses + ONI × 5 meses = 54.
CLIM_PREFIXES = [
    "allsky_sfc_sw_dwn",  # radiación solar
    "prectotcorr",        # precipitación NASA POWER
    "rh2m",               # humedad relativa
    "t2m",                # temperatura media
    "t2m_max",            # temperatura máxima
    "t2m_min",            # temperatura mínima
    "ws2m",               # viento
    "oni",                # ONI (ENSO) — solo meses disponibles en el panel
    "chirps_precip",      # precipitación CHIRPS (no está en el panel actual)
]
# NDVI: el único usado es el AVHRR mensual (1981+, `ndvi_avhrr_<mes>`) del panel
# aumentado por merge_avhrr_ndvi.py. El NDVI viejo de MODIS se RETIRÓ del panel
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
