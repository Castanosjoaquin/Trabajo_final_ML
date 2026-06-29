"""
Configuración compartida del Componente A.

Centraliza rutas, splits temporales, años excluidos y la definición de la
matriz de features. Cualquier modelo (Isolation Forest, Autoencoder, ...)
parte de esta misma configuración para que la comparación sea justa.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

# --- Rutas ---
PANEL_PATH = "data/processed/panel_union.parquet"
RUNS_DIR = "runs"  # raíz de los run-dirs locales (mirror del esquema W&B)

# --- Proyecto W&B (usado cuando se active WandbBackend) ---
WANDB_PROJECT = "componente-a-anomalias"
WANDB_ENTITY = None  # tu usuario/equipo de W&B; None = default de la cuenta

# --- Cultivos soportados ---
CULTIVOS = ["soja", "maiz"]

# --- Meses de la campaña (Sep–Mar) ---
MESES = ["sep", "oct", "nov", "dic", "ene", "feb", "mar"]

# --- Prefijos de variables climáticas (NASA POWER + CHIRPS + ONI) ---
# NO incluye rinde ni superficie. NDVI entra solo si USE_NDVI=True.
CLIM_PREFIXES = [
    "allsky_sfc_sw_dwn",  # radiación solar
    "prectotcorr",        # precipitación NASA POWER
    "rh2m",               # humedad relativa
    "t2m",                # temperatura media
    "t2m_max",            # temperatura máxima
    "t2m_min",            # temperatura mínima
    "ws2m",               # viento
    "oni",                # ONI (ENSO) — solo meses disponibles en el panel
    "chirps_precip",      # precipitación CHIRPS
]
# Solo ndvi_anomalia_pct: las otras 3 (ndvi_mean/min/max) son CONSTANTES por
# departamento (un NDVI climatológico, mismo valor en todas las campañas) →
# cero señal temporal y degeneran la normalización por depto (std=0 → NaN, tira
# todas las filas). ndvi_anomalia_pct sí varía año a año (índice de anomalía
# vegetal, ~100 = normal) y es la feature NDVI realmente informativa.
NDVI_COLS = ["ndvi_anomalia_pct"]

# --- Features ERA5-Land (estado del suelo + heladas), del merge externo ---
# Capturan lo que los promedios mensuales NO ven: el ESTADO inicial (humedad de
# suelo, la "inercia" de arrancar inundado o seco) y EVENTOS puntuales (heladas
# que un promedio mensual borra). Disponibles 1950+, se mergean con merge_era5.py.
#   sm_planting      : humedad de suelo 0-100cm en la siembra (Sep–Nov).
#   sm_winter        : humedad de suelo 0-100cm en el invierno previo (Jun–Ago)
#                      = condición inicial / inercia.
#   frost_days       : nº de días con helada (Tmin<0) en la campaña (Sep–Mar).
#   frost_days_early : nº de heladas tardías de primavera (Sep–Nov), las más dañinas.
ERA5_COLS = ["sm_planting", "sm_winter", "frost_days", "frost_days_early"]

# --- Ventana crítica del cultivo (floración + llenado de grano) ---
# Meses donde el estrés hídrico/térmico pega más fuerte en el rinde. Aproximado
# y tuneable. Soja: floración/llenado Dic–Feb. Maíz: siembra más temprana →
# período crítico Nov–Ene. Usado por las features agronómicas derivadas.
CRITICAL_MONTHS = {
    "soja": ["dic", "ene", "feb"],
    "maiz": ["nov", "dic", "ene"],
}


@dataclass
class ExperimentConfig:
    """Configuración de un experimento. Se serializa a config.json y, cuando
    se active W&B, se mapea 1:1 a wandb.config."""

    # Datos / etiqueta
    panel_path: str = PANEL_PATH
    use_ndvi: bool = False               # ablation post-2002
    use_agro_features: bool = False      # anexar features de dominio (ventana crítica)
    use_era5_features: bool = False      # anexar features ERA5-Land (suelo + heladas)
    rolling_window: int = 5              # ventana para z_rinde
    z_thresh: float = -1.5              # umbral de etiqueta anómala

    # Evaluación: operating point para F1/precision/recall
    threshold_mode: str = "contamination"   # 'contamination' | 'f1'
    eval_contamination: float = 0.10         # fracción marcada como anómala

    # Split temporal (sobre campania_inicio, entero)
    train_start: Optional[int] = None    # límite inferior del train (None = sin límite).
                                         # Usarlo p. ej. =2002 para controlar el período
                                         # al comparar con/sin NDVI (que solo existe 2002+).
    train_end: int = 2017                # train: 1981/82–2017/18
    val_start: int = 2018                # val: 2018/19–2020/21
    val_end: int = 2020
    test_start: int = 2021               # test: 2021/22–2024/25

    # Años excluidos del entrenamiento (>30% deptos anómalos)
    excluded_train_years: List[int] = field(
        default_factory=lambda: [1988, 1996, 2008, 2017]
    )

    # Reproducibilidad
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
