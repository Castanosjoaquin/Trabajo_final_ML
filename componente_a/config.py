"""
Configuración compartida del Componente A.

Centraliza rutas, splits temporales, años excluidos y la definición de la
matriz de features. Cualquier modelo (Isolation Forest, Autoencoder, ...)
parte de esta misma configuración para que la comparación sea justa.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

# --- Rutas ---
PANEL_PATH = "data/processed/panel_nucleo.parquet"
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
NDVI_COLS = ["ndvi_mean", "ndvi_min", "ndvi_max", "ndvi_anomalia_pct"]


@dataclass
class ExperimentConfig:
    """Configuración de un experimento. Se serializa a config.json y, cuando
    se active W&B, se mapea 1:1 a wandb.config."""

    # Datos / etiqueta
    panel_path: str = PANEL_PATH
    use_ndvi: bool = False               # ablation post-2002
    rolling_window: int = 5              # ventana para z_rinde
    z_thresh: float = -1.5              # umbral de etiqueta anómala

    # Evaluación: operating point para F1/precision/recall
    threshold_mode: str = "contamination"   # 'contamination' | 'f1'
    eval_contamination: float = 0.10         # fracción marcada como anómala

    # Split temporal (sobre campania_inicio, entero)
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
