"""Pipeline de datos del Componente B — predicción de rinde (regresión).

Construye, por cultivo, un dataset listo para entrenar los regresores de
`modelos/`: X (features escaladas) e y (`rinde_kgha`, kg/ha), con split temporal
train/test consistente con el Componente A (train ≤2020, test ≥2021).

Reutiliza del Componente A la carga del panel (dedup de las filas espurias por
lat/lon) y la lista de features climáticas mensuales; ver `componente_a/src/data.py`.

Las features que ve el modelo son de tres tipos, todas escaladas con parámetros
calculados SOLO en train (sin leakage):

  1. Clima mensual (Sep–Mar): las 54 columnas de NASA POWER + ONI.
  2. Codificación del departamento por su rinde medio en train ("mean/target
     encoding"): captura la estructura espacial (un depto rinde sistemáticamente
     más que otro) sin explotar en cientos de dummies. Deptos nuevos en test →
     media global de train.
  3. Año (`campania_inicio`): captura la tendencia tecnológica (el rinde sube con
     las décadas). OJO: en test (2021–2024) el año queda fuera del rango de train,
     así que los árboles (XGBoost) no extrapolan la tendencia y los lineales sí.

El target `rinde_kgha` se deja crudo (kg/ha); cada modelo lo escala internamente
si lo necesita (la red neuronal lo hace).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Reutilizar el pipeline del Componente A (carga + dedup + lista de features).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_A_ROOT = os.path.join(_REPO_ROOT, "componente_a")
for _p in (_REPO_ROOT, _A_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import data as _A_data      # noqa: E402  (componente_a/src)
from src import config as _A_config  # noqa: E402

# Split temporal (mismo criterio que el Componente A).
TRAIN_END = 2020
TEST_START = 2021

# Datasets de features disponibles:
#   'base'      : solo clima mensual (NASA POWER + ONI) — panel_union.
#   'era5_ndvi' : clima + NDVI-AVHRR mensual + ERA5-Land (humedad de suelo, heladas),
#                 fusionando panel_union_ndvi y panel_union_era5.
DATASETS = ("base", "era5_ndvi")


@dataclass
class RegDataset:
    """Datos de un cultivo listos para regresión de rinde.

    X ya están escaladas (StandardScaler ajustado en train). y es `rinde_kgha`
    crudo. Los meta_* conservan identificadores y el rinde para baselines/plots.
    """

    cultivo: str
    feature_cols: List[str]

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray          # rinde_kgha (kg/ha)
    y_test: np.ndarray

    meta_train: pd.DataFrame
    meta_test: pd.DataFrame

    # para baselines
    y_train_mean: float                    # media global de train
    depto_mean: pd.Series                  # rinde medio por (prov, depto) en train


_META_COLS = ["provincia", "departamento", "campania", "campania_inicio", "rinde_kgha"]


def load_panel(dataset: str = "base") -> pd.DataFrame:
    """Panel deduplicado con el set de features del `dataset` elegido.

    'base' delega directo en el Componente A. 'era5_ndvi' fusiona las columnas
    extra de los dos paneles aumentados (NDVI y ERA5) sobre el panel base,
    uniendo por la clave geográfica-temporal (con provincia, para no mezclar
    departamentos homónimos)."""
    if dataset == "base":
        return _A_data.load_panel()
    if dataset == "era5_ndvi":
        # Cada panel viene ya deduplicado por load_panel del Componente A.
        base_ndvi = _A_data.load_panel(_A_config.PANEL_NDVI_PATH, use_ndvi=True)
        era5 = _A_data.load_panel(_A_config.PANEL_ERA5_PATH, use_era5=True)
        keys = [c for c in ["provincia", "departamento", "campania_inicio", "cultivo"]
                if c in base_ndvi.columns]
        return base_ndvi.merge(era5[keys + list(_A_config.ERA5_COLS)], on=keys, how="left")
    raise ValueError(f"dataset desconocido: {dataset!r}. Opciones: {DATASETS}")


def crop_frame(panel: pd.DataFrame, cultivo: str, dataset: str = "base",
               train_end: int = TRAIN_END, test_start: int = TEST_START):
    """Filas de un cultivo en orden DETERMINÍSTICO, con sus features climáticas.

    Devuelve (df, tr_mask, te_mask, clim_cols, geo). El orden fijo (ordenar por
    geo + campania_inicio y resetear el índice) es lo que garantiza que el
    RegDataset (`datos`) y las features del VAE (`latente`) queden ALINEADOS fila
    a fila aunque partan de paneles ordenados distinto."""
    aug = dataset != "base"
    clim_cols = _A_data.build_feature_list(panel, use_ndvi=aug, use_era5=aug)
    geo = ["provincia", "departamento"] if "provincia" in panel.columns else ["departamento"]
    df = (panel[panel["cultivo"] == cultivo]
          .dropna(subset=clim_cols + ["rinde_kgha"])
          .sort_values(geo + ["campania_inicio"])
          .reset_index(drop=True))
    tr_mask = df["campania_inicio"] <= train_end
    te_mask = df["campania_inicio"] >= test_start
    return df, tr_mask, te_mask, clim_cols, geo


def build_reg_dataset(panel: pd.DataFrame, cultivo: str, dataset: str = "base",
                      use_depto_encoding: bool = True, use_year: bool = True,
                      use_agro: bool = False,
                      train_end: int = TRAIN_END,
                      test_start: int = TEST_START) -> RegDataset:
    """Arma el RegDataset de un cultivo: split temporal, features y escalado.

    `dataset` selecciona el set de features climáticas ('base' o 'era5_ndvi') y
    debe coincidir con el del `panel` que se pasa (ver load_panel). `use_agro`
    agrega las features agronómicas de ventana crítica del Componente A (balance
    hídrico, estrés térmico; ver `add_agro_features`)."""
    df, tr_mask, te_mask, clim_cols, geo = crop_frame(
        panel, cultivo, dataset, train_end, test_start)

    feature_cols = list(clim_cols)

    # --- Features agronómicas de dominio (ventana crítica del cultivo) ---
    if use_agro:
        df, agro_cols = _A_data.add_agro_features(df, cultivo)
        feature_cols += agro_cols

    tr = df[tr_mask].copy()
    te = df[te_mask].copy()

    # --- Codificación del departamento por su rinde medio en train (sin leakage) ---
    depto_mean = tr.groupby(geo)["rinde_kgha"].mean()
    y_train_mean = float(tr["rinde_kgha"].mean())
    if use_depto_encoding:
        def _enc(d: pd.DataFrame) -> np.ndarray:
            keys = list(d[geo].itertuples(index=False, name=None))
            return np.array([depto_mean.get(k, y_train_mean) for k in keys], dtype=float)
        tr["depto_enc"] = _enc(tr)
        te["depto_enc"] = _enc(te)
        feature_cols.append("depto_enc")

    # --- Año (tendencia) ---
    if use_year:
        tr["year"] = tr["campania_inicio"].astype(float)
        te["year"] = te["campania_inicio"].astype(float)
        feature_cols.append("year")

    # --- Escalado: StandardScaler ajustado SOLO en train ---
    mu = tr[feature_cols].mean()
    sd = tr[feature_cols].std().replace(0, 1.0)
    Xtr = ((tr[feature_cols] - mu) / sd).values.astype(np.float32)
    Xte = ((te[feature_cols] - mu) / sd).values.astype(np.float32)

    return RegDataset(
        cultivo=cultivo,
        feature_cols=feature_cols,
        X_train=Xtr,
        X_test=Xte,
        y_train=tr["rinde_kgha"].values.astype(np.float32),
        y_test=te["rinde_kgha"].values.astype(np.float32),
        meta_train=tr[[c for c in _META_COLS if c in tr.columns]].reset_index(drop=True),
        meta_test=te[[c for c in _META_COLS if c in te.columns]].reset_index(drop=True),
        y_train_mean=y_train_mean,
        depto_mean=depto_mean,
    )


def prepare(cultivo: str, dataset: str = "base", **kwargs) -> RegDataset:
    """Atajo: carga el panel del `dataset` y arma el RegDataset del cultivo."""
    return build_reg_dataset(load_panel(dataset), cultivo, dataset=dataset, **kwargs)


def assign_zonas(panel: pd.DataFrame, n_zonas: int = 6, method: str = "geo",
                 seed: int = 42) -> pd.DataFrame:
    """Agrega una columna `zona` para poder modelar/analizar por separado.

    - 'provincia' : zona = provincia (15 zonas, tamaños muy dispares).
    - 'geo'       : agrupa DEPARTAMENTOS cercanos con KMeans sobre su centroide
                    (lat, lon) en `n_zonas` zonas geográficas contiguas. Más
                    balanceadas que las provincias y respetan la cercanía (la
                    idea: dentro de una zona el clima es parecido, así se reduce
                    el ruido entre zonas de climas distintos).

    Devuelve una copia del panel con la columna `zona`."""
    out = panel.copy()
    if method == "provincia":
        out["zona"] = out["provincia"].astype(str)
        return out
    if method == "geo":
        from sklearn.cluster import KMeans
        geo = ["provincia", "departamento"] if "provincia" in out.columns else ["departamento"]
        cent = out.dropna(subset=["lat", "lon"]).groupby(geo)[["lat", "lon"]].mean()
        km = KMeans(n_clusters=n_zonas, random_state=seed, n_init=10)
        cent["zona"] = [f"z{c}" for c in km.fit_predict(cent[["lat", "lon"]].values)]
        out = out.merge(cent["zona"].reset_index(), on=geo, how="left")
        return out
    raise ValueError(f"method desconocido: {method!r}. Opciones: 'geo', 'provincia'.")
