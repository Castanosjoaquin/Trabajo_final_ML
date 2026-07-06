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


_META_COLS = ["provincia", "departamento", "campania", "campania_inicio",
              "rinde_kgha", "sup_sembrada_ha", "zona"]

# Momentos de predicción del calendario agrícola (ver build_reg_dataset):
#   'full'        : toda la campaña Sep–Mar (comportamiento histórico).
#   'pre_siembra' : solo lo conocible/estimable antes de sembrar — ONI (la
#                   propuesta lo incluye: en la práctica es el pronóstico ENSO)
#                   y humedad de suelo de invierno (sm_winter, Jun–Ago). Más las
#                   features estructurales (depto_enc, year, lags del rinde).
#   'pre_cosecha' : clima y NDVI observados hasta febrero (sin marzo), para
#                   pronosticar antes de la cosecha.
MOMENTOS = ("full", "pre_siembra", "pre_cosecha")


def _filter_momento(clim_cols: List[str], momento: str) -> List[str]:
    """Restringe las columnas climáticas/satelitales al momento de predicción."""
    if momento == "full":
        return list(clim_cols)
    if momento == "pre_cosecha":
        # Afuera todo lo con sufijo _mar y el ERA5 estacional que llega a marzo
        # (frost_days cuenta heladas Sep–Mar; frost_days_early es Sep–Nov y queda).
        return [c for c in clim_cols
                if not c.endswith("_mar") and c != "frost_days"]
    if momento == "pre_siembra":
        return [c for c in clim_cols
                if c.startswith("oni_") or c == "sm_winter"]
    raise ValueError(f"momento desconocido: {momento!r}. Opciones: {MOMENTOS}")


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
                      use_agro: bool = False, enc_smooth: float = 0.0,
                      use_lags: int = 0, momento: str = "full",
                      train_end: int = TRAIN_END,
                      test_start: int = TEST_START) -> RegDataset:
    """Arma el RegDataset de un cultivo: split temporal, features y escalado.

    `dataset` selecciona el set de features climáticas ('base' o 'era5_ndvi') y
    debe coincidir con el del `panel` que se pasa (ver load_panel). `use_agro`
    agrega las features agronómicas de ventana crítica del Componente A (balance
    hídrico, estrés térmico; ver `add_agro_features`).

    `enc_smooth` (m del m-estimate) suaviza el target encoding del departamento
    hacia la media global de train: enc = (n·media_depto + m·media_global)/(n+m).
    Con m=0 se recupera la media cruda (comportamiento histórico de los
    notebooks); m≈10 protege a los deptos con pocas campañas en train, cuyas
    medias crudas son ruidosas — recomendado al modelar por zona, donde los n
    por depto se achican.

    `use_lags` (k>0) agrega los k lags del rinde (`rinde_lag1..k`, con shift —
    NUNCA el año actual) y la media móvil de 5 campañas previas (`rinde_ma5`),
    las features autorregresivas que pide la propuesta. Los NaN del arranque de
    cada serie (primeras k campañas del depto) se rellenan con la media de
    train. Nota: si a un depto le falta una campaña intermedia, el lag es la
    última campaña DISPONIBLE (no el año calendario exacto).

    `momento` restringe las features al calendario agrícola (ver MOMENTOS):
    'pre_siembra' deja solo ONI + sm_winter + estructurales; 'pre_cosecha'
    excluye marzo. Con momento != 'full' no se admite `use_agro` (las ventanas
    críticas agronómicas llegan a marzo y filtrarían clima futuro)."""
    if momento != "full" and use_agro:
        raise ValueError("use_agro=True requiere momento='full': las ventanas "
                         "críticas agronómicas usan clima hasta marzo.")
    df, tr_mask, te_mask, clim_cols, geo = crop_frame(
        panel, cultivo, dataset, train_end, test_start)

    feature_cols = _filter_momento(clim_cols, momento)

    # --- Features agronómicas de dominio (ventana crítica del cultivo) ---
    if use_agro:
        df, agro_cols = _A_data.add_agro_features(df, cultivo)
        feature_cols += agro_cols

    # --- Lags del rinde (autorregresivas, sin filtrar el año actual) ---
    lag_cols: List[str] = []
    if use_lags > 0:
        g = df.groupby(geo)["rinde_kgha"]
        for k in range(1, use_lags + 1):
            col = f"rinde_lag{k}"
            df[col] = g.shift(k)
            lag_cols.append(col)
        df["rinde_ma5"] = df.groupby(geo)["rinde_kgha"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        lag_cols.append("rinde_ma5")
        feature_cols += lag_cols

    tr = df[tr_mask].copy()
    te = df[te_mask].copy()

    if lag_cols:
        # Relleno de los NaN del arranque de serie con la media de TRAIN (stat
        # de train → sin fuga hacia test; solo toca las primeras campañas).
        fill = float(tr["rinde_kgha"].mean())
        tr[lag_cols] = tr[lag_cols].fillna(fill)
        te[lag_cols] = te[lag_cols].fillna(fill)

    # --- Codificación del departamento por su rinde medio en train (sin leakage) ---
    y_train_mean = float(tr["rinde_kgha"].mean())
    grp = tr.groupby(geo)["rinde_kgha"]
    if enc_smooth > 0:
        # m-estimate: encoge la media del depto hacia la global según cuántas
        # campañas de train lo respaldan (n chico → más cerca de la global).
        agg = grp.agg(["mean", "size"])
        depto_mean = ((agg["size"] * agg["mean"] + enc_smooth * y_train_mean)
                      / (agg["size"] + enc_smooth))
    else:
        depto_mean = grp.mean()
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


def build_zona_datasets(cultivo: str, dataset: str = "base", n_zonas: int = 6,
                        method: str = "geo", min_train: int = 100, min_test: int = 20,
                        seed: int = 42, **kwargs) -> dict:
    """Un RegDataset por zona (para entrenar un modelo por zona).

    Asigna zonas con `assign_zonas` y arma, para cada una, un RegDataset con SU
    propio split temporal, codificación de depto y escalado (todo calculado dentro
    de la zona). Descarta zonas con pocas filas de train/test. `kwargs` se pasan a
    `build_reg_dataset` (p. ej. `use_agro=True`). Devuelve {zona: RegDataset}."""
    panel = assign_zonas(load_panel(dataset), n_zonas=n_zonas, method=method, seed=seed)
    out = {}
    for z in sorted(panel["zona"].dropna().unique()):
        ds = build_reg_dataset(panel[panel["zona"] == z], cultivo, dataset=dataset, **kwargs)
        if len(ds.y_train) >= min_train and len(ds.y_test) >= min_test:
            out[z] = ds
    return out


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
