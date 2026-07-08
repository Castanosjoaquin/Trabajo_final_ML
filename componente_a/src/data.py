"""
Pipeline de datos compartido del Componente A.

Carga el panel, computa la etiqueta proxy (z_rinde), arma los splits
temporales y normaliza por departamento SIN leakage (parámetros calculados
solo sobre las filas normales de train). Devuelve un CropDataset listo para
cualquier detector.

Todas las funciones toman parámetros explícitos con defaults (los de
config.py), así el notebook que quiera variar algo lo pasa a la vista:

    panel_z = data.prepare()                                # pipeline default
    ds = data.build_crop_dataset(panel_z, "soja")           # split default
    ds = data.build_crop_dataset(panel_z, "soja", train_end=2020)  # variante

Source-only: el panel se lee, nunca se escribe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (CLIM_PREFIXES, CRITICAL_MONTHS, ERA5_COLS, MESES,
                     PANEL_PATH, ROLLING_WINDOW, Z_THRESH,
                     TRAIN_END, TEST_START,
                     EXCLUDED_TRAIN_YEARS, _REPO_ROOT)

# --- Columnas clave que deben existir en el panel ---
_REQUIRED_BASE = ["cultivo", "campania", "campania_inicio", "departamento", "rinde_kgha"]


def build_feature_list(panel: pd.DataFrame, use_ndvi: bool = True,
                       use_era5: bool = True) -> List[str]:
    """Lista explícita de columnas que entran en X (promedios mensuales Sep–Mar
    de las variables climáticas + NDVI + ERA5). Solo incluye las que realmente
    existen en el panel. Con el panel unificado, NDVI y ERA5 vienen por defecto."""
    cols: List[str] = []
    for pfx in CLIM_PREFIXES:
        for mes in MESES:
            col = f"{pfx}_{mes}"
            if col in panel.columns:
                cols.append(col)
    if use_ndvi:
        # NDVI-AVHRR mensual (1981+, `ndvi_avhrr_<mes>`), ya en el panel unificado.
        cols += [f"ndvi_avhrr_{mes}" for mes in MESES
                 if f"ndvi_avhrr_{mes}" in panel.columns]
    if use_era5:
        cols += [c for c in ERA5_COLS if c in panel.columns]
    return cols


def load_panel(panel_path: str = PANEL_PATH, use_ndvi: bool = True,
               use_era5: bool = True) -> pd.DataFrame:
    """Carga el panel, valida columnas y deduplica.

    Un `panel_path` relativo se resuelve contra la raíz del repo (donde vive
    data/), así funciona corriendo desde cualquier cwd."""
    path = panel_path
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Panel no encontrado: {path}")
    panel = pd.read_parquet(path)

    feats = build_feature_list(panel, use_ndvi, use_era5)
    missing = [c for c in (_REQUIRED_BASE + feats) if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en el panel: {missing}\n"
            f"Columnas disponibles: {list(panel.columns)}"
        )

    # --- Dedup ---
    # El panel trae filas duplicadas (mismo depto-campaña-cultivo con lat/lon
    # distintos): el join geográfico pegó lat/lon por NOMBRE de departamento sin
    # respetar la provincia → explosión cartesiana. Dentro de cada grupo solo
    # varía lat/lon, así que el dedup es seguro Y NECESARIO: sin él, el rolling
    # de z_rinde incluye campañas repetidas y la etiqueta sale mal.
    dedup_key = [c for c in ["provincia", "departamento", "campania_inicio", "cultivo"]
                 if c in panel.columns]
    n0 = len(panel)
    panel = panel.drop_duplicates(subset=dedup_key).reset_index(drop=True)
    if len(panel) < n0:
        # ASCII a propósito: la consola de Windows (cp1252) no banca "→".
        print(f"[data] dedup panel: {n0} -> {len(panel)} filas "
              f"({n0 - len(panel)} duplicados espurios por lat/lon eliminados)")
    return panel


def compute_z_rinde(panel: pd.DataFrame, rolling_window: int = ROLLING_WINDOW,
                    z_thresh: float = Z_THRESH) -> pd.DataFrame:
    """z-score del rinde vs media móvil por departamento Y cultivo (etiqueta
    proxy). Se agrupa por [provincia, departamento, cultivo]: incluir provincia
    es imprescindible porque el nombre de departamento se repite entre
    provincias; cultivo separa soja (~2700 kg/ha) de maíz (~6700). Usa shift(1)
    para no filtrar el valor actual en su propia media. Etiqueta SOLO para
    evaluación.

    `rinde_kgha` se clipea al 0.5%/99.5% por cultivo antes del rolling: sin
    esto, valores de carga errónea (p. ej. 20000 kg/ha en soja, imposible)
    generan un roll_std minúsculo en ventanas tempranas con pocos datos y
    z_rinde explota (se vieron valores de |z|>40 sin este clip)."""
    geo = ["provincia", "departamento"] if "provincia" in panel.columns else ["departamento"]
    df = panel.sort_values(geo + ["cultivo", "campania_inicio"]).copy()
    limites = df.groupby("cultivo")["rinde_kgha"].transform(
        lambda s: s.clip(lower=s.quantile(0.005), upper=s.quantile(0.995))
    )
    n_clip = int((df["rinde_kgha"] != limites).sum())
    if n_clip:
        print(f"[data] clip rinde_kgha (0.5%/99.5% por cultivo): {n_clip} filas afectadas")
    df["rinde_kgha"] = limites
    grp = df.groupby(geo + ["cultivo"])["rinde_kgha"]
    roll_mean = grp.transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=3).mean()
    )
    roll_std = grp.transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=3).std()
    )
    df["z_rinde"] = (df["rinde_kgha"] - roll_mean) / roll_std.replace(0, np.nan)
    df["anomalia"] = (df["z_rinde"] < z_thresh).astype(int)
    return df


def prepare(panel_path: str = PANEL_PATH, use_ndvi: bool = True,
            use_era5: bool = True) -> pd.DataFrame:
    """Pipeline corto: carga el panel + computa la etiqueta. Devuelve panel_z."""
    return compute_z_rinde(load_panel(panel_path, use_ndvi, use_era5))


def add_agro_features(df: pd.DataFrame, cultivo: str) -> tuple:
    """Deriva features agronómicas centradas en la ventana crítica del cultivo,
    a partir de las columnas mensuales ya presentes (no requiere datos nuevos).

    Devuelve (df_con_cols, nombres_nuevos). Todas se normalizan luego por depto
    como el resto.

    - agro_precip_crit  : precip total (NASA POWER) en ventana crítica.
    - agro_tmax_crit    : t2m_max media en ventana crítica (estrés térmico).
    - agro_thermamp_crit: amplitud térmica media (t2m_max − t2m_min).
    - agro_waterbal_crit: balance hídrico Σ(precip − PET) con PET Hargreaves
      proxy = 0.0023·Ra·(Tmean+17.8)·√(Tmax−Tmin), Ra≈allsky_sfc_sw_dwn. Como
      la feature se z-scorea por depto, las constantes no importan, solo la
      variación relativa.
    """
    months = CRITICAL_MONTHS.get(cultivo, [])
    df = df.copy()

    def col(prefix: str, m: str):
        c = f"{prefix}_{m}"
        return df[c].values if c in df.columns else None

    precip_sum = tmax_acc = amp_acc = wb_sum = None
    n_precip = n_tmax = n_amp = n_wb = 0
    for m in months:
        precip = col("prectotcorr", m)
        tmax, tmin = col("t2m_max", m), col("t2m_min", m)
        tmean, rad = col("t2m", m), col("allsky_sfc_sw_dwn", m)
        if precip is not None:
            precip_sum = precip if precip_sum is None else precip_sum + precip
            n_precip += 1
        if tmax is not None:
            tmax_acc = tmax if tmax_acc is None else tmax_acc + tmax
            n_tmax += 1
        if tmax is not None and tmin is not None:
            amp = tmax - tmin
            amp_acc = amp if amp_acc is None else amp_acc + amp
            n_amp += 1
        if all(v is not None for v in (rad, tmean, tmax, tmin)):
            pet = 0.0023 * rad * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None))
            wb = precip - pet
            wb_sum = wb if wb_sum is None else wb_sum + wb
            n_wb += 1

    new_cols: List[str] = []
    if n_precip:
        df["agro_precip_crit"] = precip_sum;          new_cols.append("agro_precip_crit")
    if n_tmax:
        df["agro_tmax_crit"] = tmax_acc / n_tmax;     new_cols.append("agro_tmax_crit")
    if n_amp:
        df["agro_thermamp_crit"] = amp_acc / n_amp;   new_cols.append("agro_thermamp_crit")
    if n_wb:
        df["agro_waterbal_crit"] = wb_sum;            new_cols.append("agro_waterbal_crit")
    return df, new_cols


@dataclass
class CropDataset:
    """Datos de un cultivo listos para entrenar/evaluar. Los X ya están
    normalizados por departamento. Los meta_* conservan identificadores y la
    etiqueta proxy para evaluación y visualización. Solo train/test (sin val)."""

    cultivo: str
    feature_cols: List[str]

    X_train: np.ndarray       # solo campañas normales ≤ TRAIN_END (lo que ve el modelo)
    X_test: np.ndarray        # ≥ TEST_START (con sus anomalías, para evaluar)

    meta_train: pd.DataFrame
    meta_test: pd.DataFrame

    y_test: np.ndarray

    # diagnósticos
    n_excluded_year: int
    n_excluded_anom: int


_META_COLS = ["departamento", "campania", "campania_inicio", "cultivo",
              "rinde_kgha", "z_rinde", "anomalia"]


def _normalize_per_depto(df_in: pd.DataFrame, stats: pd.DataFrame,
                         feature_cols: List[str], geo: List[str],
                         global_std: pd.Series) -> pd.DataFrame:
    """Aplica z-score por departamento (clave `geo` = [provincia, departamento])
    usando estadísticas ya calculadas (de train_normal). No recomputa nada →
    sin leakage. La clave incluye provincia para no normalizar juntos
    departamentos homónimos de provincias distintas.

    Cuando la std por-depto es 0 o NaN (feature constante en ese depto, p. ej.
    `frost_days`=0 en el norte que nunca hiela), se usa la std GLOBAL del feature
    como fallback: así NO se cae la fila Y se preserva la señal — un depto que
    nunca hiela y de golpe tiene una helada se vuelve anómalo en vez de perderse."""
    df_out = df_in.copy()
    stats_r = stats.reset_index()
    for col in feature_cols:
        merged = df_out[geo].merge(
            stats_r[geo + [f"{col}_mean", f"{col}_std"]], on=geo, how="left",
        )
        std_vals = merged[f"{col}_std"].values.astype(float)
        bad = ~np.isfinite(std_vals) | (std_vals == 0)
        std_vals[bad] = global_std.get(col, np.nan)        # fallback: std global
        std_vals = np.where(std_vals == 0, np.nan, std_vals)  # global también 0 → NaN
        df_out[col] = (df_out[col].values - merged[f"{col}_mean"].values) / std_vals
    return df_out


def build_crop_dataset(panel_z: pd.DataFrame, cultivo: str,
                       use_ndvi: bool = True, use_era5: bool = True,
                       use_agro: bool = False,
                       train_start: Optional[int] = None,
                       train_end: int = TRAIN_END,
                       test_start: int = TEST_START,
                       excluded_train_years: Sequence[int] = tuple(EXCLUDED_TRAIN_YEARS),
                       ) -> CropDataset:
    """Construye el CropDataset para un cultivo: split train/test, filtrado de
    train a filas normales, y normalización por departamento sin leakage.
    No hay val (el bloque 2018–2020 se pliega al train; ver config)."""
    feats = build_feature_list(panel_z, use_ndvi, use_era5)
    df = panel_z[panel_z["cultivo"] == cultivo].copy()

    # --- Features agronómicas de dominio (ventana crítica, per-cultivo) ---
    if use_agro:
        df, agro_cols = add_agro_features(df, cultivo)
        feats = feats + agro_cols

    # --- Split temporal: solo train (≤ train_end) y test (≥ test_start) ---
    m_train = df["campania_inicio"] <= train_end
    if train_start is not None:
        m_train &= df["campania_inicio"] >= train_start
    m_test = df["campania_inicio"] >= test_start
    df_train, df_test = df[m_train].copy(), df[m_test].copy()

    # --- Train normal: excluir años problemáticos + filas anómalas ---
    m_excl = df_train["campania_inicio"].isin(list(excluded_train_years))
    m_anom = df_train["anomalia"] == 1
    df_train_normal = df_train[~m_excl & ~m_anom].copy()

    # --- Normalización por depto (fit en train_normal) ---
    # Clave geográfica con provincia: no mezcla departamentos homónimos.
    geo = ["provincia", "departamento"] if "provincia" in df.columns else ["departamento"]
    stats = df_train_normal.groupby(geo)[feats].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    # std global por feature: fallback cuando la std por-depto es 0/NaN.
    global_std = df_train_normal[feats].std()

    df_tr_n = _normalize_per_depto(df_train_normal, stats, feats, geo, global_std).dropna(subset=feats)
    df_te_n = _normalize_per_depto(df_test, stats, feats, geo, global_std).dropna(subset=feats)

    meta_cols = (["provincia"] if "provincia" in df.columns else []) + _META_COLS

    return CropDataset(
        cultivo=cultivo,
        feature_cols=feats,
        X_train=df_tr_n[feats].values,
        X_test=df_te_n[feats].values,
        meta_train=df_tr_n[meta_cols].reset_index(drop=True),
        meta_test=df_te_n[meta_cols].reset_index(drop=True),
        y_test=df_te_n["anomalia"].values,
        n_excluded_year=int(m_excl.sum()),
        n_excluded_anom=int((~m_excl & m_anom).sum()),
    )
