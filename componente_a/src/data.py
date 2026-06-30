"""
Pipeline de datos compartido del Componente A.

Carga el panel, computa la etiqueta proxy (z_rinde), arma los splits
temporales y normaliza por departamento SIN leakage (parámetros calculados
solo sobre las filas normales de train). Devuelve un CropDataset listo para
cualquier detector.

Source-only: el panel se lee, nunca se escribe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import (ExperimentConfig, CLIM_PREFIXES, CRITICAL_MONTHS, NDVI_COLS,
                     ERA5_COLS, MESES, _REPO_ROOT)


# --- Columnas clave que deben existir en el panel ---
_REQUIRED_BASE = ["cultivo", "campania", "campania_inicio", "departamento", "rinde_kgha"]


def build_feature_list(panel: pd.DataFrame, use_ndvi: bool,
                       use_era5: bool = False) -> List[str]:
    """Lista explícita de columnas que entran en X (promedios mensuales Sep–Mar
    de las variables climáticas). Solo incluye las que realmente existen."""
    cols: List[str] = []
    for pfx in CLIM_PREFIXES:
        for mes in MESES:
            col = f"{pfx}_{mes}"
            if col in panel.columns:
                cols.append(col)
    if use_ndvi:
        # Preferencia: NDVI-AVHRR mensual (1981+, `ndvi_avhrr_<mes>`, del panel
        # aumentado) si está. Solo si NO está se cae al NDVI viejo
        # (`ndvi_anomalia_pct`, MODIS 2002+) — así usar el panel con AVHRR NO
        # re-introduce el recorte a 2002+ por incluir la columna vieja.
        avhrr = [f"ndvi_avhrr_{mes}" for mes in MESES
                 if f"ndvi_avhrr_{mes}" in panel.columns]
        cols += avhrr if avhrr else [c for c in NDVI_COLS if c in panel.columns]
    if use_era5:
        cols += [c for c in ERA5_COLS if c in panel.columns]
    return cols


def load_panel(cfg: ExperimentConfig) -> pd.DataFrame:
    """Carga el panel y valida columnas. Falla con mensaje claro si falta algo.

    Un `panel_path` relativo (p. ej. el de un config override) se resuelve contra
    la raíz del repo, donde vive data/ — así funciona corriendo desde cualquier cwd."""
    path = cfg.panel_path
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Panel no encontrado: {path}")
    panel = pd.read_parquet(path)

    feats = build_feature_list(panel, cfg.use_ndvi, cfg.use_era5_features)
    missing = [c for c in (_REQUIRED_BASE + feats) if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en el panel: {missing}\n"
            f"Columnas disponibles: {list(panel.columns)}"
        )

    # --- Dedup ---
    # El panel trae filas duplicadas (mismo depto-campaña-cultivo con lat/lon
    # distintos): el join geográfico pegó lat/lon por NOMBRE de departamento sin
    # respetar la provincia → explosión cartesiana (p. ej. la "Capital" de
    # Corrientes quedó con las 6 coordenadas de las 6 "Capital" del país).
    # Dentro de cada grupo solo varía lat/lon (rinde/clima/NDVI idénticos), así
    # que el dedup es seguro Y NECESARIO: sin él, el rolling de z_rinde incluye
    # campañas repetidas y la etiqueta sale mal.
    dedup_key = [c for c in ["provincia", "departamento", "campania_inicio", "cultivo"]
                 if c in panel.columns]
    n0 = len(panel)
    panel = panel.drop_duplicates(subset=dedup_key).reset_index(drop=True)
    if len(panel) < n0:
        print(f"[data] dedup panel: {n0} → {len(panel)} filas "
              f"({n0 - len(panel)} duplicados espurios por lat/lon eliminados)")
    return panel


def compute_z_rinde(panel: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """z-score del rinde vs media móvil por departamento Y cultivo (etiqueta
    proxy). Se agrupa por [provincia, departamento, cultivo]: incluir provincia
    es imprescindible porque el nombre de departamento se repite entre
    provincias (p. ej. "25 De Mayo" en BA, La Pampa y San Juan) — agrupar solo
    por nombre mezclaría sus series de rinde en un mismo baseline. Cultivo
    separa soja (~2700 kg/ha) de maíz (~6700). Usa shift(1) para no filtrar el
    valor actual en su propia media. Etiqueta SOLO para evaluación."""
    geo = ["provincia", "departamento"] if "provincia" in panel.columns else ["departamento"]
    df = panel.sort_values(geo + ["cultivo", "campania_inicio"]).copy()
    grp = df.groupby(geo + ["cultivo"])["rinde_kgha"]
    roll_mean = grp.transform(
        lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=3).mean()
    )
    roll_std = grp.transform(
        lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=3).std()
    )
    df["z_rinde"] = (df["rinde_kgha"] - roll_mean) / roll_std.replace(0, np.nan)
    df["anomalia"] = (df["z_rinde"] < cfg.z_thresh).astype(int)
    return df


def add_agro_features(df: pd.DataFrame, cultivo: str) -> tuple:
    """Deriva features agronómicas centradas en la ventana crítica del cultivo,
    a partir de las columnas mensuales ya presentes (no requiere datos nuevos).

    Apunta a la señal que `stratified_recall` mostró que domina la detección
    (estrés hídrico/térmico en floración-llenado). Devuelve (df_con_cols,
    nombres_nuevos). Todas se normalizan luego por depto como el resto.

    - agro_precip_crit  : precip total (NASA POWER) en ventana crítica.
    - agro_tmax_crit    : t2m_max media en ventana crítica (estrés térmico).
    - agro_thermamp_crit: amplitud térmica media (t2m_max − t2m_min).
    - agro_waterbal_crit: balance hídrico Σ(precip − PET) con PET Hargreaves
      proxy = 0.0023·Ra·(Tmean+17.8)·√(Tmax−Tmin), Ra≈allsky_sfc_sw_dwn. Como
      la feature se z-scorea por depto, las unidades/constantes no importan,
      solo la variación relativa.
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
    etiqueta proxy para evaluación y visualización."""

    cultivo: str
    feature_cols: List[str]

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    meta_train: pd.DataFrame  # train NORMAL (lo que ve el modelo)
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame

    y_val: np.ndarray
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
    como fallback: así NO se cae la fila (antes `/0 → NaN → dropna` borraba el
    34% de los departamentos sin heladas) Y se preserva la señal — un depto que
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
                       cfg: ExperimentConfig) -> CropDataset:
    """Construye el CropDataset para un cultivo: split temporal, filtrado de
    train a filas normales, y normalización por departamento sin leakage."""
    feats = build_feature_list(panel_z, cfg.use_ndvi, cfg.use_era5_features)
    df = panel_z[panel_z["cultivo"] == cultivo].copy()

    # --- Features agronómicas de dominio (ventana crítica, per-cultivo) ---
    if cfg.use_agro_features:
        df, agro_cols = add_agro_features(df, cultivo)
        feats = feats + agro_cols

    # --- Split temporal ---
    m_train = df["campania_inicio"] <= cfg.train_end
    if cfg.train_start is not None:
        m_train &= df["campania_inicio"] >= cfg.train_start
    m_val = (df["campania_inicio"] >= cfg.val_start) & (df["campania_inicio"] <= cfg.val_end)
    m_test = df["campania_inicio"] >= cfg.test_start
    df_train, df_val, df_test = df[m_train].copy(), df[m_val].copy(), df[m_test].copy()

    # --- Train normal: excluir años problemáticos + filas anómalas ---
    m_excl = df_train["campania_inicio"].isin(cfg.excluded_train_years)
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
    df_va_n = _normalize_per_depto(df_val, stats, feats, geo, global_std).dropna(subset=feats)
    df_te_n = _normalize_per_depto(df_test, stats, feats, geo, global_std).dropna(subset=feats)

    meta_cols = (["provincia"] if "provincia" in df.columns else []) + _META_COLS

    return CropDataset(
        cultivo=cultivo,
        feature_cols=feats,
        X_train=df_tr_n[feats].values,
        X_val=df_va_n[feats].values,
        X_test=df_te_n[feats].values,
        meta_train=df_tr_n[meta_cols].reset_index(drop=True),
        meta_val=df_va_n[meta_cols].reset_index(drop=True),
        meta_test=df_te_n[meta_cols].reset_index(drop=True),
        y_val=df_va_n["anomalia"].values,
        y_test=df_te_n["anomalia"].values,
        n_excluded_year=int(m_excl.sum()),
        n_excluded_anom=int((~m_excl & m_anom).sum()),
    )


def prepare(cfg: ExperimentConfig):
    """Carga panel + etiqueta. Devuelve (panel_z, feature_cols)."""
    panel = load_panel(cfg)
    panel_z = compute_z_rinde(panel, cfg)
    return panel_z, build_feature_list(panel_z, cfg.use_ndvi, cfg.use_era5_features)
