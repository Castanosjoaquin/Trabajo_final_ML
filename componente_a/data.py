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

from .config import ExperimentConfig, CLIM_PREFIXES, NDVI_COLS, MESES


# --- Columnas clave que deben existir en el panel ---
_REQUIRED_BASE = ["cultivo", "campania", "campania_inicio", "departamento", "rinde_kgha"]


def build_feature_list(panel: pd.DataFrame, use_ndvi: bool) -> List[str]:
    """Lista explícita de columnas que entran en X (promedios mensuales Sep–Mar
    de las variables climáticas). Solo incluye las que realmente existen."""
    cols: List[str] = []
    for pfx in CLIM_PREFIXES:
        for mes in MESES:
            col = f"{pfx}_{mes}"
            if col in panel.columns:
                cols.append(col)
    if use_ndvi:
        cols += [c for c in NDVI_COLS if c in panel.columns]
    return cols


def load_panel(cfg: ExperimentConfig) -> pd.DataFrame:
    """Carga el panel y valida columnas. Falla con mensaje claro si falta algo."""
    if not os.path.exists(cfg.panel_path):
        raise FileNotFoundError(f"Panel no encontrado: {cfg.panel_path}")
    panel = pd.read_parquet(cfg.panel_path)

    feats = build_feature_list(panel, cfg.use_ndvi)
    missing = [c for c in (_REQUIRED_BASE + feats) if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en el panel: {missing}\n"
            f"Columnas disponibles: {list(panel.columns)}"
        )
    return panel


def compute_z_rinde(panel: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """z-score del rinde vs media móvil por departamento Y cultivo (etiqueta
    proxy). Se agrupa por [departamento, cultivo] para no mezclar las series
    de rinde de soja (~2700 kg/ha) y maíz (~6700 kg/ha) dentro del mismo
    departamento. Usa shift(1) para no filtrar el valor actual en su propia
    media. La etiqueta es SOLO para evaluación; el detector nunca la ve."""
    df = panel.sort_values(["departamento", "cultivo", "campania_inicio"]).copy()
    grp = df.groupby(["departamento", "cultivo"])["rinde_kgha"]
    roll_mean = grp.transform(
        lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=3).mean()
    )
    roll_std = grp.transform(
        lambda s: s.shift(1).rolling(cfg.rolling_window, min_periods=3).std()
    )
    df["z_rinde"] = (df["rinde_kgha"] - roll_mean) / roll_std.replace(0, np.nan)
    df["anomalia"] = (df["z_rinde"] < cfg.z_thresh).astype(int)
    return df


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
                         feature_cols: List[str]) -> pd.DataFrame:
    """Aplica z-score por departamento usando estadísticas ya calculadas
    (de train_normal). No recomputa nada → sin leakage."""
    df_out = df_in.copy()
    for col in feature_cols:
        merged = df_out[["departamento"]].merge(
            stats[[f"{col}_mean", f"{col}_std"]].reset_index(),
            on="departamento", how="left",
        )
        std_vals = merged[f"{col}_std"].replace(0, np.nan).values
        df_out[col] = (df_out[col].values - merged[f"{col}_mean"].values) / std_vals
    return df_out


def build_crop_dataset(panel_z: pd.DataFrame, cultivo: str,
                       cfg: ExperimentConfig) -> CropDataset:
    """Construye el CropDataset para un cultivo: split temporal, filtrado de
    train a filas normales, y normalización por departamento sin leakage."""
    feats = build_feature_list(panel_z, cfg.use_ndvi)
    df = panel_z[panel_z["cultivo"] == cultivo].copy()

    # --- Split temporal ---
    m_train = df["campania_inicio"] <= cfg.train_end
    m_val = (df["campania_inicio"] >= cfg.val_start) & (df["campania_inicio"] <= cfg.val_end)
    m_test = df["campania_inicio"] >= cfg.test_start
    df_train, df_val, df_test = df[m_train].copy(), df[m_val].copy(), df[m_test].copy()

    # --- Train normal: excluir años problemáticos + filas anómalas ---
    m_excl = df_train["campania_inicio"].isin(cfg.excluded_train_years)
    m_anom = df_train["anomalia"] == 1
    df_train_normal = df_train[~m_excl & ~m_anom].copy()

    # --- Normalización por depto (fit en train_normal) ---
    stats = df_train_normal.groupby("departamento")[feats].agg(["mean", "std"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]

    df_tr_n = _normalize_per_depto(df_train_normal, stats, feats).dropna(subset=feats)
    df_va_n = _normalize_per_depto(df_val, stats, feats).dropna(subset=feats)
    df_te_n = _normalize_per_depto(df_test, stats, feats).dropna(subset=feats)

    return CropDataset(
        cultivo=cultivo,
        feature_cols=feats,
        X_train=df_tr_n[feats].values,
        X_val=df_va_n[feats].values,
        X_test=df_te_n[feats].values,
        meta_train=df_tr_n[_META_COLS].reset_index(drop=True),
        meta_val=df_va_n[_META_COLS].reset_index(drop=True),
        meta_test=df_te_n[_META_COLS].reset_index(drop=True),
        y_val=df_va_n["anomalia"].values,
        y_test=df_te_n["anomalia"].values,
        n_excluded_year=int(m_excl.sum()),
        n_excluded_anom=int((~m_excl & m_anom).sum()),
    )


def prepare(cfg: ExperimentConfig):
    """Carga panel + etiqueta. Devuelve (panel_z, feature_cols)."""
    panel = load_panel(cfg)
    panel_z = compute_z_rinde(panel, cfg)
    return panel_z, build_feature_list(panel_z, cfg.use_ndvi)
